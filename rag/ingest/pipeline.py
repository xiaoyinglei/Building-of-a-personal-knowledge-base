from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from rag.assembly import (
    CapabilityBundle,
    ChatCapabilityBinding,
    EmbeddingCapabilityBinding,
)
from rag.ingest.chunking import ChunkingService, DocumentProcessingService, TOCService
from rag.ingest.extract import (
    EntityRelationExtractionResult,
    EntityRelationExtractor,
    EntityRelationMerger,
    MergedEntity,
    MergedGraph,
    MergedRelation,
    PromptedEntityRelationExtractor,
    choose_preferred_label,
    normalize_entity_key,
)
from rag.ingest.parser import (
    DoclingParserRepo,
    HttpWebFetchRepo,
    ImageParserRepo,
    MarkdownParserRepo,
    ParsedDocument,
    PDFParserRepo,
    PlainTextParserRepo,
    WebParserRepo,
    create_default_ocr_repo,
)
from rag.ingest.parsers.util import normalize_whitespace
from rag.ingest.policy import PolicyResolutionService
from rag.schema.core import (
    Chunk,
    ChunkRole,
    Document,
    DocumentProcessingPackage,
    GraphEdge,
    GraphNode,
    Segment,
    Source,
    SourceType,
)
from rag.schema.runtime import (
    AccessPolicy,
    CacheEntry,
    CacheRepo,
    DocumentPipelineStage,
    DocumentProcessingStatus,
    DocumentStatusRecord,
    FullTextSearchRepo,
    GraphRepo,
    MetadataRepo,
    ObjectStore,
    OcrVisionRepo,
    VectorRepo,
    VisualDescriptionRepo,
    WebFetchRepo,
)
from rag.storage import StorageConfig
from rag.storage.cache import CacheStore
from rag.storage.graph import GraphStore
from rag.storage.metadata import ChunkStore, DocumentStore, StatusStore
from rag.storage.vector import VectorStore

_BENCHMARK_METADATA_KEYS = (
    "benchmark",
    "dataset",
    "benchmark_dataset",
    "benchmark_doc_id",
    "parent_doc_id",
)


def _benchmark_metadata(mapping: dict[str, str] | None) -> dict[str, str]:
    if not mapping:
        return {}
    metadata = {key: mapping[key] for key in _BENCHMARK_METADATA_KEYS if key in mapping and mapping[key]}
    benchmark_dataset = metadata.get("benchmark_dataset") or metadata.get("dataset")
    benchmark_doc_id = metadata.get("benchmark_doc_id") or metadata.get("parent_doc_id")
    if benchmark_dataset:
        metadata["benchmark_dataset"] = benchmark_dataset
    if benchmark_doc_id:
        metadata["benchmark_doc_id"] = benchmark_doc_id
        metadata.setdefault("parent_doc_id", benchmark_doc_id)
    return metadata


@dataclass(frozen=True, slots=True)
class IngestRequest:
    location: str
    source_type: SourceType | str
    owner: str
    access_policy: AccessPolicy | None = None
    title: str | None = None
    content_text: str | None = None
    raw_bytes: bytes | None = None
    file_path: Path | None = None
    parsed_document: ParsedDocument | None = None


@dataclass(frozen=True, slots=True)
class DirectContentItem:
    location: str
    source_type: SourceType | str
    content: str | bytes | Path
    owner: str = "user"
    access_policy: AccessPolicy | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class IngestPipelineResult:
    source: Source
    document: Document
    segments: list[Segment]
    chunks: list[Chunk]
    is_duplicate: bool
    content_hash: str
    visible_text: str
    visual_semantics: str | None = None
    processing: DocumentProcessingPackage | None = None
    entity_count: int = 0
    relation_count: int = 0
    status: str = DocumentProcessingStatus.READY.value

    @property
    def document_id(self) -> str:
        return self.document.doc_id

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True, slots=True)
class BatchIngestError:
    index: int
    location: str
    error: str


@dataclass(frozen=True, slots=True)
class BatchIngestResult:
    results: list[IngestPipelineResult]
    errors: list[BatchIngestError]

    @property
    def success_count(self) -> int:
        return len(self.results)

    @property
    def failure_count(self) -> int:
        return len(self.errors)


@dataclass(frozen=True, slots=True)
class _PendingBatchIngest:
    index: int
    location: str
    source: Source
    document: Document
    segments: list[Segment]
    stored_chunks: list[Chunk]
    indexed_chunks: list[Chunk]
    processing: DocumentProcessingPackage | None
    content_hash: str
    visible_text: str
    visual_semantics: str | None = None


@dataclass(slots=True)
class IngestPipeline:
    documents: DocumentStore
    chunks: ChunkStore
    vectors: VectorStore
    graph: GraphStore
    status: StatusStore
    cache: CacheStore
    fts_repo: FullTextSearchRepo
    object_store: ObjectStore | None
    markdown_parser: MarkdownParserRepo
    pdf_parser: PDFParserRepo
    plain_text_parser: PlainTextParserRepo
    image_parser: ImageParserRepo
    web_parser: WebParserRepo
    web_fetch_repo: WebFetchRepo
    docling_parser: DoclingParserRepo
    policy_resolution_service: PolicyResolutionService
    toc_service: TOCService
    chunking_service: ChunkingService
    document_processing_service: DocumentProcessingService
    embedding_capabilities: tuple[EmbeddingCapabilityBinding, ...] = ()
    chat_capabilities: tuple[ChatCapabilityBinding, ...] = ()
    extractor: EntityRelationExtractor | None = None
    merger: EntityRelationMerger | None = None

    def __post_init__(self) -> None:
        if self.extractor is None:
            provider = self.chat_capabilities[0] if self.chat_capabilities else None
            self.extractor = PromptedEntityRelationExtractor(model_provider=provider)
        if self.merger is None:
            self.merger = EntityRelationMerger()

    def run(self, request: IngestRequest) -> IngestPipelineResult:
        source_type = SourceType(request.source_type)
        raw_bytes = self._resolve_raw_bytes(request=request, source_type=source_type)
        content_hash = sha256(raw_bytes).hexdigest()
        try:
            parsed = self._resolve_parsed_document(
                request=request,
                source_type=source_type,
                raw_bytes=raw_bytes,
            )
        except Exception as exc:
            failed_source_id = self._deterministic_id(request.location, content_hash, "source")
            failed_doc_id = self._deterministic_id(request.location, content_hash, "document")
            self._save_status(
                doc_id=failed_doc_id,
                source_id=failed_source_id,
                location=request.location,
                content_hash=content_hash,
                status=DocumentProcessingStatus.FAILED,
                stage=DocumentPipelineStage.PARSE,
                error_message=str(exc),
            )
            raise
        return self.ingest_parsed_document(
            location=request.location,
            raw_bytes=raw_bytes,
            parsed=parsed,
            owner=request.owner,
            access_policy=request.access_policy,
        )

    def run_many(
        self,
        requests: Sequence[IngestRequest],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        if requests and all(request.parsed_document is not None for request in requests):
            return self._run_many_preparsed(requests, continue_on_error=continue_on_error)
        results: list[IngestPipelineResult] = []
        errors: list[BatchIngestError] = []
        for index, request in enumerate(requests):
            try:
                results.append(self.run(request))
            except Exception as exc:
                if not continue_on_error:
                    raise
                errors.append(BatchIngestError(index=index, location=request.location, error=str(exc)))
        return BatchIngestResult(results=results, errors=errors)

    def _run_many_preparsed(
        self,
        requests: Sequence[IngestRequest],
        *,
        continue_on_error: bool,
    ) -> BatchIngestResult:
        pending: list[_PendingBatchIngest] = []
        results_by_index: dict[int, IngestPipelineResult] = {}
        errors: list[BatchIngestError] = []

        for index, request in enumerate(requests):
            try:
                staged = self._stage_preparsed_request(index=index, request=request)
                if isinstance(staged, IngestPipelineResult):
                    results_by_index[index] = staged
                else:
                    pending.append(staged)
            except Exception as exc:
                if not continue_on_error:
                    raise
                errors.append(BatchIngestError(index=index, location=request.location, error=str(exc)))

        self._index_pending_batch(pending)

        for item in pending:
            try:
                self._save_status(
                    doc_id=item.document.doc_id,
                    source_id=item.source.source_id,
                    location=item.location,
                    content_hash=item.content_hash,
                    status=DocumentProcessingStatus.PROCESSING,
                    stage=DocumentPipelineStage.EXTRACT,
                )
                entity_count, relation_count = self._extract_and_persist_graph(
                    source=item.source,
                    document=item.document,
                    chunks=item.indexed_chunks,
                )
                final_status = self._save_status(
                    doc_id=item.document.doc_id,
                    source_id=item.source.source_id,
                    location=item.location,
                    content_hash=item.content_hash,
                    status=DocumentProcessingStatus.READY,
                    stage=DocumentPipelineStage.INDEX,
                )
                results_by_index[item.index] = IngestPipelineResult(
                    source=item.source,
                    document=item.document,
                    segments=item.segments,
                    chunks=item.indexed_chunks,
                    is_duplicate=False,
                    content_hash=item.content_hash,
                    visible_text=item.visible_text,
                    visual_semantics=item.visual_semantics,
                    processing=item.processing,
                    entity_count=entity_count,
                    relation_count=relation_count,
                    status=final_status.status.value,
                )
            except Exception as exc:
                self._save_status(
                    doc_id=item.document.doc_id,
                    source_id=item.source.source_id,
                    location=item.location,
                    content_hash=item.content_hash,
                    status=DocumentProcessingStatus.FAILED,
                    stage=DocumentPipelineStage.EXTRACT,
                    error_message=str(exc),
                )
                if not continue_on_error:
                    raise
                errors.append(BatchIngestError(index=item.index, location=item.location, error=str(exc)))

        ordered_results = [results_by_index[index] for index in sorted(results_by_index)]
        return BatchIngestResult(results=ordered_results, errors=errors)

    def run_content_list(
        self,
        items: Sequence[DirectContentItem],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        results: list[IngestPipelineResult] = []
        errors: list[BatchIngestError] = []
        for index, item in enumerate(items):
            try:
                results.append(self.run(self._ingest_request_from_content_item(item)))
            except Exception as exc:
                if not continue_on_error:
                    raise
                errors.append(BatchIngestError(index=index, location=item.location, error=str(exc)))
        return BatchIngestResult(results=results, errors=errors)

    def ingest_parsed_document(
        self,
        *,
        location: str,
        raw_bytes: bytes,
        parsed: ParsedDocument,
        owner: str,
        access_policy: AccessPolicy | None,
    ) -> IngestPipelineResult:
        normalized_policy = access_policy or AccessPolicy.default()
        content_hash = sha256(raw_bytes).hexdigest()
        existing_source = self.documents.get_source_by_location_and_hash(location, content_hash)
        if existing_source is not None:
            existing_document = self.documents.get_active_document_by_location_and_hash(location, content_hash)
            if existing_document is None:
                existing_document = self.documents.get_latest_document_for_location(location)
            if existing_document is None:
                raise RuntimeError("duplicate source exists without an active document")
            existing_segments = self.documents.list_segments(existing_document.doc_id)
            stored_existing_chunks = self.chunks.list_by_document(existing_document.doc_id)
            existing_chunks = self._retrievable_chunks(stored_existing_chunks)
            self._save_status(
                doc_id=existing_document.doc_id,
                source_id=existing_source.source_id,
                location=location,
                content_hash=content_hash,
                status=DocumentProcessingStatus.PROCESSING,
                stage=DocumentPipelineStage.INDEX,
            )
            self._repair_duplicate_indexes(
                source=existing_source,
                document=existing_document,
                segments=existing_segments,
                chunks=existing_chunks,
            )
            entity_count, relation_count = self._extract_and_persist_graph(
                source=existing_source,
                document=existing_document,
                chunks=existing_chunks,
            )
            final_status = self._save_status(
                doc_id=existing_document.doc_id,
                source_id=existing_source.source_id,
                location=location,
                content_hash=content_hash,
                status=DocumentProcessingStatus.READY,
                stage=DocumentPipelineStage.INDEX,
            )
            return IngestPipelineResult(
                source=existing_source,
                document=existing_document,
                segments=existing_segments,
                chunks=existing_chunks,
                is_duplicate=True,
                content_hash=content_hash,
                visible_text=parsed.visible_text,
                visual_semantics=parsed.visual_semantics,
                processing=None,
                entity_count=entity_count,
                relation_count=relation_count,
                status=final_status.status.value,
            )

        latest_source = self.documents.get_latest_source_for_location(location)
        ingest_version = 1 if latest_source is None else latest_source.ingest_version + 1
        source_id = self._deterministic_id(location, content_hash, "source")
        source_metadata = {"source_type": parsed.source_type.value}
        if self.object_store is not None:
            object_key = self.object_store.put_bytes(raw_bytes, suffix=self._suffix_for(parsed.source_type))
            source_metadata["object_key"] = object_key
        source = Source(
            source_id=source_id,
            source_type=parsed.source_type,
            location=location,
            owner=owner,
            content_hash=content_hash,
            effective_access_policy=normalized_policy,
            ingest_version=ingest_version,
            metadata=source_metadata,
        )
        self.documents.save_source(source)
        self.documents.deactivate_documents_for_location(location)

        document = Document(
            doc_id=self._deterministic_id(source_id, parsed.title, "document"),
            source_id=source.source_id,
            doc_type=parsed.doc_type,
            title=parsed.title,
            authors=list(parsed.authors or [owner]),
            created_at=datetime.now(UTC),
            language=parsed.language,
            effective_access_policy=normalized_policy,
            metadata={**parsed.metadata, "location": location, "content_hash": content_hash},
        )
        self.documents.save_document(document, location=location, content_hash=content_hash)
        stage = DocumentPipelineStage.PARSE
        self._save_status(
            doc_id=document.doc_id,
            source_id=source.source_id,
            location=location,
            content_hash=content_hash,
            status=DocumentProcessingStatus.PROCESSING,
            stage=stage,
        )

        try:
            stage = DocumentPipelineStage.CHUNK
            self._save_status(
                doc_id=document.doc_id,
                source_id=source.source_id,
                location=location,
                content_hash=content_hash,
                status=DocumentProcessingStatus.PROCESSING,
                stage=stage,
            )
            segments, stored_chunks, indexed_chunks, processing = self._prepare_chunks(
                location=location,
                source=source,
                document=document,
                parsed=parsed,
                access_policy=normalized_policy,
            )

            stage = DocumentPipelineStage.PERSIST
            self._save_status(
                doc_id=document.doc_id,
                source_id=source.source_id,
                location=location,
                content_hash=content_hash,
                status=DocumentProcessingStatus.PROCESSING,
                stage=stage,
            )
            for segment in segments:
                self.documents.save_segment(segment)
            self.chunks.save_many(stored_chunks)
            self._index_chunks(source=source, document=document, segments=segments, chunks=indexed_chunks)

            stage = DocumentPipelineStage.EXTRACT
            self._save_status(
                doc_id=document.doc_id,
                source_id=source.source_id,
                location=location,
                content_hash=content_hash,
                status=DocumentProcessingStatus.PROCESSING,
                stage=stage,
            )
            entity_count, relation_count = self._extract_and_persist_graph(
                source=source,
                document=document,
                chunks=indexed_chunks,
            )

            stage = DocumentPipelineStage.INDEX
            final_status = self._save_status(
                doc_id=document.doc_id,
                source_id=source.source_id,
                location=location,
                content_hash=content_hash,
                status=DocumentProcessingStatus.READY,
                stage=stage,
            )
        except Exception as exc:
            self._save_status(
                doc_id=document.doc_id,
                source_id=source.source_id,
                location=location,
                content_hash=content_hash,
                status=DocumentProcessingStatus.FAILED,
                stage=stage,
                error_message=str(exc),
            )
            raise

        return IngestPipelineResult(
            source=source,
            document=document,
            segments=segments,
            chunks=indexed_chunks,
            is_duplicate=False,
            content_hash=content_hash,
            visible_text=parsed.visible_text,
            visual_semantics=parsed.visual_semantics,
            processing=processing,
            entity_count=entity_count,
            relation_count=relation_count,
            status=final_status.status.value,
        )

    def _stage_preparsed_request(
        self,
        *,
        index: int,
        request: IngestRequest,
    ) -> _PendingBatchIngest | IngestPipelineResult:
        source_type = SourceType(request.source_type)
        raw_bytes = self._resolve_raw_bytes(request=request, source_type=source_type)
        parsed = self._resolve_parsed_document(request=request, source_type=source_type, raw_bytes=raw_bytes)
        if request.parsed_document is None:
            raise ValueError("_stage_preparsed_request requires parsed_document input")
        content_hash = sha256(raw_bytes).hexdigest()
        existing_source = self.documents.get_source_by_location_and_hash(request.location, content_hash)
        if existing_source is not None:
            return self.ingest_parsed_document(
                location=request.location,
                raw_bytes=raw_bytes,
                parsed=parsed,
                owner=request.owner,
                access_policy=request.access_policy,
            )

        normalized_policy = request.access_policy or AccessPolicy.default()
        latest_source = self.documents.get_latest_source_for_location(request.location)
        ingest_version = 1 if latest_source is None else latest_source.ingest_version + 1
        source_id = self._deterministic_id(request.location, content_hash, "source")
        source_metadata = {"source_type": parsed.source_type.value}
        if self.object_store is not None:
            object_key = self.object_store.put_bytes(raw_bytes, suffix=self._suffix_for(parsed.source_type))
            source_metadata["object_key"] = object_key
        source = Source(
            source_id=source_id,
            source_type=parsed.source_type,
            location=request.location,
            owner=request.owner,
            content_hash=content_hash,
            effective_access_policy=normalized_policy,
            ingest_version=ingest_version,
            metadata=source_metadata,
        )
        self.documents.save_source(source)
        self.documents.deactivate_documents_for_location(request.location)

        document = Document(
            doc_id=self._deterministic_id(source.source_id, parsed.title, "document"),
            source_id=source.source_id,
            doc_type=parsed.doc_type,
            title=parsed.title,
            authors=list(parsed.authors or [request.owner]),
            created_at=datetime.now(UTC),
            language=parsed.language,
            effective_access_policy=normalized_policy,
            metadata={**parsed.metadata, "location": request.location, "content_hash": content_hash},
        )
        self.documents.save_document(document, location=request.location, content_hash=content_hash)
        self._save_status(
            doc_id=document.doc_id,
            source_id=source.source_id,
            location=request.location,
            content_hash=content_hash,
            status=DocumentProcessingStatus.PROCESSING,
            stage=DocumentPipelineStage.CHUNK,
        )
        segments, stored_chunks, indexed_chunks, processing = self._prepare_chunks(
            location=request.location,
            source=source,
            document=document,
            parsed=parsed,
            access_policy=normalized_policy,
        )
        self._save_status(
            doc_id=document.doc_id,
            source_id=source.source_id,
            location=request.location,
            content_hash=content_hash,
            status=DocumentProcessingStatus.PROCESSING,
            stage=DocumentPipelineStage.PERSIST,
        )
        for segment in segments:
            self.documents.save_segment(segment)
        self.chunks.save_many(stored_chunks)
        return _PendingBatchIngest(
            index=index,
            location=request.location,
            source=source,
            document=document,
            segments=segments,
            stored_chunks=stored_chunks,
            indexed_chunks=indexed_chunks,
            processing=processing,
            content_hash=content_hash,
            visible_text=parsed.visible_text,
            visual_semantics=parsed.visual_semantics,
        )

    def _index_pending_batch(self, pending: Sequence[_PendingBatchIngest]) -> None:
        if not pending:
            return
        chunk_records: list[tuple[Document, Segment, Chunk]] = []
        for item in pending:
            segments_by_id = {segment.segment_id: segment for segment in item.segments}
            for chunk in item.indexed_chunks:
                segment = segments_by_id.get(chunk.segment_id)
                toc_path = [] if segment is None else list(segment.toc_path)
                self.fts_repo.index_chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_id=item.source.source_id,
                    title=item.document.title,
                    toc_path=toc_path,
                    text=chunk.text,
                )
                if segment is not None:
                    chunk_records.append((item.document, segment, chunk))
            for segment in item.segments:
                segment_chunks = [chunk for chunk in item.indexed_chunks if chunk.segment_id == segment.segment_id]
                self._save_section_graph(
                    source=item.source,
                    document=item.document,
                    segment=segment,
                    chunks=segment_chunks,
                )

        if not chunk_records:
            return
        texts = [chunk.text for _document, _segment, chunk in chunk_records]
        for binding in self.embedding_capabilities:
            vectors = self._embed_texts(binding, texts)
            if vectors is None:
                continue
            for (document, segment, chunk), vector in zip(chunk_records, vectors, strict=True):
                self.vectors.upsert_chunk(
                    chunk.chunk_id,
                    vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "segment_id": segment.segment_id,
                        "text": chunk.text,
                        **_benchmark_metadata(chunk.metadata),
                    },
                    embedding_space=binding.space,
                )

    def _prepare_chunks(
        self,
        *,
        location: str,
        source: Source,
        document: Document,
        parsed: ParsedDocument,
        access_policy: AccessPolicy,
    ) -> tuple[list[Segment], list[Chunk], list[Chunk], DocumentProcessingPackage | None]:
        if parsed.source_type in {SourceType.PDF, SourceType.MARKDOWN, SourceType.DOCX, SourceType.IMAGE}:
            prepared = self.document_processing_service.build(
                location=location,
                source=source,
                document=document,
                parsed=parsed,
                access_policy=access_policy,
            )
            return prepared.segments, prepared.stored_chunks, prepared.indexed_chunks, prepared.package

        segments: list[Segment] = []
        indexed_chunks: list[Chunk] = []
        path_to_segment_id: dict[tuple[str, ...], str] = {}
        for section in parsed.sections:
            normalized_path = self.toc_service.normalize_path(section.toc_path)
            parent_path = tuple(normalized_path[:-1])
            parent_segment_id = path_to_segment_id.get(parent_path)
            anchor = self.toc_service.stable_anchor(
                location,
                normalized_path,
                section.order_index,
                page_range=section.page_range,
                anchor_hint=section.anchor_hint,
            )
            segment = Segment(
                segment_id=self._deterministic_id(source.source_id, anchor, "segment"),
                doc_id=document.doc_id,
                parent_segment_id=parent_segment_id,
                toc_path=list(normalized_path),
                heading_level=section.heading_level,
                page_range=section.page_range,
                order_index=section.order_index,
                anchor=anchor,
                visible_text=section.text or None,
                visual_semantics=section.metadata.get("visual_semantics"),
                metadata=section.metadata | _benchmark_metadata(document.metadata) | {"location": location},
            )
            segments.append(segment)
            path_to_segment_id[tuple(normalized_path)] = segment.segment_id
            chunk_list = self.chunking_service.chunk_section(
                location=location,
                doc_id=document.doc_id,
                segment=segment,
                text=section.text,
                access_policy=self.policy_resolution_service.resolve_effective_access_policy(
                    source_policy=access_policy,
                    chunk_policy=access_policy,
                ),
                document_metadata=_benchmark_metadata(document.metadata),
            )
            indexed_chunks.extend(chunk_list)

        return segments, indexed_chunks, indexed_chunks, None

    def _index_chunks(
        self,
        *,
        source: Source,
        document: Document,
        segments: list[Segment],
        chunks: list[Chunk],
    ) -> None:
        segments_by_id = {segment.segment_id: segment for segment in segments}
        for chunk in chunks:
            segment = segments_by_id.get(chunk.segment_id)
            toc_path = [] if segment is None else list(segment.toc_path)
            self.fts_repo.index_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_id=source.source_id,
                title=document.title,
                toc_path=toc_path,
                text=chunk.text,
            )
        for segment in segments:
            segment_chunks = [chunk for chunk in chunks if chunk.segment_id == segment.segment_id]
            self._index_chunk_vectors(document=document, segment=segment, chunks=segment_chunks)
            self._save_section_graph(source=source, document=document, segment=segment, chunks=segment_chunks)

    def _extract_and_persist_graph(
        self,
        *,
        source: Source,
        document: Document,
        chunks: list[Chunk],
    ) -> tuple[int, int]:
        if not chunks or self.extractor is None or self.merger is None:
            return 0, 0
        extraction = self._extract_entities_and_relations(document=document, chunks=chunks)
        merged = self._resolve_entities_against_existing_graph(
            document=document,
            merged=self.merger.merge(document=document, extraction=extraction),
        )

        for entity in merged.entities:
            node = GraphNode(
                node_id=entity.node_id,
                node_type="entity",
                label=entity.label,
                metadata=entity.metadata | {"source_id": source.source_id},
            )
            self.graph.save_node(node, evidence_chunk_ids=entity.evidence_chunk_ids)
        for relation in merged.relations:
            edge = GraphEdge(
                edge_id=relation.edge_id,
                from_node_id=relation.from_node_id,
                to_node_id=relation.to_node_id,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                evidence_chunk_ids=relation.evidence_chunk_ids,
                metadata=relation.metadata | {"source_id": source.source_id},
            )
            self.graph.save_edge(edge)

        self._persist_multimodal_graph_overlay(
            source=source,
            document=document,
            chunks=chunks,
            entities=merged.entities,
        )
        self._index_multimodal_vectors(source=source, document=document, chunks=chunks)
        self._index_entity_vectors(source=source, document=document, entities=merged.entities)
        self._index_relation_vectors(source=source, document=document, relations=merged.relations)
        return len(merged.entities), len(merged.relations)

    def _persist_multimodal_graph_overlay(
        self,
        *,
        source: Source,
        document: Document,
        chunks: list[Chunk],
        entities: Sequence[MergedEntity],
    ) -> None:
        special_chunks = [
            chunk for chunk in chunks if chunk.chunk_role is ChunkRole.SPECIAL and chunk.special_chunk_type
        ]
        if not special_chunks:
            return

        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        entity_segments: dict[str, set[str]] = {}
        entity_aliases: dict[str, set[str]] = {}
        for entity in entities:
            entity_segments[entity.node_id] = {
                chunk_by_id[chunk_id].segment_id for chunk_id in entity.evidence_chunk_ids if chunk_id in chunk_by_id
            }
            aliases = {
                normalize_whitespace(alias).lower()
                for alias in entity.metadata.get("aliases", "").split("||")
                if normalize_whitespace(alias)
            }
            aliases.add(normalize_whitespace(entity.label).lower())
            entity_aliases[entity.node_id] = aliases

        for chunk in special_chunks:
            node_id = self._deterministic_id(chunk.chunk_id, "multimodal")
            label = normalize_whitespace(
                f"{(chunk.special_chunk_type or 'special').replace('_', ' ').title()}: {chunk.text}"
            )[:240]
            node = GraphNode(
                node_id=node_id,
                node_type=chunk.special_chunk_type or "multimodal",
                label=label,
                metadata={
                    "doc_id": document.doc_id,
                    "source_id": source.source_id,
                    "segment_id": chunk.segment_id,
                    "chunk_id": chunk.chunk_id,
                    "special_chunk_type": chunk.special_chunk_type or "special",
                    **{key: value for key, value in chunk.metadata.items() if value},
                },
            )
            self.graph.save_node(node, evidence_chunk_ids=[chunk.chunk_id])
            self.graph.save_edge(
                GraphEdge(
                    edge_id=self._deterministic_id(chunk.segment_id, node_id, "contains_special", "edge"),
                    from_node_id=chunk.segment_id,
                    to_node_id=node_id,
                    relation_type="contains_special",
                    confidence=1.0,
                    evidence_chunk_ids=[chunk.chunk_id],
                    metadata={"doc_id": document.doc_id, "source_id": source.source_id},
                )
            )

            normalized_special_text = normalize_whitespace(chunk.text).lower()
            for entity in entities:
                same_segment = chunk.segment_id in entity_segments.get(entity.node_id, set())
                mentioned = any(
                    alias and alias in normalized_special_text for alias in entity_aliases.get(entity.node_id, set())
                )
                if not same_segment and not mentioned:
                    continue
                relation_type = self._multimodal_relation_type(chunk.special_chunk_type)
                self.graph.save_edge(
                    GraphEdge(
                        edge_id=self._deterministic_id(entity.node_id, node_id, relation_type, "edge"),
                        from_node_id=entity.node_id,
                        to_node_id=node_id,
                        relation_type=relation_type,
                        confidence=1.0,
                        evidence_chunk_ids=[chunk.chunk_id],
                        metadata={
                            "doc_id": document.doc_id,
                            "source_id": source.source_id,
                            "from_label": entity.label,
                            "to_label": label,
                            "association_basis": (
                                "segment_and_alias"
                                if same_segment and mentioned
                                else "alias_mention"
                                if mentioned
                                else "shared_segment"
                            ),
                        },
                    )
                )

    def _resolve_entities_against_existing_graph(
        self,
        *,
        document: Document,
        merged: MergedGraph,
    ) -> MergedGraph:
        if not merged.entities:
            return merged

        resolved_entities: dict[str, MergedEntity] = {}
        rewritten_node_ids: dict[str, str] = {}
        for entity in merged.entities:
            matched = self._find_existing_entity_match(entity)
            resolved_node_id = entity.node_id if matched is None else matched.node_id
            resolved_key = entity.key
            if matched is not None:
                resolved_key = matched.metadata.get("entity_key") or normalize_entity_key(matched.label) or entity.key

            current = resolved_entities.get(resolved_node_id)
            alias_values = self._merge_alias_values(
                []
                if matched is None
                else self._entity_alias_values(matched.label, matched.metadata.get("aliases", "")),
                []
                if current is None
                else self._entity_alias_values(current.label, current.metadata.get("aliases", "")),
                self._entity_alias_values(entity.label, entity.metadata.get("aliases", "")),
            )
            preferred_label = (
                choose_preferred_label(
                    [
                        *([] if matched is None else [matched.label]),
                        *([] if current is None else [current.label]),
                        entity.label,
                        *alias_values,
                    ]
                )
                or entity.label
            )
            evidence_chunk_ids = self._merge_chunk_ids(
                [] if current is None else current.evidence_chunk_ids,
                entity.evidence_chunk_ids,
            )
            entity_type = (
                current.entity_type if current is not None and current.entity_type != "concept" else entity.entity_type
            )
            metadata = self._merge_entity_metadata(
                existing=None if matched is None else matched.metadata,
                current=None if current is None else current.metadata,
                incoming=entity.metadata,
                aliases=alias_values,
                entity_key=resolved_key,
                entity_type=entity_type,
                doc_id=document.doc_id,
            )
            resolved_entities[resolved_node_id] = MergedEntity(
                node_id=resolved_node_id,
                key=resolved_key,
                label=preferred_label,
                entity_type=metadata.get("entity_type", entity_type),
                description=self._choose_description(
                    "" if current is None else current.description, entity.description
                ),
                evidence_chunk_ids=evidence_chunk_ids,
                metadata=metadata | {"evidence_count": str(len(evidence_chunk_ids))},
            )
            rewritten_node_ids[entity.node_id] = resolved_node_id

        resolved_relations: dict[tuple[str, str, str], MergedRelation] = {}
        for relation in merged.relations:
            from_node_id = rewritten_node_ids.get(relation.from_node_id, relation.from_node_id)
            to_node_id = rewritten_node_ids.get(relation.to_node_id, relation.to_node_id)
            if from_node_id == to_node_id:
                continue
            relation_key = (from_node_id, to_node_id, relation.relation_type)
            current_relation = resolved_relations.get(relation_key)
            from_entity = resolved_entities.get(from_node_id)
            to_entity = resolved_entities.get(to_node_id)
            evidence_chunk_ids = self._merge_chunk_ids(
                [] if current_relation is None else current_relation.evidence_chunk_ids,
                relation.evidence_chunk_ids,
            )
            metadata = self._merge_relation_metadata(
                current=None if current_relation is None else current_relation.metadata,
                incoming=relation.metadata,
                from_label=relation.metadata.get("from_label") if from_entity is None else from_entity.label,
                to_label=relation.metadata.get("to_label") if to_entity is None else to_entity.label,
                doc_id=document.doc_id,
            )
            resolved_relations[relation_key] = MergedRelation(
                edge_id=self._canonical_relation_edge_id(from_node_id, to_node_id, relation.relation_type),
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                relation_type=relation.relation_type,
                description=self._choose_description(
                    "" if current_relation is None else current_relation.description, relation.description
                ),
                confidence=max(
                    relation.confidence,
                    0.0 if current_relation is None else current_relation.confidence,
                ),
                evidence_chunk_ids=evidence_chunk_ids,
                metadata=metadata | {"evidence_count": str(len(evidence_chunk_ids))},
            )

        return MergedGraph(
            entities=list(resolved_entities.values()),
            relations=list(resolved_relations.values()),
        )

    def _find_existing_entity_match(self, entity: MergedEntity) -> GraphNode | None:
        incoming_aliases = self._entity_alias_values(entity.label, entity.metadata.get("aliases", ""))
        if not incoming_aliases:
            return None

        incoming_normalized = {normalized for alias in incoming_aliases if (normalized := self._normalize_alias(alias))}
        incoming_acronyms = {acronym for alias in incoming_aliases if (acronym := self._label_acronym(alias))}
        exact_key_match: GraphNode | None = None
        normalized_overlap_matches: list[GraphNode] = []
        acronym_matches: list[GraphNode] = []

        for alias in incoming_aliases:
            for node in self.graph.list_nodes_by_alias(alias, node_type="entity"):
                existing_key = node.metadata.get("entity_key") or normalize_entity_key(node.label)
                if existing_key and existing_key == entity.key:
                    exact_key_match = node
                    continue
                existing_aliases = self._entity_alias_values(node.label, node.metadata.get("aliases", ""))
                existing_normalized = {
                    normalized for item in existing_aliases if (normalized := self._normalize_alias(item))
                }
                existing_acronyms = {acronym for item in existing_aliases if (acronym := self._label_acronym(item))}
                if incoming_normalized & existing_normalized:
                    normalized_overlap_matches.append(node)
                    continue
                if incoming_acronyms & existing_acronyms:
                    acronym_matches.append(node)

        if exact_key_match is not None:
            return exact_key_match
        if normalized_overlap_matches:
            return sorted(normalized_overlap_matches, key=lambda node: node.node_id)[0]
        if len({node.node_id for node in acronym_matches}) == 1:
            return acronym_matches[0]
        return None

    def _extract_entities_and_relations(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
    ) -> EntityRelationExtractionResult:
        extractor = self.extractor
        if extractor is None:
            return EntityRelationExtractionResult()
        entities = []
        relations = []
        for chunk in chunks:
            cache_key = f"{chunk.content_hash or chunk.chunk_id}::entity_relation::v2"
            cached = self.cache.get(cache_key, namespace="extract")
            if cached is not None and isinstance(cached.payload, dict):
                try:
                    result = EntityRelationExtractionResult.model_validate(cached.payload)
                except Exception:
                    result = extractor.extract(document=document, chunks=[chunk])
            else:
                result = extractor.extract(document=document, chunks=[chunk])
                self.cache.save(
                    CacheEntry(
                        namespace="extract",
                        cache_key=cache_key,
                        payload=result.model_dump(mode="json"),
                        expires_at=datetime.now(UTC) + timedelta(days=7),
                    )
                )
            entities.extend(result.entities)
            relations.extend(result.relations)
        return EntityRelationExtractionResult(entities=entities, relations=relations)

    def _index_multimodal_vectors(self, *, source: Source, document: Document, chunks: list[Chunk]) -> None:
        special_chunks = [
            chunk for chunk in chunks if chunk.chunk_role is ChunkRole.SPECIAL and chunk.special_chunk_type
        ]
        if not special_chunks:
            return
        texts = [self._multimodal_embedding_text(chunk) for chunk in special_chunks]
        for binding in self.embedding_capabilities:
            vectors = self._embed_texts(binding, texts)
            if vectors is None:
                continue
            for chunk, vector, text in zip(special_chunks, vectors, texts, strict=True):
                self._upsert_graph_vector(
                    item_id=self._deterministic_id(chunk.chunk_id, "multimodal"),
                    item_kind="multimodal",
                    vector=vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "doc_ids": document.doc_id,
                        "source_id": source.source_id,
                        "source_ids": source.source_id,
                        "segment_id": chunk.segment_id,
                        "chunk_id": chunk.chunk_id,
                        "text": text,
                        "special_chunk_type": chunk.special_chunk_type or "special",
                    },
                    embedding_space=binding.space,
                )

    def _index_entity_vectors(
        self,
        *,
        source: Source,
        document: Document,
        entities: Sequence[MergedEntity],
    ) -> None:
        texts = [self._entity_embedding_text(entity) for entity in entities]
        if not texts:
            return
        for binding in self.embedding_capabilities:
            vectors = self._embed_texts(binding, texts)
            if vectors is None:
                continue
            for entity, vector, text in zip(entities, vectors, texts, strict=True):
                self._upsert_graph_vector(
                    item_id=entity.node_id,
                    item_kind="entity",
                    vector=vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "doc_ids": document.doc_id,
                        "source_id": source.source_id,
                        "source_ids": source.source_id,
                        "segment_id": "",
                        "text": text,
                        "entity_type": entity.entity_type,
                        "entity_key": entity.key,
                        "aliases": entity.metadata.get("aliases", ""),
                    },
                    embedding_space=binding.space,
                )

    def _index_relation_vectors(
        self,
        *,
        source: Source,
        document: Document,
        relations: Sequence[MergedRelation],
    ) -> None:
        texts = [
            (
                f"{relation.metadata.get('from_label', relation.from_node_id)} "
                f"{relation.relation_type} "
                f"{relation.metadata.get('to_label', relation.to_node_id)}: "
                f"{relation.description}"
            )
            for relation in relations
        ]
        if not texts:
            return
        for binding in self.embedding_capabilities:
            vectors = self._embed_texts(binding, texts)
            if vectors is None:
                continue
            for relation, vector, text in zip(relations, vectors, texts, strict=True):
                self._upsert_graph_vector(
                    item_id=relation.edge_id,
                    item_kind="relation",
                    vector=vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "doc_ids": document.doc_id,
                        "source_id": source.source_id,
                        "source_ids": source.source_id,
                        "segment_id": "",
                        "text": text,
                        "relation_type": relation.relation_type,
                    },
                    embedding_space=binding.space,
                )

    @staticmethod
    def _multimodal_relation_type(special_chunk_type: str | None) -> str:
        mapping = {
            "table": "tabulated_in",
            "figure": "illustrated_by",
            "caption": "captioned_by",
            "ocr_region": "observed_in",
            "image_summary": "summarized_by",
            "formula": "expressed_by_formula",
        }
        return mapping.get(special_chunk_type or "", "supported_by_multimodal")

    def _repair_duplicate_indexes(
        self,
        *,
        source: Source,
        document: Document,
        segments: list[Segment],
        chunks: list[Chunk],
    ) -> int:
        segments_by_id = {segment.segment_id: segment for segment in segments}
        for chunk in chunks:
            segment = segments_by_id.get(chunk.segment_id)
            if segment is None:
                continue
            self.fts_repo.index_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_id=source.source_id,
                title=document.title,
                toc_path=list(segment.toc_path),
                text=chunk.text,
            )
            self._save_section_graph(source=source, document=document, segment=segment, chunks=[chunk])
        repaired = 0
        for binding in self.embedding_capabilities:
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            missing_vector_ids = set(chunk_ids) - self.vectors.existing_chunk_ids(
                chunk_ids,
                embedding_space=binding.space,
            )
            if not missing_vector_ids:
                continue
            missing_chunks = [chunk for chunk in chunks if chunk.chunk_id in missing_vector_ids]
            vectors = self._embed_texts(binding, [chunk.text for chunk in missing_chunks])
            if vectors is None:
                continue
            repaired += len(missing_chunks)
            for chunk, vector in zip(missing_chunks, vectors, strict=True):
                self.vectors.upsert_chunk(
                    chunk.chunk_id,
                    vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "segment_id": chunk.segment_id,
                        "text": chunk.text,
                        **_benchmark_metadata(chunk.metadata),
                    },
                    embedding_space=binding.space,
                )
        return repaired

    def _save_section_graph(
        self,
        *,
        source: Source,
        document: Document,
        segment: Segment,
        chunks: list[Chunk],
    ) -> None:
        document_node = GraphNode(
            node_id=document.doc_id,
            node_type="document",
            label=document.title,
            metadata={"source_id": document.source_id},
        )
        section_node = GraphNode(
            node_id=segment.segment_id,
            node_type="section",
            label=" > ".join(segment.toc_path),
            metadata={"doc_id": document.doc_id},
        )
        evidence_chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.segment_id == segment.segment_id]
        self.graph.save_node(document_node, evidence_chunk_ids=evidence_chunk_ids)
        self.graph.save_node(section_node, evidence_chunk_ids=evidence_chunk_ids)
        if not evidence_chunk_ids:
            return
        edge = GraphEdge(
            edge_id=self._deterministic_id(document.doc_id, segment.segment_id, "edge"),
            from_node_id=document.doc_id,
            to_node_id=segment.segment_id,
            relation_type="contains",
            confidence=1.0,
            evidence_chunk_ids=evidence_chunk_ids,
            metadata={"doc_id": document.doc_id, "source_id": source.source_id},
        )
        self.graph.save_edge(edge)

    def _index_chunk_vectors(
        self,
        *,
        document: Document,
        segment: Segment,
        chunks: list[Chunk],
    ) -> None:
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        for binding in self.embedding_capabilities:
            vectors = self._embed_texts(binding, texts)
            if vectors is None:
                continue
            for chunk, vector in zip(chunks, vectors, strict=True):
                self.vectors.upsert_chunk(
                    chunk.chunk_id,
                    vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "segment_id": segment.segment_id,
                        "text": chunk.text,
                        **_benchmark_metadata(chunk.metadata),
                    },
                    embedding_space=binding.space,
                )

    def _resolve_parsed_document(
        self,
        *,
        request: IngestRequest,
        source_type: SourceType,
        raw_bytes: bytes,
    ) -> ParsedDocument:
        if request.parsed_document is not None:
            return replace(request.parsed_document, source_type=source_type)

        if source_type is SourceType.MARKDOWN:
            if request.file_path is not None:
                return self.docling_parser.parse(
                    request.file_path,
                    location=request.location,
                    title=request.title,
                    owner=request.owner,
                )
            markdown = request.content_text or raw_bytes.decode("utf-8")
            return self._parse_docling_text(
                content=markdown,
                location=request.location,
                title=request.title,
                owner=request.owner,
                suffix=".md",
            )

        if source_type in {SourceType.PLAIN_TEXT, SourceType.PASTED_TEXT}:
            text = request.content_text or raw_bytes.decode("utf-8")
            parsed = self.plain_text_parser.parse(
                text,
                location=request.location,
                title=request.title,
                owner=request.owner,
            )
            return replace(parsed, source_type=source_type)

        if source_type in {SourceType.WEB, SourceType.BROWSER_CLIP}:
            html = request.content_text or raw_bytes.decode("utf-8")
            parsed = self.web_parser.parse(
                html,
                location=request.location,
                title=request.title,
                owner=request.owner,
            )
            return replace(parsed, source_type=source_type)

        file_path = request.file_path or Path(request.location)
        if source_type is SourceType.IMAGE:
            try:
                return self.docling_parser.parse(
                    file_path,
                    location=request.location,
                    title=request.title,
                    owner=request.owner,
                )
            except ValueError:
                return self.image_parser.parse(
                    file_path,
                    location=request.location,
                    title=request.title,
                    owner=request.owner,
                )

        if source_type is SourceType.PDF:
            try:
                return self.docling_parser.parse(
                    file_path,
                    location=request.location,
                    title=request.title,
                    owner=request.owner,
                )
            except ValueError:
                return self.pdf_parser.parse(
                    file_path,
                    location=request.location,
                    title=request.title,
                    owner=request.owner,
                )

        if source_type in {SourceType.DOCX, SourceType.PPTX, SourceType.XLSX}:
            return self.docling_parser.parse(
                file_path,
                location=request.location,
                title=request.title,
                owner=request.owner,
            )

        raise ValueError(f"Unsupported source type for ingest pipeline: {source_type}")

    def _resolve_raw_bytes(self, *, request: IngestRequest, source_type: SourceType) -> bytes:
        if request.raw_bytes is not None:
            return request.raw_bytes
        if request.file_path is not None:
            return request.file_path.read_bytes()
        if source_type in {SourceType.WEB, SourceType.BROWSER_CLIP} and request.content_text is None:
            return self.web_fetch_repo.fetch(request.location).encode("utf-8")
        if request.content_text is not None:
            return request.content_text.replace("\r\n", "\n").encode("utf-8")
        fallback_path = Path(request.location)
        if fallback_path.exists():
            return fallback_path.read_bytes()
        raise ValueError(f"No raw content available for ingest request: {request.location}")

    def _parse_docling_text(
        self,
        *,
        content: str,
        location: str,
        title: str | None,
        owner: str,
        suffix: str,
    ) -> ParsedDocument:
        with NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        try:
            return self.docling_parser.parse(temp_path, location=location, title=title, owner=owner)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _ingest_request_from_content_item(item: DirectContentItem) -> IngestRequest:
        file_path: Path | None = None
        raw_bytes: bytes | None = None
        content_text: str | None = None

        if isinstance(item.content, Path):
            file_path = item.content
        elif isinstance(item.content, bytes):
            raw_bytes = item.content
        elif isinstance(item.content, str):
            content_text = item.content
        else:
            raise TypeError(f"Unsupported direct content type: {type(item.content).__name__}")

        source_type = SourceType(item.source_type)
        if source_type in {SourceType.PDF, SourceType.DOCX, SourceType.PPTX, SourceType.XLSX, SourceType.IMAGE}:
            if file_path is None and raw_bytes is None:
                raise ValueError(f"Binary content for {source_type.value} requires bytes or a file path")

        return IngestRequest(
            location=item.location,
            source_type=source_type,
            owner=item.owner,
            access_policy=item.access_policy,
            title=item.title,
            content_text=content_text,
            raw_bytes=raw_bytes,
            file_path=file_path,
        )

    def _save_status(
        self,
        *,
        doc_id: str,
        source_id: str,
        location: str,
        content_hash: str,
        status: DocumentProcessingStatus,
        stage: DocumentPipelineStage,
        error_message: str | None = None,
    ) -> DocumentStatusRecord:
        existing = self.status.get(doc_id)
        attempts = (
            1 if existing is None else existing.attempts + (1 if status is DocumentProcessingStatus.FAILED else 0)
        )
        record = DocumentStatusRecord(
            doc_id=doc_id,
            source_id=source_id,
            location=location,
            content_hash=content_hash,
            status=status,
            stage=stage,
            attempts=attempts,
            error_message=error_message,
        )
        return self.status.save(record)

    @staticmethod
    def _embed_texts(binding: EmbeddingCapabilityBinding, texts: list[str]) -> list[list[float]] | None:
        try:
            vectors = binding.embed(texts)
        except RuntimeError:
            return None
        if len(vectors) != len(texts):
            return None
        return vectors

    def _upsert_graph_vector(
        self,
        *,
        item_id: str,
        item_kind: str,
        vector: list[float],
        metadata: dict[str, str],
        embedding_space: str,
    ) -> None:
        existing = self.vectors.vector_repo.get_entry(
            item_id,
            embedding_space=embedding_space,
            item_kind=item_kind,
        )
        merged_metadata = self._merge_vector_metadata(
            existing=None if existing is None else existing.metadata,
            incoming=metadata,
        )
        merged_vector = self._merge_vector_values(
            existing=None if existing is None else existing.vector,
            incoming=vector,
            previous_count=0 if existing is None else int(existing.metadata.get("merge_count", "1")),
        )
        self.vectors.vector_repo.upsert(
            item_id,
            merged_vector,
            metadata=merged_metadata,
            embedding_space=embedding_space,
            item_kind=item_kind,
        )

    @staticmethod
    def _merge_vector_metadata(
        *,
        existing: dict[str, str] | None,
        incoming: dict[str, str],
    ) -> dict[str, str]:
        merged = dict(existing or {})
        merged.update(incoming)
        for scalar_key, plural_key in (("doc_id", "doc_ids"), ("source_id", "source_ids")):
            values: list[str] = []
            for container in (existing or {}, incoming):
                scalar_value = container.get(scalar_key)
                if scalar_value:
                    values.append(scalar_value)
                plural_value = container.get(plural_key)
                if plural_value:
                    values.extend(item.strip() for item in plural_value.split(",") if item.strip())
            if values:
                merged[plural_key] = ",".join(sorted(dict.fromkeys(values)))
        existing_count = 0 if existing is None else int(existing.get("merge_count", "1"))
        merged["merge_count"] = str(existing_count + 1)
        current_text = incoming.get("text", "")
        previous_text = "" if existing is None else existing.get("text", "")
        if previous_text and len(previous_text) > len(current_text):
            merged["text"] = previous_text
        return merged

    @staticmethod
    def _merge_alias_values(*groups: list[str]) -> list[str]:
        aliases: list[str] = []
        for group in groups:
            for alias in group:
                normalized = alias.strip()
                if normalized and normalized not in aliases:
                    aliases.append(normalized)
        return aliases

    @staticmethod
    def _entity_alias_values(label: str, aliases_blob: str) -> list[str]:
        aliases = [label]
        if aliases_blob:
            aliases.extend(alias for alias in aliases_blob.split("||") if alias)
        return list(dict.fromkeys(alias.strip() for alias in aliases if alias and alias.strip()))

    @staticmethod
    def _merge_chunk_ids(left: list[str], right: list[str]) -> list[str]:
        return list(dict.fromkeys([*left, *right]))

    @staticmethod
    def _merge_entity_metadata(
        *,
        existing: dict[str, str] | None,
        current: dict[str, str] | None,
        incoming: dict[str, str],
        aliases: list[str],
        entity_key: str,
        entity_type: str,
        doc_id: str,
    ) -> dict[str, str]:
        merged = dict(existing or {})
        merged.update(current or {})
        merged.update(incoming)
        doc_ids = IngestPipeline._merge_csv_values(
            doc_id,
            *(container.get("doc_id", "") for container in (existing or {}, current or {}, incoming)),
            *(container.get("doc_ids", "") for container in (existing or {}, current or {}, incoming)),
        )
        if doc_ids:
            merged["doc_ids"] = doc_ids
        merged["doc_id"] = doc_id
        merged["aliases"] = "||".join(aliases)
        merged["entity_key"] = entity_key
        merged["entity_type"] = entity_type
        return merged

    @staticmethod
    def _merge_relation_metadata(
        *,
        current: dict[str, str] | None,
        incoming: dict[str, str],
        from_label: str | None,
        to_label: str | None,
        doc_id: str,
    ) -> dict[str, str]:
        merged = dict(current or {})
        merged.update(incoming)
        doc_ids = IngestPipeline._merge_csv_values(
            doc_id,
            *(container.get("doc_id", "") for container in (current or {}, incoming)),
            *(container.get("doc_ids", "") for container in (current or {}, incoming)),
        )
        if doc_ids:
            merged["doc_ids"] = doc_ids
        merged["doc_id"] = doc_id
        if from_label:
            merged["from_label"] = from_label
        if to_label:
            merged["to_label"] = to_label
        return merged

    @staticmethod
    def _merge_csv_values(*values: str) -> str:
        items: list[str] = []
        for value in values:
            if not value:
                continue
            items.extend(item.strip() for item in value.split(",") if item.strip())
        return ",".join(sorted(dict.fromkeys(items)))

    @staticmethod
    def _choose_description(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return left if len(left) >= len(right) else right

    @staticmethod
    def _normalize_alias(alias: str) -> str:
        return normalize_whitespace(alias).lower()

    @staticmethod
    def _label_acronym(label: str) -> str | None:
        tokens = [token for token in normalize_whitespace(label).replace("-", " ").split(" ") if token]
        if len(tokens) < 2:
            return None
        acronym = "".join(token[0].upper() for token in tokens if token and token[0].isalnum())
        return acronym or None

    @staticmethod
    def _canonical_relation_edge_id(from_node_id: str, to_node_id: str, relation_type: str) -> str:
        digest = sha256("\0".join((from_node_id, to_node_id, relation_type, "edge")).encode("utf-8")).hexdigest()
        return f"edge-{digest[:16]}"

    def _entity_embedding_text(self, entity: MergedEntity) -> str:
        label = entity.label
        metadata = entity.metadata
        aliases = [alias for alias in self._entity_alias_values(label, metadata.get("aliases", "")) if alias != label]
        description = entity.description
        parts = [label]
        if description:
            parts.append(description)
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases[:6])}")
        return ". ".join(part for part in parts if part)

    @staticmethod
    def _multimodal_embedding_text(chunk: Chunk) -> str:
        special_type = (chunk.special_chunk_type or "special").replace("_", " ")
        toc_path = chunk.metadata.get("toc_path", "")
        parts = [f"{special_type.title()}: {chunk.text}"]
        if toc_path:
            parts.append(f"Section: {toc_path}")
        return ". ".join(part for part in parts if part)

    @staticmethod
    def _merge_vector_values(
        *,
        existing: list[float] | None,
        incoming: list[float],
        previous_count: int,
    ) -> list[float]:
        if existing is None or not existing or len(existing) != len(incoming):
            return [float(value) for value in incoming]
        weight = max(previous_count, 1)
        return [
            ((float(current) * weight) + float(next_value)) / (weight + 1)
            for current, next_value in zip(existing, incoming, strict=True)
        ]

    @staticmethod
    def _retrievable_chunks(chunks: list[Chunk]) -> list[Chunk]:
        return [chunk for chunk in chunks if chunk.chunk_role is not ChunkRole.PARENT]

    @staticmethod
    def _suffix_for(source_type: SourceType) -> str:
        return {
            SourceType.PDF: ".pdf",
            SourceType.MARKDOWN: ".md",
            SourceType.DOCX: ".docx",
            SourceType.PPTX: ".pptx",
            SourceType.XLSX: ".xlsx",
            SourceType.IMAGE: ".png",
            SourceType.WEB: ".html",
            SourceType.PLAIN_TEXT: ".txt",
            SourceType.PASTED_TEXT: ".txt",
            SourceType.BROWSER_CLIP: ".html",
        }[source_type]

    @staticmethod
    def _deterministic_id(*parts: str) -> str:
        digest = sha256("\0".join(parts).encode("utf-8")).hexdigest()
        return f"{parts[-1]}-{digest[:12]}"


@dataclass(frozen=True, slots=True)
class DeleteRequest:
    doc_id: str | None = None
    source_id: str | None = None
    location: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedLifecycleTarget:
    source: Source
    document: Document
    chunks: list[Chunk]


@dataclass(frozen=True, slots=True)
class PurgeSummary:
    deleted_chunk_ids: list[str] = field(default_factory=list)
    deleted_node_ids: list[str] = field(default_factory=list)
    deleted_edge_ids: list[str] = field(default_factory=list)
    deleted_fts_count: int = 0
    deleted_vector_count: int = 0
    deleted_chunk_record_count: int = 0
    deleted_segment_record_count: int = 0


@dataclass(frozen=True, slots=True)
class DeletePipelineResult:
    deleted_doc_ids: list[str] = field(default_factory=list)
    deleted_source_ids: list[str] = field(default_factory=list)
    deleted_chunk_ids: list[str] = field(default_factory=list)
    deleted_node_ids: list[str] = field(default_factory=list)
    deleted_edge_ids: list[str] = field(default_factory=list)
    deleted_fts_count: int = 0
    deleted_vector_count: int = 0


@dataclass(slots=True)
class DeletePipeline:
    documents: DocumentStore
    chunks: ChunkStore
    vectors: VectorStore
    graph: GraphStore
    status: StatusStore
    fts_repo: FullTextSearchRepo
    ingest_pipeline: IngestPipeline

    def run(self, request: DeleteRequest) -> DeletePipelineResult:
        targets = self.resolve_targets(request, include_inactive=False)
        if not targets:
            raise ValueError("No active document matched the delete request")

        deleted_doc_ids: list[str] = []
        deleted_source_ids: list[str] = []
        deleted_chunk_ids: list[str] = []
        deleted_node_ids: list[str] = []
        deleted_edge_ids: list[str] = []
        deleted_fts_count = 0
        deleted_vector_count = 0

        for target in targets:
            self.ingest_pipeline._save_status(
                doc_id=target.document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.DELETING,
                stage=DocumentPipelineStage.DELETE,
            )
            self.documents.set_active(target.document.doc_id, active=False)
            purge = self.purge_document_artifacts(target, delete_chunk_records=False)
            self.ingest_pipeline._save_status(
                doc_id=target.document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.DELETED,
                stage=DocumentPipelineStage.DELETE,
            )
            deleted_doc_ids.append(target.document.doc_id)
            deleted_source_ids.append(target.source.source_id)
            deleted_chunk_ids.extend(purge.deleted_chunk_ids)
            deleted_node_ids.extend(purge.deleted_node_ids)
            deleted_edge_ids.extend(purge.deleted_edge_ids)
            deleted_fts_count += purge.deleted_fts_count
            deleted_vector_count += purge.deleted_vector_count

        return DeletePipelineResult(
            deleted_doc_ids=deleted_doc_ids,
            deleted_source_ids=list(dict.fromkeys(deleted_source_ids)),
            deleted_chunk_ids=list(dict.fromkeys(deleted_chunk_ids)),
            deleted_node_ids=list(dict.fromkeys(deleted_node_ids)),
            deleted_edge_ids=list(dict.fromkeys(deleted_edge_ids)),
            deleted_fts_count=deleted_fts_count,
            deleted_vector_count=deleted_vector_count,
        )

    def resolve_targets(
        self,
        request: DeleteRequest,
        *,
        include_inactive: bool,
    ) -> list[ResolvedLifecycleTarget]:
        self._validate_request(request)
        if request.doc_id is not None:
            document = self.documents.get_document(request.doc_id)
            if document is None:
                return []
            if not include_inactive and not self.documents.is_active(document.doc_id):
                return []
            source = self.documents.get_source(document.source_id)
            if source is None:
                return []
            return [self._target(source, document)]

        if request.source_id is not None:
            source = self.documents.get_source(request.source_id)
            if source is None:
                return []
            documents = self.documents.list_documents(
                source.source_id,
                active_only=not include_inactive,
            )
            if not documents and include_inactive:
                documents = self.documents.list_documents(source.source_id, active_only=False)
            return [self._target(source, document) for document in documents]

        assert request.location is not None
        source = self.documents.get_latest_source_for_location(request.location)
        if source is None:
            return []
        documents = self.documents.list_documents(source.source_id, active_only=True)
        if not documents and include_inactive:
            documents = self.documents.list_documents(source.source_id, active_only=False)
        return [self._target(source, document) for document in documents]

    def purge_document_artifacts(
        self,
        target: ResolvedLifecycleTarget,
        *,
        delete_chunk_records: bool,
    ) -> PurgeSummary:
        chunk_ids = [chunk.chunk_id for chunk in target.chunks]
        deleted_fts_count = self.fts_repo.delete_by_chunk_ids(chunk_ids)
        deleted_vector_count = self.vectors.delete_for_documents([target.document.doc_id])
        deleted_node_ids, deleted_edge_ids = self.graph.delete_by_chunk_ids(chunk_ids)
        deleted_chunk_record_count = 0
        deleted_segment_record_count = 0
        if delete_chunk_records:
            deleted_chunk_record_count = self.chunks.delete_for_document(target.document.doc_id)
            deleted_segment_record_count = self.documents.delete_segments_for_document(target.document.doc_id)
        return PurgeSummary(
            deleted_chunk_ids=chunk_ids,
            deleted_node_ids=deleted_node_ids,
            deleted_edge_ids=deleted_edge_ids,
            deleted_fts_count=deleted_fts_count,
            deleted_vector_count=deleted_vector_count,
            deleted_chunk_record_count=deleted_chunk_record_count,
            deleted_segment_record_count=deleted_segment_record_count,
        )

    def _target(self, source: Source, document: Document) -> ResolvedLifecycleTarget:
        return ResolvedLifecycleTarget(
            source=source,
            document=document,
            chunks=self.chunks.list_by_document(document.doc_id),
        )

    @staticmethod
    def _validate_request(request: DeleteRequest) -> None:
        selectors = [request.doc_id, request.source_id, request.location]
        if sum(1 for value in selectors if value is not None) != 1:
            raise ValueError("DeleteRequest requires exactly one of doc_id, source_id, or location")


@dataclass(frozen=True, slots=True)
class RebuildRequest:
    doc_id: str | None = None
    source_id: str | None = None
    location: str | None = None


@dataclass(frozen=True, slots=True)
class RebuildPipelineResult:
    results: list[IngestPipelineResult] = field(default_factory=list)

    @property
    def rebuilt_doc_ids(self) -> list[str]:
        return [result.document_id for result in self.results]


@dataclass(slots=True)
class RebuildPipeline:
    ingest_pipeline: IngestPipeline
    delete_pipeline: DeletePipeline
    object_store: ObjectStore | None

    def run(self, request: RebuildRequest) -> RebuildPipelineResult:
        delete_request = DeleteRequest(
            doc_id=request.doc_id,
            source_id=request.source_id,
            location=request.location,
        )
        targets = self.delete_pipeline.resolve_targets(delete_request, include_inactive=True)
        if not targets:
            raise ValueError("No document matched the rebuild request")

        results: list[IngestPipelineResult] = []
        for target in targets:
            results.append(self._rebuild_target(target))
        return RebuildPipelineResult(results=results)

    def _rebuild_target(self, target: ResolvedLifecycleTarget) -> IngestPipelineResult:
        self.ingest_pipeline._save_status(
            doc_id=target.document.doc_id,
            source_id=target.source.source_id,
            location=target.source.location,
            content_hash=target.source.content_hash,
            status=DocumentProcessingStatus.REBUILDING,
            stage=DocumentPipelineStage.REBUILD,
        )
        self.documents.deactivate_documents_for_location(target.source.location)
        self.documents.set_active(target.document.doc_id, active=False)
        stage = DocumentPipelineStage.REBUILD
        try:
            request, raw_bytes = self._build_ingest_request(target)
            parsed = self.ingest_pipeline._resolve_parsed_document(
                request=request,
                source_type=SourceType(target.source.source_type),
                raw_bytes=raw_bytes,
            )
            rebuilt_document = self._rebuilt_document(target.document, parsed)
            self.documents.save_document(
                rebuilt_document,
                location=target.source.location,
                content_hash=target.source.content_hash,
                active=False,
            )
            self.delete_pipeline.purge_document_artifacts(target, delete_chunk_records=True)

            stage = DocumentPipelineStage.CHUNK
            self.ingest_pipeline._save_status(
                doc_id=rebuilt_document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.REBUILDING,
                stage=stage,
            )
            segments, stored_chunks, indexed_chunks, processing = self.ingest_pipeline._prepare_chunks(
                location=target.source.location,
                source=target.source,
                document=rebuilt_document,
                parsed=parsed,
                access_policy=target.source.effective_access_policy,
            )

            stage = DocumentPipelineStage.PERSIST
            self.ingest_pipeline._save_status(
                doc_id=rebuilt_document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.REBUILDING,
                stage=stage,
            )
            for segment in segments:
                self.documents.save_segment(segment)
            self.chunks.save_many(stored_chunks)
            self.ingest_pipeline._index_chunks(
                source=target.source,
                document=rebuilt_document,
                segments=segments,
                chunks=indexed_chunks,
            )

            stage = DocumentPipelineStage.EXTRACT
            self.ingest_pipeline._save_status(
                doc_id=rebuilt_document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.REBUILDING,
                stage=stage,
            )
            entity_count, relation_count = self.ingest_pipeline._extract_and_persist_graph(
                source=target.source,
                document=rebuilt_document,
                chunks=indexed_chunks,
            )
            self.documents.save_document(
                rebuilt_document,
                location=target.source.location,
                content_hash=target.source.content_hash,
                active=True,
            )
            final_status = self.ingest_pipeline._save_status(
                doc_id=rebuilt_document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.READY,
                stage=DocumentPipelineStage.INDEX,
            )
        except Exception as exc:
            self.ingest_pipeline._save_status(
                doc_id=target.document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.FAILED,
                stage=stage,
                error_message=str(exc),
            )
            raise

        return IngestPipelineResult(
            source=target.source,
            document=rebuilt_document,
            segments=segments,
            chunks=indexed_chunks,
            is_duplicate=False,
            content_hash=target.source.content_hash,
            visible_text=parsed.visible_text,
            visual_semantics=parsed.visual_semantics,
            processing=processing,
            entity_count=entity_count,
            relation_count=relation_count,
            status=final_status.status.value,
        )

    @property
    def documents(self) -> DocumentStore:
        return self.ingest_pipeline.documents

    @property
    def chunks(self) -> ChunkStore:
        return self.ingest_pipeline.chunks

    def _build_ingest_request(self, target: ResolvedLifecycleTarget) -> tuple[IngestRequest, bytes]:
        source = target.source
        source_type = SourceType(source.source_type)
        object_key = source.metadata.get("object_key")
        file_path: Path | None = None
        raw_bytes: bytes | None = None
        if object_key and self.object_store is not None and self.object_store.exists(object_key):
            raw_bytes = self.object_store.read_bytes(object_key)
            if source_type in {SourceType.PDF, SourceType.DOCX, SourceType.PPTX, SourceType.XLSX, SourceType.IMAGE}:
                file_path = self.object_store.path_for_key(object_key)
        elif Path(source.location).exists():
            file_path = Path(source.location)
            raw_bytes = file_path.read_bytes()
        if raw_bytes is None:
            raise ValueError(f"No rebuildable source payload available for {source.location}")

        content_text = None
        if source_type in {
            SourceType.MARKDOWN,
            SourceType.PLAIN_TEXT,
            SourceType.PASTED_TEXT,
            SourceType.WEB,
            SourceType.BROWSER_CLIP,
        }:
            content_text = raw_bytes.decode("utf-8")

        return (
            IngestRequest(
                location=source.location,
                source_type=source_type,
                owner=source.owner,
                title=target.document.title,
                content_text=content_text,
                raw_bytes=raw_bytes,
                file_path=file_path,
            ),
            raw_bytes,
        )

    @staticmethod
    def _rebuilt_document(document: Document, parsed: ParsedDocument) -> Document:
        return document.model_copy(
            update={
                "doc_type": parsed.doc_type,
                "title": parsed.title or document.title,
                "authors": list(parsed.authors or document.authors),
                "language": parsed.language or document.language,
                "metadata": {
                    **parsed.metadata,
                    "location": document.metadata.get("location", ""),
                    "content_hash": document.metadata.get("content_hash", ""),
                },
            }
        )


@dataclass(frozen=True)
class IngestResult:
    source: Source
    document: Document
    segments: list[Segment]
    chunks: list[Chunk]
    is_duplicate: bool
    content_hash: str
    visible_text: str
    visual_semantics: str | None = None
    processing: DocumentProcessingPackage | None = None
    entity_count: int = 0
    relation_count: int = 0
    status: str = "ready"


class IngestService:
    def __init__(
        self,
        *,
        metadata_repo: MetadataRepo,
        cache_repo: CacheRepo | None,
        fts_repo: FullTextSearchRepo,
        vector_repo: VectorRepo,
        graph_repo: GraphRepo,
        object_store: ObjectStore,
        markdown_parser: MarkdownParserRepo,
        pdf_parser: PDFParserRepo,
        plain_text_parser: PlainTextParserRepo,
        image_parser: ImageParserRepo,
        web_parser: WebParserRepo,
        web_fetch_repo: WebFetchRepo,
        docling_parser: DoclingParserRepo,
        policy_resolution_service: PolicyResolutionService,
        toc_service: TOCService,
        chunking_service: ChunkingService,
        document_processing_service: DocumentProcessingService,
        capability_bundle: CapabilityBundle,
    ) -> None:
        self.metadata_repo = metadata_repo
        self.cache_repo = cache_repo or cast(CacheRepo, metadata_repo)
        self.fts_repo = fts_repo
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo
        self.object_store = object_store
        self.markdown_parser = markdown_parser
        self.pdf_parser = pdf_parser
        self.plain_text_parser = plain_text_parser
        self.image_parser = image_parser
        self.web_parser = web_parser
        self.web_fetch_repo = web_fetch_repo
        self.docling_parser = docling_parser
        self.policy_resolution_service = policy_resolution_service
        self.toc_service = toc_service
        self.chunking_service = chunking_service
        self.document_processing_service = document_processing_service
        self.capability_bundle = capability_bundle
        self.embedding_capabilities = self.capability_bundle.embedding_bindings
        self.chat_capabilities = self.capability_bundle.chat_bindings
        self.ingest_pipeline = IngestPipeline(
            documents=DocumentStore(metadata_repo=self.metadata_repo),
            chunks=ChunkStore(metadata_repo=self.metadata_repo),
            vectors=VectorStore(vector_repo=self.vector_repo),
            graph=GraphStore(graph_repo=self.graph_repo),
            status=StatusStore(metadata_repo=self.metadata_repo),
            cache=CacheStore(cache_repo=self.cache_repo),
            fts_repo=self.fts_repo,
            object_store=self.object_store,
            markdown_parser=self.markdown_parser,
            pdf_parser=self.pdf_parser,
            plain_text_parser=self.plain_text_parser,
            image_parser=self.image_parser,
            web_parser=self.web_parser,
            web_fetch_repo=self.web_fetch_repo,
            docling_parser=self.docling_parser,
            policy_resolution_service=self.policy_resolution_service,
            toc_service=self.toc_service,
            chunking_service=self.chunking_service,
            document_processing_service=self.document_processing_service,
            embedding_capabilities=self.embedding_capabilities,
            chat_capabilities=self.chat_capabilities,
        )

    @classmethod
    def create_in_memory(
        cls,
        root: Path,
        *,
        capability_bundle: CapabilityBundle,
        ocr_repo: OcrVisionRepo | None = None,
        vlm_repo: VisualDescriptionRepo | None = None,
        web_fetch_repo: WebFetchRepo | None = None,
    ) -> IngestService:
        root.mkdir(parents=True, exist_ok=True)
        resolved_ocr_repo = ocr_repo or create_default_ocr_repo()
        resolved_web_fetch_repo = web_fetch_repo or HttpWebFetchRepo()
        token_contract = capability_bundle.token_contract
        token_accounting = capability_bundle.token_accounting
        stores = StorageConfig(root=root).build()
        return cls(
            metadata_repo=stores.metadata_repo,
            cache_repo=stores.cache_repo,
            fts_repo=stores.fts_repo,
            vector_repo=stores.vector_repo,
            graph_repo=stores.graph_repo,
            object_store=stores.object_store,
            markdown_parser=MarkdownParserRepo(),
            pdf_parser=PDFParserRepo(),
            plain_text_parser=PlainTextParserRepo(),
            image_parser=ImageParserRepo(resolved_ocr_repo),
            web_parser=WebParserRepo(),
            web_fetch_repo=resolved_web_fetch_repo,
            docling_parser=DoclingParserRepo(resolved_ocr_repo, vlm_repo),
            policy_resolution_service=PolicyResolutionService(),
            toc_service=TOCService(),
            chunking_service=ChunkingService(token_accounting=token_accounting),
            document_processing_service=DocumentProcessingService(
                toc_service=TOCService(),
                token_accounting=token_accounting,
                tokenizer_contract=token_contract,
            ),
            capability_bundle=capability_bundle,
        )

    def ingest_markdown(
        self,
        *,
        location: str,
        markdown: str,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        parsed = self._parse_docling_text(
            content=markdown,
            location=location,
            title=title,
            owner=owner,
            suffix=".md",
        )
        return self._ingest_parsed_document(
            location=location,
            raw_bytes=markdown.replace("\r\n", "\n").encode("utf-8"),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_plain_text(
        self,
        *,
        location: str,
        text: str,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
        source_type: SourceType | str = SourceType.PLAIN_TEXT,
    ) -> IngestResult:
        parsed = self.plain_text_parser.parse(
            text,
            location=location,
            title=title,
            owner=owner,
        )
        parsed = replace(parsed, source_type=SourceType(source_type))
        return self._ingest_parsed_document(
            location=location,
            raw_bytes=text.replace("\r\n", "\n").encode("utf-8"),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_pdf(
        self,
        *,
        location: str,
        pdf_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        try:
            parsed = self.docling_parser.parse(pdf_path, location=location, title=title, owner=owner)
        except ValueError:
            parsed = self.pdf_parser.parse(pdf_path, location=location, title=title, owner=owner)
        return self._ingest_parsed_document(
            location=location,
            raw_bytes=pdf_path.read_bytes(),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_image(
        self,
        *,
        location: str,
        image_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        try:
            parsed = self.docling_parser.parse(image_path, location=location, title=title, owner=owner)
        except ValueError:
            parsed = self.image_parser.parse(image_path, location=location, title=title, owner=owner)
        return self._ingest_parsed_document(
            location=location,
            raw_bytes=image_path.read_bytes(),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_docx(
        self,
        *,
        location: str,
        docx_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        parsed = self.docling_parser.parse(docx_path, location=location, title=title, owner=owner)
        return self._ingest_parsed_document(
            location=location,
            raw_bytes=docx_path.read_bytes(),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_pptx(
        self,
        *,
        location: str,
        pptx_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        parsed = self.docling_parser.parse(pptx_path, location=location, title=title, owner=owner)
        return self._ingest_parsed_document(
            location=location,
            raw_bytes=pptx_path.read_bytes(),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_xlsx(
        self,
        *,
        location: str,
        xlsx_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        parsed = self.docling_parser.parse(xlsx_path, location=location, title=title, owner=owner)
        return self._ingest_parsed_document(
            location=location,
            raw_bytes=xlsx_path.read_bytes(),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_file(
        self,
        *,
        location: str,
        file_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        source_type = self.docling_parser.infer_source_type(file_path)
        if source_type is SourceType.PDF:
            return self.ingest_pdf(
                location=location,
                pdf_path=file_path,
                owner=owner,
                access_policy=access_policy,
                title=title,
            )
        if source_type is SourceType.MARKDOWN:
            return self.ingest_markdown(
                location=location,
                markdown=file_path.read_text(encoding="utf-8"),
                owner=owner,
                access_policy=access_policy,
                title=title,
            )
        if source_type is SourceType.DOCX:
            return self.ingest_docx(
                location=location,
                docx_path=file_path,
                owner=owner,
                access_policy=access_policy,
                title=title,
            )
        if source_type is SourceType.PPTX:
            return self.ingest_pptx(
                location=location,
                pptx_path=file_path,
                owner=owner,
                access_policy=access_policy,
                title=title,
            )
        if source_type is SourceType.XLSX:
            return self.ingest_xlsx(
                location=location,
                xlsx_path=file_path,
                owner=owner,
                access_policy=access_policy,
                title=title,
            )
        if source_type is SourceType.IMAGE:
            return self.ingest_image(
                location=location,
                image_path=file_path,
                owner=owner,
                access_policy=access_policy,
                title=title,
            )
        raise ValueError(f"Unsupported file type for pipeline entry: {file_path.suffix or file_path.name}")

    def ingest_web(
        self,
        *,
        location: str,
        html: str,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
        source_type: SourceType | str = SourceType.WEB,
    ) -> IngestResult:
        parsed = self.web_parser.parse(html, location=location, title=title, owner=owner)
        parsed = replace(parsed, source_type=SourceType(source_type))
        return self._ingest_parsed_document(
            location=location,
            raw_bytes=html.encode("utf-8"),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_web_url(
        self,
        *,
        location: str,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        html = self.web_fetch_repo.fetch(location)
        return self.ingest_web(
            location=location,
            html=html,
            owner=owner,
            access_policy=access_policy,
            title=title,
        )

    def repair_indexes(self) -> dict[str, int]:
        documents = self.metadata_repo.list_documents(active_only=True)
        chunk_count = 0
        repaired_vector_count = 0
        for document in documents:
            source = self.metadata_repo.get_source(document.source_id)
            if source is None:
                continue
            segments = self.metadata_repo.list_segments(document.doc_id)
            stored_chunks = self.metadata_repo.list_chunks(document.doc_id)
            chunks = self._retrievable_chunks(stored_chunks)
            chunk_count += len(chunks)
            repaired_vector_count += self._repair_duplicate_indexes(
                source=source,
                document=document,
                segments=segments,
                chunks=chunks,
            )
        return {
            "document_count": len(documents),
            "chunk_count": chunk_count,
            "repaired_vector_count": repaired_vector_count,
        }

    def _ingest_parsed_document(
        self,
        *,
        location: str,
        raw_bytes: bytes,
        parsed: ParsedDocument,
        owner: str,
        access_policy: AccessPolicy | None,
    ) -> IngestResult:
        result = self.ingest_pipeline.ingest_parsed_document(
            location=location,
            raw_bytes=raw_bytes,
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )
        return self._from_pipeline_result(result)

    @staticmethod
    def _from_pipeline_result(result: IngestPipelineResult) -> IngestResult:
        return IngestResult(
            source=result.source,
            document=result.document,
            segments=result.segments,
            chunks=result.chunks,
            is_duplicate=result.is_duplicate,
            content_hash=result.content_hash,
            visible_text=result.visible_text,
            visual_semantics=result.visual_semantics,
            processing=result.processing,
            entity_count=result.entity_count,
            relation_count=result.relation_count,
            status=result.status,
        )

    def _repair_duplicate_indexes(
        self,
        *,
        source: Source,
        document: Document,
        segments: list[Segment],
        chunks: list[Chunk],
    ) -> int:
        segments_by_id = {segment.segment_id: segment for segment in segments}
        for chunk in chunks:
            segment = segments_by_id.get(chunk.segment_id)
            if segment is None:
                continue
            self.fts_repo.index_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_id=source.source_id,
                title=document.title,
                toc_path=list(segment.toc_path),
                text=chunk.text,
            )

        repaired_vectors = self._repair_chunk_vectors(document=document, chunks=chunks)

        for segment in segments:
            segment_chunks = [chunk for chunk in chunks if chunk.segment_id == segment.segment_id]
            self._save_graph_candidates(document, segment, segment_chunks)
        return repaired_vectors

    def _repair_chunk_vectors(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
    ) -> int:
        repaired = 0
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        for binding in self.embedding_capabilities:
            missing_vector_ids = set(chunk_ids) - self.vector_repo.existing_item_ids(
                chunk_ids,
                embedding_space=binding.space,
            )
            if not missing_vector_ids:
                continue
            missing_chunks = [chunk for chunk in chunks if chunk.chunk_id in missing_vector_ids]
            vectors = self._embed_texts(binding, [chunk.text for chunk in missing_chunks])
            if vectors is None:
                continue
            repaired += len(missing_chunks)
            for chunk, vector in zip(missing_chunks, vectors, strict=True):
                self.vector_repo.upsert(
                    chunk.chunk_id,
                    vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "segment_id": chunk.segment_id,
                        "text": chunk.text,
                    },
                    embedding_space=binding.space,
                )
        return repaired

    @staticmethod
    def _retrievable_chunks(chunks: list[Chunk]) -> list[Chunk]:
        return [chunk for chunk in chunks if chunk.chunk_role is not ChunkRole.PARENT]

    @staticmethod
    def _embed_texts(binding: EmbeddingCapabilityBinding, texts: list[str]) -> list[list[float]] | None:
        try:
            vectors = binding.embed(texts)
        except RuntimeError:
            return None
        if len(vectors) != len(texts):
            return None
        return vectors

    def _save_graph_candidates(
        self,
        document: Document,
        segment: Segment,
        chunks: list[Chunk],
    ) -> None:
        document_node = GraphNode(
            node_id=document.doc_id,
            node_type="document",
            label=document.title,
            metadata={"source_id": document.source_id},
        )
        section_node = GraphNode(
            node_id=segment.segment_id,
            node_type="section",
            label=" > ".join(segment.toc_path),
            metadata={"doc_id": document.doc_id},
        )
        self.graph_repo.save_node(document_node)
        self.graph_repo.save_node(section_node)
        if not chunks:
            return
        edge = GraphEdge(
            edge_id=self._deterministic_id(document.doc_id, segment.segment_id, "edge"),
            from_node_id=document.doc_id,
            to_node_id=segment.segment_id,
            relation_type="contains",
            confidence=1.0,
            evidence_chunk_ids=[chunk.chunk_id for chunk in chunks if chunk.segment_id == segment.segment_id],
        )
        if edge.evidence_chunk_ids:
            self.graph_repo.save_candidate_edge(edge)

    def _parse_docling_text(
        self,
        *,
        content: str,
        location: str,
        title: str | None,
        owner: str,
        suffix: str,
    ) -> ParsedDocument:
        with NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        try:
            return self.docling_parser.parse(temp_path, location=location, title=title, owner=owner)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _deterministic_id(*parts: str) -> str:
        digest = sha256("\0".join(parts).encode("utf-8")).hexdigest()
        return f"{parts[-1]}-{digest[:12]}"


DeletePipelineRequest = DeleteRequest
RebuildPipelineRequest = RebuildRequest

__all__ = [
    "DeletePipeline",
    "DeletePipelineRequest",
    "DeletePipelineResult",
    "DeleteRequest",
    "IngestResult",
    "IngestService",
    "IngestPipeline",
    "IngestPipelineResult",
    "IngestRequest",
    "RebuildPipeline",
    "RebuildPipelineRequest",
    "RebuildPipelineResult",
    "RebuildRequest",
]
