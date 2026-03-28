from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from pkp.algorithms.extract.entity_relation_extractor import (
    EntityRelationExtractionResult,
    EntityRelationExtractor,
    PromptedEntityRelationExtractor,
)
from pkp.algorithms.extract.entity_relation_merger import EntityRelationMerger
from pkp.repo.interfaces import EmbeddingProviderBinding, ParsedDocument, WebFetchRepo
from pkp.repo.models.fallback_embedding_repo import FallbackEmbeddingRepo
from pkp.repo.parse.docling_parser_repo import DoclingParserRepo
from pkp.repo.parse.image_parser_repo import ImageParserRepo
from pkp.repo.parse.markdown_parser_repo import MarkdownParserRepo
from pkp.repo.parse.pdf_parser_repo import PDFParserRepo
from pkp.repo.parse.plain_text_parser_repo import PlainTextParserRepo
from pkp.repo.parse.web_parser_repo import WebParserRepo
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.storage.file_object_store import FileObjectStore
from pkp.service.chunking_service import ChunkingService
from pkp.service.document_processing_service import DocumentProcessingService
from pkp.service.policy_resolution_service import PolicyResolutionService
from pkp.service.toc_service import TOCService
from pkp.stores.cache_store import CacheStore
from pkp.stores.chunk_store import ChunkStore
from pkp.stores.document_store import DocumentStore
from pkp.stores.graph_store import GraphStore
from pkp.stores.status_store import StatusStore
from pkp.stores.vector_store import VectorStore
from pkp.types.access import AccessPolicy
from pkp.types.content import Chunk, ChunkRole, Document, GraphEdge, GraphNode, Segment, Source, SourceType
from pkp.types.processing import DocumentProcessingPackage
from pkp.types.storage import CacheEntry, DocumentPipelineStage, DocumentProcessingStatus, DocumentStatusRecord


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


@dataclass(slots=True)
class IngestPipeline:
    documents: DocumentStore
    chunks: ChunkStore
    vectors: VectorStore
    graph: GraphStore
    status: StatusStore
    cache: CacheStore
    fts_repo: SQLiteFTSRepo
    object_store: FileObjectStore | None
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
    embedding_bindings: tuple[EmbeddingProviderBinding, ...] = ()
    extractor: EntityRelationExtractor | None = None
    merger: EntityRelationMerger | None = None

    def __post_init__(self) -> None:
        if not self.embedding_bindings:
            self.embedding_bindings = (
                EmbeddingProviderBinding(provider=FallbackEmbeddingRepo(), space="default"),
            )
        if self.extractor is None:
            provider = next(
                (
                    cast(object, binding.provider)
                    for binding in self.embedding_bindings
                    if callable(getattr(binding.provider, "chat", None))
                ),
                None,
            )
            self.extractor = PromptedEntityRelationExtractor(
                model_provider=cast(object, provider),
            )
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
                metadata=section.metadata | {"location": location},
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
        merged = self.merger.merge(document=document, extraction=extraction)

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

        self._index_entity_vectors(source=source, document=document, entities=merged.entities)
        self._index_relation_vectors(source=source, document=document, relations=merged.relations)
        return len(merged.entities), len(merged.relations)

    def _extract_entities_and_relations(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
    ) -> EntityRelationExtractionResult:
        entities = []
        relations = []
        for chunk in chunks:
            cache_key = f"{chunk.content_hash or chunk.chunk_id}::entity_relation::v1"
            cached = self.cache.get(cache_key, namespace="extract")
            if cached is not None and isinstance(cached.payload, dict):
                try:
                    result = EntityRelationExtractionResult.model_validate(cached.payload)
                except Exception:
                    result = self.extractor.extract(document=document, chunks=[chunk])
            else:
                result = self.extractor.extract(document=document, chunks=[chunk])
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

    def _index_entity_vectors(self, *, source: Source, document: Document, entities: list[object]) -> None:
        texts = [f"{entity.label}: {entity.description}" for entity in entities]
        if not texts:
            return
        for binding in self.embedding_bindings:
            vectors = self._embed_texts(binding.provider, texts)
            if vectors is None:
                continue
            for entity, vector in zip(entities, vectors, strict=True):
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
                        "text": f"{entity.label}: {entity.description}",
                        "entity_type": entity.entity_type,
                    },
                    embedding_space=binding.space,
                )

    def _index_relation_vectors(self, *, source: Source, document: Document, relations: list[object]) -> None:
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
        for binding in self.embedding_bindings:
            vectors = self._embed_texts(binding.provider, texts)
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
        for binding in self.embedding_bindings:
            missing_vector_ids = set(chunks and [chunk.chunk_id for chunk in chunks]) - self.vectors.existing_chunk_ids(
                [chunk.chunk_id for chunk in chunks],
                embedding_space=binding.space,
            )
            if not missing_vector_ids:
                continue
            missing_chunks = [chunk for chunk in chunks if chunk.chunk_id in missing_vector_ids]
            vectors = self._embed_texts(binding.provider, [chunk.text for chunk in missing_chunks])
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
        for binding in self.embedding_bindings:
            vectors = self._embed_texts(binding.provider, texts)
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

        if source_type in {SourceType.PDF, SourceType.DOCX}:
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
        attempts = 1 if existing is None else existing.attempts + (1 if status is DocumentProcessingStatus.FAILED else 0)
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
    def _embed_texts(provider: object, texts: list[str]) -> list[list[float]] | None:
        embed = getattr(provider, "embed", None)
        if not callable(embed):
            return None
        try:
            vectors = cast(list[list[float]], embed(texts))
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
