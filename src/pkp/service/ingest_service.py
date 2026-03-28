from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from pkp.core.pipelines.ingest_pipeline import IngestPipeline, IngestPipelineResult
from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import (
    EmbeddingProviderBinding,
    ModelProviderRepo,
    OcrVisionRepo,
    ParsedDocument,
    VectorRepo,
    WebFetchRepo,
)
from pkp.repo.models.fallback_embedding_repo import FallbackEmbeddingRepo
from pkp.repo.parse.docling_parser_repo import DoclingParserRepo
from pkp.repo.parse.image_parser_repo import ImageParserRepo
from pkp.repo.parse.markdown_parser_repo import MarkdownParserRepo
from pkp.repo.parse.pdf_parser_repo import PDFParserRepo
from pkp.repo.parse.plain_text_parser_repo import PlainTextParserRepo
from pkp.repo.parse.web_fetch_repo import WebFetchRepo as HttpWebFetchRepo
from pkp.repo.parse.web_parser_repo import WebParserRepo
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.search.sqlite_vector_repo import SQLiteVectorRepo
from pkp.repo.storage.file_object_store import FileObjectStore
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.repo.vision.ocr_vision_repo import create_default_ocr_repo
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
        metadata_repo: SQLiteMetadataRepo,
        fts_repo: SQLiteFTSRepo,
        vector_repo: VectorRepo,
        graph_repo: SQLiteGraphRepo,
        object_store: FileObjectStore,
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
        embedding_repo: ModelProviderRepo | None = None,
        embedding_bindings: tuple[EmbeddingProviderBinding, ...] = (),
    ) -> None:
        self.metadata_repo = metadata_repo
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
        if embedding_bindings:
            self.embedding_bindings = tuple(embedding_bindings)
        else:
            self.embedding_bindings = (
                EmbeddingProviderBinding(
                    provider=embedding_repo or FallbackEmbeddingRepo(),
                    space="default",
                ),
            )
        self.ingest_pipeline = IngestPipeline(
            documents=DocumentStore(metadata_repo=self.metadata_repo),
            chunks=ChunkStore(metadata_repo=self.metadata_repo),
            vectors=VectorStore(vector_repo=cast(SQLiteVectorRepo, self.vector_repo)),
            graph=GraphStore(graph_repo=self.graph_repo),
            status=StatusStore(metadata_repo=self.metadata_repo),
            cache=CacheStore(metadata_repo=self.metadata_repo),
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
            embedding_bindings=self.embedding_bindings,
        )

    @classmethod
    def create_in_memory(
        cls,
        root: Path,
        *,
        ocr_repo: OcrVisionRepo | None = None,
        web_fetch_repo: WebFetchRepo | None = None,
    ) -> IngestService:
        root.mkdir(parents=True, exist_ok=True)
        resolved_ocr_repo = ocr_repo or create_default_ocr_repo()
        resolved_web_fetch_repo = web_fetch_repo or HttpWebFetchRepo()
        return cls(
            metadata_repo=SQLiteMetadataRepo(root / "metadata.sqlite3"),
            fts_repo=SQLiteFTSRepo(root / "fts.sqlite3"),
            vector_repo=cast(VectorRepo, SQLiteVectorRepo(root / "vectors.sqlite3")),
            graph_repo=SQLiteGraphRepo(root / "graph.sqlite3"),
            object_store=FileObjectStore(root / "objects"),
            markdown_parser=MarkdownParserRepo(),
            pdf_parser=PDFParserRepo(),
            plain_text_parser=PlainTextParserRepo(),
            image_parser=ImageParserRepo(resolved_ocr_repo),
            web_parser=WebParserRepo(),
            web_fetch_repo=resolved_web_fetch_repo,
            docling_parser=DoclingParserRepo(resolved_ocr_repo),
            policy_resolution_service=PolicyResolutionService(),
            toc_service=TOCService(),
            chunking_service=ChunkingService(),
            document_processing_service=DocumentProcessingService(toc_service=TOCService()),
            embedding_bindings=(
                EmbeddingProviderBinding(
                    provider=FallbackEmbeddingRepo(),
                    space="default",
                ),
            ),
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
        parsed = self.docling_parser.parse(pdf_path, location=location, title=title, owner=owner)
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

    def _index_chunk_vectors(
        self,
        *,
        document: Document,
        segment: Segment,
        chunks: list[Chunk],
    ) -> None:
        if not chunks:
            return

        embedded_any = False
        for binding in self.embedding_bindings:
            vectors = self._embed_texts(binding.provider, [chunk.text for chunk in chunks])
            if vectors is None:
                continue
            embedded_any = True
            for chunk, vector in zip(chunks, vectors, strict=True):
                self.vector_repo.upsert(
                    chunk.chunk_id,
                    vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "segment_id": segment.segment_id,
                        "text": chunk.text,
                    },
                    embedding_space=binding.space,
                )

        if not embedded_any:
            raise RuntimeError("No embedding provider available for ingest")

    def _repair_chunk_vectors(
        self,
        *,
        document: Document,
        chunks: list[Chunk],
    ) -> int:
        repaired = 0
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        for binding in self.embedding_bindings:
            missing_vector_ids = set(chunk_ids) - self.vector_repo.existing_item_ids(
                chunk_ids,
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
