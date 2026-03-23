from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import OcrVisionRepo, ParsedDocument
from pkp.repo.models.fallback_embedding_repo import FallbackEmbeddingRepo
from pkp.repo.parse.image_parser_repo import ImageParserRepo
from pkp.repo.parse.markdown_parser_repo import MarkdownParserRepo
from pkp.repo.parse.pdf_parser_repo import PDFParserRepo
from pkp.repo.parse.plain_text_parser_repo import PlainTextParserRepo
from pkp.repo.parse.web_parser_repo import WebParserRepo
from pkp.repo.search.in_memory_vector_repo import InMemoryVectorRepo
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.storage.file_object_store import FileObjectStore
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.repo.vision.ocr_vision_repo import DeterministicOcrVisionRepo
from pkp.service.chunking_service import ChunkingService
from pkp.service.policy_resolution_service import PolicyResolutionService
from pkp.service.toc_service import TOCService
from pkp.types.access import AccessPolicy
from pkp.types.content import Chunk, Document, GraphEdge, GraphNode, Segment, Source, SourceType


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


class IngestService:
    def __init__(
        self,
        *,
        metadata_repo: SQLiteMetadataRepo,
        fts_repo: SQLiteFTSRepo,
        vector_repo: InMemoryVectorRepo,
        graph_repo: SQLiteGraphRepo,
        object_store: FileObjectStore,
        markdown_parser: MarkdownParserRepo,
        pdf_parser: PDFParserRepo,
        plain_text_parser: PlainTextParserRepo,
        image_parser: ImageParserRepo,
        web_parser: WebParserRepo,
        policy_resolution_service: PolicyResolutionService,
        toc_service: TOCService,
        chunking_service: ChunkingService,
        embedding_repo: FallbackEmbeddingRepo,
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
        self.policy_resolution_service = policy_resolution_service
        self.toc_service = toc_service
        self.chunking_service = chunking_service
        self.embedding_repo = embedding_repo

    @classmethod
    def create_in_memory(
        cls,
        root: Path,
        *,
        ocr_repo: OcrVisionRepo | None = None,
    ) -> IngestService:
        root.mkdir(parents=True, exist_ok=True)
        resolved_ocr_repo = ocr_repo or DeterministicOcrVisionRepo()
        return cls(
            metadata_repo=SQLiteMetadataRepo(root / "metadata.sqlite3"),
            fts_repo=SQLiteFTSRepo(root / "fts.sqlite3"),
            vector_repo=InMemoryVectorRepo(),
            graph_repo=SQLiteGraphRepo(root / "graph.sqlite3"),
            object_store=FileObjectStore(root / "objects"),
            markdown_parser=MarkdownParserRepo(),
            pdf_parser=PDFParserRepo(),
            plain_text_parser=PlainTextParserRepo(),
            image_parser=ImageParserRepo(resolved_ocr_repo),
            web_parser=WebParserRepo(),
            policy_resolution_service=PolicyResolutionService(),
            toc_service=TOCService(),
            chunking_service=ChunkingService(),
            embedding_repo=FallbackEmbeddingRepo(),
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
        parsed = self.markdown_parser.parse(
            markdown,
            location=location,
            title=title,
            owner=owner,
        )
        return self._ingest_parsed_document(
            location=location,
            source_type=SourceType.MARKDOWN,
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
    ) -> IngestResult:
        parsed = self.plain_text_parser.parse(
            text,
            location=location,
            title=title,
            owner=owner,
        )
        return self._ingest_parsed_document(
            location=location,
            source_type=SourceType.PLAIN_TEXT,
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
        parsed = self.pdf_parser.parse(pdf_path, location=location, title=title, owner=owner)
        return self._ingest_parsed_document(
            location=location,
            source_type=SourceType.PDF,
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
        parsed = self.image_parser.parse(image_path, location=location, title=title, owner=owner)
        return self._ingest_parsed_document(
            location=location,
            source_type=SourceType.IMAGE,
            raw_bytes=image_path.read_bytes(),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def ingest_web(
        self,
        *,
        location: str,
        html: str,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult:
        parsed = self.web_parser.parse(html, location=location, title=title, owner=owner)
        return self._ingest_parsed_document(
            location=location,
            source_type=SourceType.WEB,
            raw_bytes=html.encode("utf-8"),
            parsed=parsed,
            owner=owner,
            access_policy=access_policy,
        )

    def _ingest_parsed_document(
        self,
        *,
        location: str,
        source_type: SourceType,
        raw_bytes: bytes,
        parsed: ParsedDocument,
        owner: str,
        access_policy: AccessPolicy | None,
    ) -> IngestResult:
        normalized_policy = access_policy or AccessPolicy.default()
        content_hash = sha256(raw_bytes).hexdigest()
        existing_source = self.metadata_repo.get_source_by_location_and_hash(
            location,
            content_hash,
        )
        if existing_source is not None:
            existing_document = self.metadata_repo.get_active_document_by_location_and_hash(
                location,
                content_hash,
            )
            if existing_document is None:
                existing_document = self.metadata_repo.get_latest_document_for_location(location)
            if existing_document is None:
                raise RuntimeError("duplicate source exists without an active document")
            return IngestResult(
                source=existing_source,
                document=existing_document,
                segments=self.metadata_repo.list_segments(existing_document.doc_id),
                chunks=self.metadata_repo.list_chunks(existing_document.doc_id),
                is_duplicate=True,
                content_hash=content_hash,
                visible_text=parsed.visible_text,
                visual_semantics=parsed.visual_semantics,
            )

        latest_source = self.metadata_repo.get_latest_source_for_location(location)
        ingest_version = 1 if latest_source is None else latest_source.ingest_version + 1
        source_id = self._deterministic_id(location, content_hash, "source")
        object_key = self.object_store.put_bytes(raw_bytes, suffix=self._suffix_for(source_type))
        source = Source(
            source_id=source_id,
            source_type=source_type,
            location=location,
            owner=owner,
            content_hash=content_hash,
            effective_access_policy=normalized_policy,
            ingest_version=ingest_version,
            metadata={"object_key": object_key},
        )
        self.metadata_repo.save_source(source)
        self.metadata_repo.deactivate_documents_for_location(location)

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
        self.metadata_repo.save_document(
            document,
            location=location,
            content_hash=content_hash,
        )

        segments: list[Segment] = []
        chunks: list[Chunk] = []
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
                segment_id=self._deterministic_id(source_id, anchor, "segment"),
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
            self.metadata_repo.save_segment(segment)
            segments.append(segment)
            path_to_segment_id[tuple(normalized_path)] = segment.segment_id

            chunk_list = self.chunking_service.chunk_section(
                location=location,
                doc_id=document.doc_id,
                segment=segment,
                text=section.text,
                access_policy=self.policy_resolution_service.resolve_effective_access_policy(
                    source_policy=normalized_policy,
                    chunk_policy=normalized_policy,
                ),
            )
            for chunk in chunk_list:
                self.metadata_repo.save_chunk(chunk)
                chunks.append(chunk)
                self.fts_repo.index_chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_id=source.source_id,
                    title=document.title,
                    toc_path=list(segment.toc_path),
                    text=chunk.text,
                )
                vector = self.embedding_repo.embed([chunk.text])[0]
                self.vector_repo.upsert(
                    chunk.chunk_id,
                    vector,
                    metadata={
                        "doc_id": document.doc_id,
                        "segment_id": segment.segment_id,
                        "text": chunk.text,
                    },
                )

            self._save_graph_candidates(document, segment, chunk_list)

        return IngestResult(
            source=source,
            document=document,
            segments=segments,
            chunks=chunks,
            is_duplicate=False,
            content_hash=content_hash,
            visible_text=parsed.visible_text,
            visual_semantics=parsed.visual_semantics,
        )

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
            SourceType.IMAGE: ".png",
            SourceType.WEB: ".html",
            SourceType.PLAIN_TEXT: ".txt",
        }[source_type]

    @staticmethod
    def _deterministic_id(*parts: str) -> str:
        digest = sha256("\0".join(parts).encode("utf-8")).hexdigest()
        return f"{parts[-1]}-{digest[:12]}"
