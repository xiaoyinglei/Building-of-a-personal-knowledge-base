from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from rag.schema._types.artifact import KnowledgeArtifact
from rag.schema._types.content import DocumentType, SourceType
from rag.schema._types.storage import CacheEntry, DocumentStatusRecord
from rag.schema.chunk import Chunk
from rag.schema.document import Document, Segment, Source
from rag.schema.graph import GraphEdge, GraphNode


@dataclass(frozen=True)
class ParsedSection:
    toc_path: tuple[str, ...]
    heading_level: int | None
    page_range: tuple[int, int] | None
    order_index: int
    text: str
    anchor_hint: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedElement:
    element_id: str
    kind: str
    text: str
    toc_path: tuple[str, ...] = ()
    heading_level: int | None = None
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    parent_ref: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    source_type: SourceType
    doc_type: DocumentType
    authors: list[str]
    language: str
    sections: list[ParsedSection]
    visible_text: str
    visual_semantics: str | None = None
    elements: list[ParsedElement] = field(default_factory=list)
    page_count: int | None = None
    doc_model: Any | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkSearchResult:
    chunk_id: str
    doc_id: str
    source_id: str
    title: str
    toc_path: tuple[str, ...]
    snippet: str
    score: float


@dataclass(frozen=True)
class RetrievalRecord:
    item_id: str
    item_kind: str = "chunk"
    doc_id: str = ""
    source_id: str = ""
    segment_id: str = ""
    text: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.item_id


@dataclass(frozen=True)
class VectorSearchResult(RetrievalRecord):
    score: float = 0.0


@dataclass(frozen=True)
class StoredVectorEntry:
    item_id: str
    item_kind: str
    embedding_space: str
    doc_id: str
    segment_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class GraphNodeRecord:
    node_id: str
    node_type: str
    label: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float = 0.0
    source: str = "web"


@dataclass(frozen=True)
class OcrRegion:
    text: str
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class OcrResult:
    visible_text: str
    visual_semantics: str
    regions: list[OcrRegion] = field(default_factory=list)


class OcrVisionRepo(Protocol):
    def extract(self, image_path: Path) -> OcrResult: ...


class VisualDescriptionRepo(Protocol):
    def describe_visual(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        prompt: str | None = None,
    ) -> str: ...


class WebFetchRepo(Protocol):
    def fetch(self, location: str) -> str: ...


class WebSearchRepo(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]: ...


class ModelProviderRepo(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def chat(self, prompt: str) -> str: ...

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]: ...


class VectorRepo(Protocol):
    def upsert(
        self,
        item_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> None: ...

    def search(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> list[VectorSearchResult]: ...

    def get_entry(
        self,
        item_id: str,
        *,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> StoredVectorEntry | None: ...

    def existing_item_ids(
        self,
        item_ids: Sequence[str],
        *,
        embedding_space: str | None = None,
        item_kind: str | None = "chunk",
    ) -> set[str]: ...

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_chunks: bool = False,
    ) -> int: ...

    def delete_for_documents(
        self,
        doc_ids: Sequence[str],
        *,
        item_kind: str | None = None,
    ) -> int: ...

    def close(self) -> None: ...


class MetadataRepo(Protocol):
    def save_source(self, source: Source) -> None: ...

    def get_source(self, source_id: str) -> Source | None: ...

    def get_source_by_location_and_hash(self, location: str, content_hash: str) -> Source | None: ...

    def find_source_by_content_hash(self, content_hash: str) -> Source | None: ...

    def get_latest_source_for_location(self, location: str) -> Source | None: ...

    def list_sources(self, location: str | None = None) -> list[Source]: ...

    def save_document(
        self,
        document: Document,
        *,
        location: str,
        content_hash: str,
        active: bool = True,
    ) -> None: ...

    def get_document(self, doc_id: str) -> Document | None: ...

    def is_document_active(self, doc_id: str) -> bool: ...

    def list_documents(self, source_id: str | None = None, *, active_only: bool = False) -> list[Document]: ...

    def get_active_document_by_location_and_hash(self, location: str, content_hash: str) -> Document | None: ...

    def get_latest_document_for_location(self, location: str) -> Document | None: ...

    def deactivate_documents_for_location(self, location: str) -> None: ...

    def set_document_active(self, doc_id: str, *, active: bool) -> None: ...

    def save_segment(self, segment: Segment) -> None: ...

    def get_segment(self, segment_id: str) -> Segment | None: ...

    def list_segments(self, doc_id: str) -> list[Segment]: ...

    def delete_segments_for_document(self, doc_id: str) -> int: ...

    def save_chunk(self, chunk: Chunk) -> None: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    def list_chunks(self, doc_id: str) -> list[Chunk]: ...

    def list_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...

    def delete_chunks_for_document(self, doc_id: str) -> int: ...

    def save_artifact(self, artifact: KnowledgeArtifact) -> None: ...

    def get_artifact(self, artifact_id: str) -> KnowledgeArtifact | None: ...

    def list_artifacts(self) -> list[KnowledgeArtifact]: ...

    def save_document_status(self, status: DocumentStatusRecord) -> DocumentStatusRecord: ...

    def get_document_status(self, doc_id: str) -> DocumentStatusRecord | None: ...

    def list_document_statuses(
        self,
        *,
        source_id: str | None = None,
        status: str | None = None,
    ) -> list[DocumentStatusRecord]: ...

    def delete_document_status(self, doc_id: str) -> None: ...

    def close(self) -> None: ...


class CacheRepo(Protocol):
    def save_cache_entry(self, entry: CacheEntry) -> CacheEntry: ...

    def get_cache_entry(self, cache_key: str, *, namespace: str = "default") -> CacheEntry | None: ...

    def list_cache_entries(self, *, namespace: str | None = None) -> list[CacheEntry]: ...

    def delete_cache_entry(self, cache_key: str, *, namespace: str = "default") -> None: ...

    def purge_expired_cache_entries(self, *, now: datetime | None = None) -> int: ...

    def close(self) -> None: ...


class GraphRepo(Protocol):
    def save_node(self, node: GraphNode) -> None: ...

    def merge_node_evidence(self, node_id: str, chunk_ids: Sequence[str]) -> None: ...

    def bind_node_evidence(self, node_id: str, chunk_ids: Sequence[str]) -> None: ...

    def get_node(self, node_id: str) -> GraphNode | None: ...

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]: ...

    def list_nodes_by_alias(self, alias: str, *, node_type: str | None = None) -> list[GraphNode]: ...

    def list_nodes_for_chunk(self, chunk_id: str) -> list[GraphNode]: ...

    def list_node_evidence_chunk_ids(self, node_id: str) -> list[str]: ...

    def save_candidate_edge(self, edge: GraphEdge) -> None: ...

    def save_edge(self, edge: GraphEdge) -> None: ...

    def promote_candidate_edge(self, edge_id: str) -> None: ...

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None: ...

    def list_candidate_edges(self) -> list[GraphEdge]: ...

    def list_edges(self) -> list[GraphEdge]: ...

    def list_edges_for_node(self, node_id: str, *, include_candidates: bool = False) -> list[GraphEdge]: ...

    def list_edges_for_chunk(self, chunk_id: str, *, include_candidates: bool = False) -> list[GraphEdge]: ...

    def delete_node(self, node_id: str) -> None: ...

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None: ...

    def delete_by_chunk_ids(self, chunk_ids: Sequence[str]) -> tuple[list[str], list[str]]: ...

    def close(self) -> None: ...


class FullTextSearchRepo(Protocol):
    def index_chunk(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        source_id: str,
        title: str,
        toc_path: list[str],
        text: str,
    ) -> None: ...

    def search(self, query: str, *, limit: int = 10, doc_ids: list[str] | None = None) -> list[ChunkSearchResult]: ...

    def delete_by_chunk_ids(self, chunk_ids: Sequence[str]) -> int: ...

    def close(self) -> None: ...


class ObjectStore(Protocol):
    def put_bytes(self, content: bytes, *, suffix: str = "") -> str: ...

    def read_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def path_for_key(self, key: str) -> Path: ...
