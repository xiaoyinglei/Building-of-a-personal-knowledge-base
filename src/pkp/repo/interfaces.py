from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pkp.types.content import DocumentType, SourceType


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
class VectorSearchResult:
    item_id: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.item_id


@dataclass(frozen=True)
class EmbeddingProviderBinding:
    provider: object
    space: str
    location: str = "runtime"


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

    def existing_item_ids(
        self,
        item_ids: Sequence[str],
        *,
        embedding_space: str | None = None,
        item_kind: str | None = "chunk",
    ) -> set[str]: ...
