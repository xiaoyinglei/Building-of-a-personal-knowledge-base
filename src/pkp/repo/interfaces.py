from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pkp.types.content import DocumentType


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
class ParsedDocument:
    title: str
    doc_type: DocumentType
    authors: list[str]
    language: str
    sections: list[ParsedSection]
    visible_text: str
    visual_semantics: str | None = None
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
