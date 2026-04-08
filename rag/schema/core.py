from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.schema.runtime import AccessPolicy


class SourceType(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    IMAGE = "image"
    WEB = "web"
    PLAIN_TEXT = "plain_text"
    PASTED_TEXT = "pasted_text"
    BROWSER_CLIP = "browser_clip"


class DocumentType(StrEnum):
    ARTICLE = "article"
    NOTE = "note"
    REPORT = "report"
    IMAGE = "image"
    WEB_PAGE = "web_page"


class ChunkRole(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    SPECIAL = "special"


class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    source_type: SourceType
    location: str
    owner: str
    content_hash: str
    effective_access_policy: AccessPolicy
    ingest_version: int
    metadata: dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_id: str
    doc_type: DocumentType
    title: str
    authors: list[str]
    created_at: datetime
    language: str
    effective_access_policy: AccessPolicy
    metadata: dict[str, str] = Field(default_factory=dict)


class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment_id: str
    doc_id: str
    parent_segment_id: str | None
    toc_path: list[str]
    heading_level: int | None
    page_range: tuple[int, int] | None
    order_index: int
    anchor: str | None = None
    visible_text: str | None = None
    visual_semantics: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    segment_id: str
    doc_id: str
    text: str
    token_count: int
    citation_anchor: str
    citation_span: tuple[int, int]
    effective_access_policy: AccessPolicy
    extraction_quality: float
    embedding_ref: str | None
    order_index: int = 0
    chunk_role: ChunkRole = ChunkRole.CHILD
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    content_hash: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class GraphNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    node_type: str
    label: str
    metadata: dict[str, str] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    from_node_id: str
    to_node_id: str
    relation_type: str
    confidence: float
    evidence_chunk_ids: list[str]
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("evidence_chunk_ids")
    @classmethod
    def validate_evidence_chunk_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("evidence_chunk_ids must not be empty")
        return value


class ChunkingStrategy(StrEnum):
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"
    IMAGE = "image"


class DocumentFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_type: SourceType
    section_count: int
    word_count: int
    heading_count: int
    table_count: int
    figure_count: int
    caption_count: int
    ocr_region_count: int
    avg_section_words: float
    structure_depth: int
    has_dense_structure: bool
    metadata: dict[str, str] = Field(default_factory=dict)


class ChunkRoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_type: SourceType
    selected_strategy: ChunkingStrategy
    special_chunk_mode: bool = True
    local_refine: bool = False
    fallback: bool = False
    reasons: list[str] = Field(default_factory=list)
    debug: dict[str, str] = Field(default_factory=dict)


class ChunkStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    parent_chunk_count: int
    child_chunk_count: int
    special_chunk_count: int
    total_chunks: int
    deduplicated_chunks: int = 0
    merged_chunks: int = 0


class DocumentProcessingPackage(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Source
    document: Document
    analysis: DocumentFeatures
    routing: ChunkRoutingDecision
    parent_chunks: list[Chunk]
    child_chunks: list[Chunk]
    special_chunks: list[Chunk]
    metadata_summary: dict[str, str | int | float | bool] = Field(default_factory=dict)
    stats: ChunkStatistics


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
class OcrRegion:
    text: str
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class OcrResult:
    visible_text: str
    visual_semantics: str
    regions: list[OcrRegion] = field(default_factory=list)


__all__ = [
    "Chunk",
    "ChunkRole",
    "ChunkRoutingDecision",
    "ChunkStatistics",
    "ChunkingStrategy",
    "Document",
    "DocumentFeatures",
    "DocumentProcessingPackage",
    "DocumentType",
    "GraphEdge",
    "GraphNode",
    "OcrRegion",
    "OcrResult",
    "ParsedDocument",
    "ParsedElement",
    "ParsedSection",
    "Segment",
    "Source",
    "SourceType",
]
