from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pkp.schema._types.access import AccessPolicy


class SourceType(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"
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
