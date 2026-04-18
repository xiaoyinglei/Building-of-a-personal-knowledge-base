from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
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


class DocumentStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    RETIRED = "retired"


class PiiStatus(StrEnum):
    UNKNOWN = "unknown"
    CLEAN = "clean"
    MASKED = "masked"
    RESTRICTED = "restricted"


class IndexingMode(StrEnum):
    EAGER = "eager"
    LAZY = "lazy"


class StorageTier(StrEnum):
    HOT = "hot"
    COLD = "cold"


class PartitionKey(StrEnum):
    HOT = "hot"
    COLD = "cold"


class ChunkRole(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    SPECIAL = "special"


class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: int = 0
    source_type: SourceType
    location: str
    original_file_name: str | None = None
    bucket: str | None = None
    object_key: str | None = None
    content_hash: str
    file_size_bytes: int | None = None
    mime_type: str | None = None
    owner_id: str | None = None
    ingest_version: int = 1
    pii_status: PiiStatus = PiiStatus.UNKNOWN
    effective_access_policy: AccessPolicy = Field(default_factory=AccessPolicy.default)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: int = 0
    source_id: int
    title: str | None = None
    doc_type: DocumentType
    language: str | None = None
    authors: list[str] = Field(default_factory=list)
    file_hash: str
    version_group_id: int = 0
    version_no: int = 1
    doc_status: DocumentStatus | str = DocumentStatus.PUBLISHED
    effective_date: datetime | None = None
    is_active: bool = True
    is_indexed: bool = False
    index_ready: bool = False
    index_priority: str = "high"
    indexing_mode: IndexingMode = IndexingMode.EAGER
    storage_tier: StorageTier = StorageTier.HOT
    pii_status: PiiStatus = PiiStatus.UNKNOWN
    reference_count: int = 1
    page_count: int | None = None
    tenant_id: str | None = None
    department_id: str | None = None
    auth_tag: str | None = None
    embedding_model_id: str = "default"
    indexed_at: datetime | None = None
    last_index_error: str | None = None
    effective_access_policy: AccessPolicy = Field(default_factory=AccessPolicy.default)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)


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

class SectionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: int = 0
    doc_id: int
    source_id: int
    parent_section_id: int | None = None
    toc_path: list[str] = Field(default_factory=list)
    heading_level: int | None = None
    order_index: int
    anchor: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    raw_locator: dict[str, Any] = Field(default_factory=dict)
    byte_range_start: int | None = None
    byte_range_end: int | None = None
    visible_text_key: str | None = None
    section_kind: str
    content_hash: str
    has_table: bool = False
    has_figure: bool = False
    neighbor_asset_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class AssetRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: int = 0
    doc_id: int
    source_id: int
    section_id: int | None = None
    asset_type: str
    element_ref: str | None = None
    page_no: int
    bbox: dict[str, Any] = Field(default_factory=dict)
    caption: str | None = None
    raw_locator: dict[str, Any] = Field(default_factory=dict)
    neighbor_section_id: int | None = None
    content_hash: str
    storage_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class DocSummaryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: int
    source_id: int
    version_group_id: int
    version_no: int = 1
    doc_status: DocumentStatus | str = DocumentStatus.PUBLISHED
    effective_date: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    index_ready: bool = True
    tenant_id: str | None = None
    department_id: str | None = None
    auth_tag: str | None = None
    source_type: SourceType | None = None
    embedding_model_id: str = "default"
    partition_key: PartitionKey = PartitionKey.HOT
    title: str | None = None
    summary_text: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class SectionSummaryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: int
    doc_id: int
    source_id: int
    version_group_id: int
    version_no: int = 1
    doc_status: DocumentStatus | str = DocumentStatus.PUBLISHED
    effective_date: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    index_ready: bool = True
    tenant_id: str | None = None
    department_id: str | None = None
    auth_tag: str | None = None
    source_type: SourceType | None = None
    embedding_model_id: str = "default"
    partition_key: PartitionKey = PartitionKey.HOT
    page_start: int | None = None
    page_end: int | None = None
    section_kind: str
    toc_path: list[str] = Field(default_factory=list)
    summary_text: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class AssetSummaryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: int
    doc_id: int
    source_id: int
    section_id: int | None = None
    version_group_id: int
    version_no: int = 1
    doc_status: DocumentStatus | str = DocumentStatus.PUBLISHED
    effective_date: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    index_ready: bool = True
    tenant_id: str | None = None
    department_id: str | None = None
    auth_tag: str | None = None
    embedding_model_id: str = "default"
    partition_key: PartitionKey = PartitionKey.HOT
    asset_type: str
    page_no: int | None = None
    caption: str | None = None
    summary_text: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class LayoutMetaCacheRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_id: int = 0
    source_id: int
    doc_id: int | None = None
    content_hash: str
    object_key: str | None = None
    layout_json: dict[str, Any] = Field(default_factory=dict)
    layout_version: str = "v1"
    page_count: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProcessingStateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: int
    source_id: int
    stage: str
    status: str
    attempts: int = 0
    priority: str = "normal"
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Chunk",
    "ChunkRole",
    "ChunkRoutingDecision",
    "ChunkStatistics",
    "ChunkingStrategy",
    "Document",
    "DocumentFeatures",
    "DocumentStatus",
    "DocumentProcessingPackage",
    "DocumentType",
    "GraphEdge",
    "GraphNode",
    "IndexingMode",
    "OcrRegion",
    "OcrResult",
    "ParsedDocument",
    "ParsedElement",
    "ParsedSection",
    "PartitionKey",
    "PiiStatus",
    "Segment",
    "Source",
    "SourceType",
    "StorageTier",

    "SectionRecord",
    "AssetRecord",
    "DocSummaryRecord",
    "SectionSummaryRecord",
    "AssetSummaryRecord",
    "LayoutMetaCacheRecord",
    "ProcessingStateRecord",
]
