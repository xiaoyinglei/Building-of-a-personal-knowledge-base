from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pkp.schema._types.content import Chunk, ChunkRole, Document, Source, SourceType


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
    heading_quality_score: float
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


__all__ = [
    "ChunkingStrategy",
    "ChunkRole",
    "ChunkRoutingDecision",
    "ChunkStatistics",
    "DocumentFeatures",
    "DocumentProcessingPackage",
]
