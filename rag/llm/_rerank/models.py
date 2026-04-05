from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag.schema._types.query import QueryUnderstanding


class RerankCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    parent_id: str | None = None
    text: str
    chunk_type: str
    section_path: list[str] = Field(default_factory=list)
    heading_text: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    dense_score: float | None = None
    sparse_score: float | None = None
    special_score: float | None = None
    structure_score: float | None = None
    metadata_score: float | None = None
    fusion_score: float | None = None
    rrf_score: float | None = None
    unified_rank: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)
    parent_text: str | None = None


class RerankRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    query_analysis: QueryUnderstanding
    candidate_list: list[RerankCandidate]
    top_k: int | None = None
    top_n: int | None = None
    debug: bool = False


class FeatureRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    feature_dict: dict[str, float | int | bool | str] = Field(default_factory=dict)


class RerankResultItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    rerank_score: float
    final_score: float
    rank_before: int
    rank_after: int
    feature_summary: dict[str, float | int | bool | str] = Field(default_factory=dict)
    channel_summary: list[str] = Field(default_factory=list)
    text: str
    doc_id: str
    parent_id: str | None = None
    chunk_type: str
    metadata: dict[str, str] = Field(default_factory=dict)
    drop_reason: str | None = None


class RerankResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    query_analysis: QueryUnderstanding
    model_name: str
    backend_name: str
    raw_candidates: list[RerankCandidate]
    feature_logs: list[FeatureRecord] = Field(default_factory=list)
    items: list[RerankResultItem] = Field(default_factory=list)
    dropped_items: list[RerankResultItem] = Field(default_factory=list)
