from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag.schema._types.query import QueryUnderstanding


class ProviderAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    capability: str
    provider: str
    location: str
    model: str | None = None
    status: str
    error: str | None = None


class RetrievalDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode_executor: str | None = None
    branch_hits: dict[str, int] = Field(default_factory=dict)
    branch_limits: dict[str, int] = Field(default_factory=dict)
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    embedding_provider: str | None = None
    rerank_provider: str | None = None
    attempts: list[ProviderAttempt] = Field(default_factory=list)
    fusion_input_count: int = 0
    fused_count: int = 0
    graph_expanded: bool = False
    query_understanding: QueryUnderstanding | None = None
    parent_backfilled_count: int = 0
    collapsed_candidate_count: int = 0


class ModelDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    synthesis_provider: str | None = None
    attempts: list[ProviderAttempt] = Field(default_factory=list)
    fallback_reason: str | None = None
    failed_stage: str | None = None
    degraded_to_retrieval_only: bool = False


class QueryDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    model: ModelDiagnostics = Field(default_factory=ModelDiagnostics)


class CapabilityHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured: bool
    available: bool
    model: str | None = None
    error: str | None = None


class ProviderHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    location: str
    capabilities: dict[str, CapabilityHealth] = Field(default_factory=dict)


class IndexHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    missing_vectors: int = 0


class HealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    providers: list[ProviderHealth] = Field(default_factory=list)
    indices: IndexHealth = Field(default_factory=IndexHealth)
