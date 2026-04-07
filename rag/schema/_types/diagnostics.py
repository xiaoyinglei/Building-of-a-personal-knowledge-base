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
    query_understanding_debug: dict[str, object] = Field(default_factory=dict)
    parent_backfilled_count: int = 0
    collapsed_candidate_count: int = 0 #候选合并/折叠后，减少到了多少。


class ModelDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    synthesis_provider: str | None = None
    attempts: list[ProviderAttempt] = Field(default_factory=list)
    fallback_reason: str | None = None
    failed_stage: str | None = None
    degraded_to_retrieval_only: bool = False  #是不是退化成“只返回检索结果，不做生成


class QueryDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    model: ModelDiagnostics = Field(default_factory=ModelDiagnostics)

#某项能力的健康状态
class CapabilityHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured: bool #是否配置了这个能力，比如是否配置了文本生成能力，或者检索能力等。
    available: bool #在当前时刻是否可用，比如虽然配置了文本生成能力，但由于外部依赖不可用，导致当前不可用。
    model: str | None = None
    error: str | None = None


class ProviderHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    location: str
    capabilities: dict[str, CapabilityHealth] = Field(default_factory=dict)

#索引/知识库的健康状态
class IndexHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: int = 0 #知识库里有多少文档
    chunks: int = 0 #知识库里有多少chunk
    vectors: int = 0 #知识库里有多少向量
    missing_vectors: int = 0 #知识库里有多少chunk缺失向量


class HealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str #整体健康状态，比如"healthy", "degraded", "unhealthy"等
    providers: list[ProviderHealth] = Field(default_factory=list) #每个外部服务提供者的健康状态
    indices: IndexHealth = Field(default_factory=IndexHealth) #索引/知识库的健康状态
