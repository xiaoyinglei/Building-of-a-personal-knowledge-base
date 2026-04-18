from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from rag.schema.core import Chunk, Document, GraphEdge, GraphNode, OcrResult, Segment, Source
    from rag.schema.query import KnowledgeArtifact, QueryUnderstanding


class Residency(StrEnum):
    CLOUD_ALLOWED = "cloud_allowed"
    LOCAL_PREFERRED = "local_preferred"
    LOCAL_REQUIRED = "local_required"


class ExternalRetrievalPolicy(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class RuntimeMode(StrEnum):
    FAST = "fast"
    DEEP = "deep"


class ExecutionLocation(StrEnum):
    CLOUD = "cloud"
    LOCAL = "local"


class ExecutionLocationPreference(StrEnum):
    CLOUD_FIRST = "cloud_first"
    LOCAL_FIRST = "local_first"
    LOCAL_ONLY = "local_only"


_RESIDENCY_ORDER: dict[Residency, int] = {
    Residency.CLOUD_ALLOWED: 0,
    Residency.LOCAL_PREFERRED: 1,
    Residency.LOCAL_REQUIRED: 2,
}


class AccessPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    residency: Residency = Residency.CLOUD_ALLOWED
    external_retrieval: ExternalRetrievalPolicy = ExternalRetrievalPolicy.ALLOW
    allowed_runtimes: frozenset[RuntimeMode] = Field(
        default_factory=lambda: frozenset({RuntimeMode.FAST, RuntimeMode.DEEP})
    )
    allowed_locations: frozenset[ExecutionLocation] = Field(
        default_factory=lambda: frozenset({ExecutionLocation.CLOUD, ExecutionLocation.LOCAL})
    )
    sensitivity_tags: frozenset[str] = Field(default_factory=frozenset)

    @classmethod
    def default(cls) -> AccessPolicy:
        return cls()

    def narrow(self, other: AccessPolicy) -> AccessPolicy:
        allowed_runtimes = self.allowed_runtimes & other.allowed_runtimes
        if not allowed_runtimes:
            raise ValueError("allowed_runtimes cannot become empty during narrowing")
        allowed_locations = self.allowed_locations & other.allowed_locations
        if not allowed_locations:
            raise ValueError("allowed_locations cannot become empty during narrowing")
        residency = max((self.residency, other.residency), key=_RESIDENCY_ORDER.__getitem__)
        external_retrieval = (
            ExternalRetrievalPolicy.DENY
            if ExternalRetrievalPolicy.DENY in {self.external_retrieval, other.external_retrieval}
            else ExternalRetrievalPolicy.ALLOW
        )
        return AccessPolicy(
            residency=residency,
            external_retrieval=external_retrieval,
            allowed_runtimes=allowed_runtimes,
            allowed_locations=allowed_locations,
            sensitivity_tags=self.sensitivity_tags | other.sensitivity_tags,
        )

    @property
    def local_only(self) -> bool:
        return self.residency is Residency.LOCAL_REQUIRED

    def allows_runtime(self, mode: RuntimeMode) -> bool:
        return mode in self.allowed_runtimes

    def allows_location(self, location: ExecutionLocation) -> bool:
        return location in self.allowed_locations


class ProviderAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    capability: str
    provider: str
    location: str
    model: str | None = None
    status: str
    error: str | None = None
    latency_ms: float | None = None


class RetrievalDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode_executor: str | None = None
    branch_hits: dict[str, int] = Field(default_factory=dict)
    branch_limits: dict[str, int] = Field(default_factory=dict)
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    reranked_benchmark_doc_ids: list[str] = Field(default_factory=list)
    embedding_provider: str | None = None
    rerank_provider: str | None = None
    attempts: list[ProviderAttempt] = Field(default_factory=list)
    fusion_input_count: int = 0
    fused_count: int = 0
    graph_expanded: bool = False
    query_understanding: QueryUnderstanding | None = None
    query_understanding_debug: dict[str, object] = Field(default_factory=dict)
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


class DocumentProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"
    REBUILDING = "rebuilding"


class DocumentPipelineStage(StrEnum):
    INGEST = "ingest"
    PARSE = "parse"
    ROUTE = "route"
    CHUNK = "chunk"
    EXTRACT = "extract"
    PERSIST = "persist"
    INDEX = "index"
    DELETE = "delete"
    REBUILD = "rebuild"


class DocumentStatusRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_id: str
    location: str
    content_hash: str
    status: DocumentProcessingStatus
    stage: DocumentPipelineStage | str
    attempts: int = 0
    error_message: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = Field(default_factory=dict)


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    namespace: str
    cache_key: str
    payload: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    category: str
    payload: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationMetricInput(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    citation_precision: float = Field(ge=0.0, le=1.0)
    evidence_sufficient: bool
    conflict_detected: bool
    simple_query_latency_seconds: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("latency_seconds", "simple_query_latency_seconds"),
    )
    deep_query_completion_quality: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("deep_quality", "deep_query_completion_quality"),
    )
    preservation_useful: bool


class EvaluationMetricSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_precision: float
    evidence_sufficiency_rate: float
    conflict_detection_quality: float
    simple_query_latency: float
    deep_query_completion_quality: float
    preservation_usefulness: float

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


def coerce_evaluation_metric_input(
    item: EvaluationMetricInput | Mapping[str, Any],
) -> EvaluationMetricInput:
    if isinstance(item, EvaluationMetricInput):
        return item
    return EvaluationMetricInput.model_validate(item)


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

    def merge_node_evidence(self, node_id: str, evidence_chunk_ids: Sequence[str]) -> None: ...

    def get_node(self, node_id: str) -> GraphNode | None: ...

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]: ...

    def list_nodes_by_alias(self, alias: str, *, node_type: str | None = None) -> list[GraphNode]: ...

    def list_node_evidence_chunk_ids(self, node_id: str) -> list[str]: ...

    def save_candidate_edge(self, edge: GraphEdge) -> None: ...

    def save_edge(self, edge: GraphEdge) -> None: ...

    def bind_node_evidence(self, node_id: str, chunk_ids: Sequence[str]) -> None: ...

    def promote_candidate_edge(self, edge_id: str) -> None: ...

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None: ...

    def list_candidate_edges(self) -> list[GraphEdge]: ...

    def list_edges(self) -> list[GraphEdge]: ...

    def delete_node(self, node_id: str) -> None: ...

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None: ...

    def list_edges_for_node(self, node_id: str, *, include_candidates: bool = False) -> list[GraphEdge]: ...

    def list_edges_for_chunk(self, chunk_id: str, *, include_candidates: bool = False) -> list[GraphEdge]: ...

    def delete_by_chunk_ids(self, chunk_ids: Sequence[str]) -> tuple[list[str], list[str]]: ...

    def close(self) -> None: ...


class FullTextSearchRepo(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        doc_ids: Sequence[str] | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> list[ChunkSearchResult]: ...

    def list_indexed_chunk_ids(self) -> set[str]: ...

    def delete_for_documents(self, doc_ids: Sequence[str]) -> int: ...

    def close(self) -> None: ...


class ObjectStore(Protocol):
    def put_bytes(self, path: str, payload: bytes, *, content_type: str | None = None) -> None: ...

    def get_bytes(self, path: str) -> bytes: ...

    def delete(self, path: str) -> None: ...

    def exists(self, path: str) -> bool: ...


__all__ = [
    "AccessPolicy",
    "CacheEntry",
    "CacheRepo",
    "CapabilityHealth",
    "ChunkSearchResult",
    "DocumentPipelineStage",
    "DocumentProcessingStatus",
    "DocumentStatusRecord",
    "EvaluationMetricInput",
    "EvaluationMetricSummary",
    "ExecutionLocation",
    "ExecutionLocationPreference",
    "ExternalRetrievalPolicy",
    "FullTextSearchRepo",
    "GraphNodeRecord",
    "GraphRepo",
    "HealthReport",
    "IndexHealth",
    "MetadataRepo",
    "ModelDiagnostics",
    "ModelProviderRepo",
    "ObjectStore",
    "OcrVisionRepo",
    "ProviderAttempt",
    "ProviderHealth",
    "QueryDiagnostics",
    "Residency",
    "RetrievalDiagnostics",
    "RetrievalRecord",
    "RuntimeMode",
    "SearchResult",
    "StoredVectorEntry",
    "TelemetryEvent",
    "VectorRepo",
    "VectorSearchResult",
    "VisualDescriptionRepo",
    "WebFetchRepo",
    "WebSearchRepo",
    "coerce_evaluation_metric_input",
]
