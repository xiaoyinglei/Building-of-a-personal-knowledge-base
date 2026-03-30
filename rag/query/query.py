from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.schema._types.access import AccessPolicy, ExecutionLocationPreference
from rag.schema._types.content import ChunkRole
from rag.schema._types.diagnostics import ProviderAttempt
from rag.schema._types.envelope import EvidenceItem
from rag.schema._types.generation import GroundedAnswer
from rag.schema._types.retrieval import RetrievalResult

if TYPE_CHECKING:
    from rag.query.retrieve import QueryPipeline, RAGQueryPipeline


class QueryMode(StrEnum):
    BYPASS = "bypass"
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"


def normalize_query_mode(mode: QueryMode | str | None) -> QueryMode:
    if mode is None:
        return QueryMode.MIX
    if isinstance(mode, QueryMode):
        return mode
    return QueryMode(mode)


@dataclass(frozen=True, slots=True)
class QueryOptions:
    mode: Literal["bypass", "naive", "local", "global", "hybrid", "mix"] = "mix"
    source_scope: tuple[str, ...] = ()
    access_policy: AccessPolicy = field(default_factory=AccessPolicy.default)
    execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST
    max_context_tokens: int = 1200
    max_evidence_chunks: int = 8
    top_k: int = 8
    chunk_top_k: int | None = None
    response_type: str = "Multiple Paragraphs"
    user_prompt: str | None = None
    conversation_history: tuple[tuple[str, str], ...] = ()
    enable_rerank: bool = True


class ContextEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    chunk_id: str
    doc_id: str
    source_id: str | None = None
    citation_anchor: str
    text: str
    score: float
    evidence_kind: str = "internal"
    chunk_role: ChunkRole | None = None
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    file_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_type: str | None = None
    source_type: str | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    retrieval_family: str | None = None
    token_count: int
    selected_token_count: int
    truncated: bool = False

    def as_evidence_item(self) -> EvidenceItem:
        return EvidenceItem(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            source_id=self.source_id,
            citation_anchor=self.citation_anchor,
            text=self.text,
            score=self.score,
            evidence_kind=self.evidence_kind,
            chunk_role=self.chunk_role,
            special_chunk_type=self.special_chunk_type,
            parent_chunk_id=self.parent_chunk_id,
            file_name=self.file_name,
            section_path=self.section_path,
            page_start=self.page_start,
            page_end=self.page_end,
            chunk_type=self.chunk_type,
            source_type=self.source_type,
            retrieval_channels=list(self.retrieval_channels),
            retrieval_family=self.retrieval_family,
        )


class BuiltContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence: list[ContextEvidence] = Field(default_factory=list)
    token_budget: int
    token_count: int
    truncated_count: int = 0
    grounded_candidate: str
    prompt: str


class RAGQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    mode: str
    answer: GroundedAnswer
    retrieval: RetrievalResult
    context: BuiltContext
    generation_provider: str | None = None
    generation_model: str | None = None
    generation_attempts: list[ProviderAttempt] = Field(default_factory=list)


def __getattr__(name: str) -> object:
    if name in {"QueryPipeline", "RAGQueryPipeline"}:
        from rag.query.retrieve import QueryPipeline, RAGQueryPipeline

        return {
            "QueryPipeline": QueryPipeline,
            "RAGQueryPipeline": RAGQueryPipeline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BuiltContext",
    "ContextEvidence",
    "QueryMode",
    "QueryOptions",
    "QueryPipeline",
    "RAGQueryPipeline",
    "RAGQueryResult",
    "normalize_query_mode",
]
