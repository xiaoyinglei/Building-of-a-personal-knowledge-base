from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pkp.types.access import AccessPolicy, ExecutionLocationPreference
from pkp.types.content import ChunkRole
from pkp.types.diagnostics import ProviderAttempt
from pkp.types.envelope import EvidenceItem
from pkp.types.generation import GroundedAnswer
from pkp.types.retrieval import RetrievalResult


class QueryMode(StrEnum):
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
    mode: Literal["naive", "local", "global", "hybrid", "mix"] = "mix"
    source_scope: tuple[str, ...] = ()
    access_policy: AccessPolicy = field(default_factory=AccessPolicy.default)
    execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST
    max_context_tokens: int = 1200
    max_evidence_chunks: int = 8


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
        from pkp.query.retrieve import QueryPipeline, RAGQueryPipeline

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
