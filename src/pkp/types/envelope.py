from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pkp.types.access import AccessPolicy, ExecutionLocationPreference, RuntimeMode
from pkp.types.content import ChunkRole
from pkp.types.diagnostics import QueryDiagnostics
from pkp.types.query import ComplexityLevel, TaskType


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

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


class PreservationSuggestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    suggested: bool
    artifact_id: str | None = None
    artifact_type: str | None = None
    title: str | None = None
    rationale: str | None = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    conclusion: str
    evidence: list[EvidenceItem]
    differences_or_conflicts: list[str] = Field(default_factory=list)
    uncertainty: str
    preservation_suggestion: PreservationSuggestion
    runtime_mode: RuntimeMode
    diagnostics: QueryDiagnostics = Field(default_factory=QueryDiagnostics)


class ExecutionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    effective_access_policy: AccessPolicy
    task_type: TaskType
    complexity_level: ComplexityLevel
    latency_budget: int
    cost_budget: float
    token_budget: int | None = None
    execution_location_preference: ExecutionLocationPreference
    fallback_allowed: bool
    source_scope: list[str] = Field(default_factory=list)
    allowed_runtimes: frozenset[RuntimeMode] = Field(
        default_factory=lambda: frozenset({RuntimeMode.FAST, RuntimeMode.DEEP})
    )
