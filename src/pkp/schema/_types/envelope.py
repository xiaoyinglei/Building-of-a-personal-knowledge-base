from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pkp.schema._types.access import AccessPolicy, ExecutionLocationPreference, RuntimeMode
from pkp.schema._types.content import ChunkRole
from pkp.schema._types.diagnostics import QueryDiagnostics
from pkp.schema._types.generation import AnswerCitation, AnswerEvidenceLink, AnswerSection
from pkp.schema._types.query import ComplexityLevel, TaskType


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
    file_name: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    chunk_type: str | None = None
    source_type: str | None = None


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
    answer_text: str | None = None
    answer_sections: list[AnswerSection] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    evidence_links: list[AnswerEvidenceLink] = Field(default_factory=list)
    groundedness_flag: bool = False
    insufficient_evidence_flag: bool = False


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
