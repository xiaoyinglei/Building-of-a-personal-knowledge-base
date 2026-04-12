from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag.schema.core import ChunkRole
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference, QueryDiagnostics, RuntimeMode


class TaskType(StrEnum):
    LOOKUP = "lookup"
    SINGLE_DOC_QA = "single_doc_qa"
    COMPARISON = "comparison"
    SYNTHESIS = "synthesis"
    TIMELINE = "timeline"
    RESEARCH = "research"


class ComplexityLevel(StrEnum):
    L1_DIRECT = "L1_direct"
    L2_SCOPED = "L2_scoped"
    L3_COMPARATIVE = "L3_comparative"
    L4_RESEARCH = "L4_research"


class QueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    preferred_runtime: str | None = None
    source_scope: list[str] = Field(default_factory=list)
    metadata_filters: dict[str, str] = Field(default_factory=dict)


class ResearchSubQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str
    purpose: str


class PageRangeConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int


class StructureConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_strategy: str = "none"
    requires_structure_match: bool = False
    prefer_heading_match: bool = False
    semantic_section_families: list[str] = Field(default_factory=list)
    preferred_section_terms: list[str] = Field(default_factory=list)
    heading_hints: list[str] = Field(default_factory=list)
    title_hints: list[str] = Field(default_factory=list)
    locator_terms: list[str] = Field(default_factory=list)

    def has_constraints(self) -> bool:
        return any(
            (
                self.requires_structure_match,
                self.prefer_heading_match,
                bool(self.semantic_section_families),
                bool(self.preferred_section_terms),
                bool(self.heading_hints),
                bool(self.title_hints),
                bool(self.locator_terms),
            )
        )


class MetadataFilters(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_numbers: list[int] = Field(default_factory=list)
    page_ranges: list[PageRangeConstraint] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    document_titles: list[str] = Field(default_factory=list)
    file_names: list[str] = Field(default_factory=list)

    def has_constraints(self) -> bool:
        return any(
            (
                bool(self.page_numbers),
                bool(self.page_ranges),
                bool(self.source_types),
                bool(self.document_titles),
                bool(self.file_names),
            )
        )


class PolicyHints(BaseModel):
    model_config = ConfigDict(frozen=True)

    disable_external_retrieval: bool = False
    local_only: bool = False
    source_type_scope: list[str] = Field(default_factory=list)

    def has_hints(self) -> bool:
        return any((self.disable_external_retrieval, self.local_only, bool(self.source_type_scope)))


class QueryUnderstanding(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType = TaskType.LOOKUP
    complexity_level: ComplexityLevel = ComplexityLevel.L1_DIRECT
    query_type: str = "lookup"
    needs_special: bool = False
    needs_structure: bool = False
    needs_metadata: bool = False
    needs_graph_expansion: bool = False
    structure_constraints: StructureConstraints = Field(default_factory=StructureConstraints)
    metadata_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    special_targets: list[str] = Field(default_factory=list)
    preferred_section_terms: list[str] = Field(default_factory=list)
    source_scope_hints: list[str] = Field(default_factory=list)
    quoted_terms: list[str] = Field(default_factory=list)
    policy_hints: PolicyHints = Field(default_factory=PolicyHints)

    def has_explicit_constraints(self) -> bool:
        return any(
            (
                self.needs_special,
                self.needs_structure,
                self.needs_metadata,
                self.needs_graph_expansion,
                self.structure_constraints.has_constraints(),
                self.metadata_filters.has_constraints(),
                bool(self.special_targets),
                bool(self.preferred_section_terms),
                bool(self.source_scope_hints),
                bool(self.quoted_terms),
                self.policy_hints.has_hints(),
            )
        )


class AnswerCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str
    file_name: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: str
    chunk_type: str
    citation_anchor: str | None = None
    doc_id: str | None = None
    benchmark_doc_id: str | None = None
    source_id: str | None = None
    source_type: str | None = None


class AnswerEvidenceLink(BaseModel):
    model_config = ConfigDict(frozen=True)

    link_id: str
    answer_section_id: str
    answer_excerpt: str
    evidence_chunk_id: str
    citation_id: str | None = None
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)


class AnswerSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: str
    title: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer_text: str
    answer_sections: list[AnswerSection] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    evidence_links: list[AnswerEvidenceLink] = Field(default_factory=list)
    groundedness_flag: bool
    insufficient_evidence_flag: bool


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    benchmark_doc_id: str | None = None
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
    retrieval_channels: list[str] = Field(default_factory=list)
    retrieval_family: str | None = None


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


class ArtifactType(StrEnum):
    DOCUMENT_SUMMARY = "document_summary"
    SECTION_SUMMARY = "section_summary"
    COMPARISON_PAGE = "comparison_page"
    TOPIC_PAGE = "topic_page"
    TIMELINE = "timeline"
    OPEN_QUESTION_PAGE = "open_question_page"


class ArtifactStatus(StrEnum):
    SUGGESTED = "suggested"
    APPROVED = "approved"
    STALE = "stale"
    CONFLICTED = "conflicted"
    ARCHIVED = "archived"


class KnowledgeArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    artifact_type: ArtifactType
    title: str
    supported_chunk_ids: list[str]
    confidence: float | None = None
    status: ArtifactStatus
    last_reviewed_at: datetime
    body_markdown: str
    source_scope: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


def _rebuild_runtime_schema_refs() -> None:
    import rag.schema.runtime as _runtime_schema

    _runtime_schema.RetrievalDiagnostics.model_rebuild(
        _types_namespace={"QueryUnderstanding": QueryUnderstanding}
    )


_rebuild_runtime_schema_refs()

__all__ = [
    "AnswerCitation",
    "AnswerEvidenceLink",
    "AnswerSection",
    "ArtifactStatus",
    "ArtifactType",
    "ComplexityLevel",
    "EvidenceItem",
    "ExecutionPolicy",
    "GroundedAnswer",
    "KnowledgeArtifact",
    "MetadataFilters",
    "PageRangeConstraint",
    "PolicyHints",
    "PreservationSuggestion",
    "QueryRequest",
    "QueryResponse",
    "QueryUnderstanding",
    "ResearchSubQuestion",
    "StructureConstraints",
    "TaskType",
]
