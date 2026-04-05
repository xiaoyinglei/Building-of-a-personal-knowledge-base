from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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


class QueryIntent(StrEnum):
    FACTUAL_LOOKUP = "factual_lookup"
    SEMANTIC_LOOKUP = "semantic_lookup"
    LOCALIZED_LOOKUP = "localized_lookup"
    STRUCTURE_LOOKUP = "structure_lookup"
    SECTION_LOOKUP = "section_lookup"
    SUMMARY_REQUEST = "summary_request"
    FLOW_PROCESS_REQUEST = "flow_process_request"
    COMPARISON_REQUEST = "comparison_request"
    SPECIAL_CONTENT_LOOKUP = "special_content_lookup"
    METADATA_CONSTRAINED_LOOKUP = "metadata_constrained_lookup"


class ConfidenceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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


class RoutingHints(BaseModel):
    model_config = ConfigDict(frozen=True)

    dense_priority: float = 0.0
    sparse_priority: float = 0.0
    structure_priority: float = 0.0
    metadata_priority: float = 0.0
    special_priority: float = 0.0
    graph_priority: float = 0.0
    primary_channels: list[str] = Field(default_factory=list)
    rewrite_focus_terms: list[str] = Field(default_factory=list)
    decomposition_axes: list[str] = Field(default_factory=list)


class QueryUnderstanding(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent: QueryIntent
    query_type: str
    confidence: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.LOW
    needs_dense: bool = False
    needs_sparse: bool = False
    needs_special: bool = False
    needs_structure: bool = False
    needs_metadata: bool = False
    needs_graph_expansion: bool = False
    should_rewrite_query: bool = False
    should_decompose_query: bool = False
    structure_constraints: StructureConstraints = Field(default_factory=StructureConstraints)
    metadata_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    special_targets: list[str] = Field(default_factory=list)
    preferred_section_terms: list[str] = Field(default_factory=list)
    routing_hints: RoutingHints = Field(default_factory=RoutingHints)
