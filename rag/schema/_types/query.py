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

#研究型问题拆出来的子问题
class ResearchSubQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str
    purpose: str

#页码范围约束
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
        return any(
            (
                self.disable_external_retrieval,
                self.local_only,
                bool(self.source_type_scope),
            )
        )


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
