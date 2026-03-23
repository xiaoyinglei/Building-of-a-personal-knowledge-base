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
