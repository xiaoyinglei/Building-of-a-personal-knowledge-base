from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from pkp.config.policies import RoutingThresholds
from pkp.types.access import AccessPolicy, RuntimeMode
from pkp.types.query import ComplexityLevel, TaskType


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType
    complexity_level: ComplexityLevel
    runtime_mode: RuntimeMode
    source_scope: list[str] = Field(default_factory=list)
    web_search_allowed: bool = False
    graph_expansion_allowed: bool = False
    rerank_required: bool = True


_COMPARISON_TOKENS = ("compare", "versus", " vs ", "difference", "contrast")
_TIMELINE_TOKENS = ("timeline", "trend", "over time", "chronology")
_RESEARCH_TOKENS = ("research", "synthesize", "summary", "summarize", "why", "how ")
_SCOPED_TOKENS = ("this document", "this source", "in this doc")
_DEEP_TASK_TYPES = {
    TaskType.COMPARISON,
    TaskType.SYNTHESIS,
    TaskType.TIMELINE,
    TaskType.RESEARCH,
}


class RoutingService:
    def __init__(self, thresholds: RoutingThresholds | None = None) -> None:
        self._thresholds = thresholds or RoutingThresholds()

    @staticmethod
    def _normalized_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def _classify_task_type(self, query: str, source_scope: Sequence[str]) -> TaskType:
        normalized = self._normalized_query(query)
        if any(token in normalized for token in _COMPARISON_TOKENS):
            return TaskType.COMPARISON
        if any(token in normalized for token in _TIMELINE_TOKENS):
            return TaskType.TIMELINE
        if any(token in normalized for token in _RESEARCH_TOKENS) or len(source_scope) > 1:
            return TaskType.RESEARCH if len(source_scope) > 1 or "research" in normalized else TaskType.SYNTHESIS
        if len(source_scope) == 1 or any(token in normalized for token in _SCOPED_TOKENS):
            return TaskType.SINGLE_DOC_QA
        return TaskType.LOOKUP

    def _classify_complexity(
        self,
        task_type: TaskType,
        source_scope: Sequence[str],
    ) -> ComplexityLevel:
        if task_type is TaskType.COMPARISON:
            return ComplexityLevel.L3_COMPARATIVE
        if task_type in {TaskType.TIMELINE, TaskType.RESEARCH, TaskType.SYNTHESIS}:
            return ComplexityLevel.L4_RESEARCH
        if len(source_scope) == 1:
            return ComplexityLevel.L2_SCOPED
        return ComplexityLevel.L1_DIRECT

    @staticmethod
    def _choose_runtime(task_type: TaskType, complexity_level: ComplexityLevel) -> RuntimeMode:
        if task_type in _DEEP_TASK_TYPES:
            return RuntimeMode.DEEP
        if complexity_level in {
            ComplexityLevel.L3_COMPARATIVE,
            ComplexityLevel.L4_RESEARCH,
        }:
            return RuntimeMode.DEEP
        return RuntimeMode.FAST

    def route(
        self,
        query: str,
        *,
        source_scope: Sequence[str] = (),
        access_policy: AccessPolicy | None = None,
    ) -> RoutingDecision:
        del access_policy
        task_type = self._classify_task_type(query, source_scope)
        complexity_level = self._classify_complexity(task_type, source_scope)
        runtime_mode = self._choose_runtime(task_type, complexity_level)
        web_search_allowed = task_type in {
            TaskType.COMPARISON,
            TaskType.SYNTHESIS,
            TaskType.TIMELINE,
            TaskType.RESEARCH,
        }
        graph_expansion_allowed = runtime_mode is RuntimeMode.DEEP
        return RoutingDecision(
            task_type=task_type,
            complexity_level=complexity_level,
            runtime_mode=runtime_mode,
            source_scope=list(source_scope),
            web_search_allowed=web_search_allowed,
            graph_expansion_allowed=graph_expansion_allowed,
        )
