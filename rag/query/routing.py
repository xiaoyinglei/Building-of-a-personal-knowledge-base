from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag.query.policies import RoutingThresholds
from rag.schema._types.access import AccessPolicy, RuntimeMode
from rag.schema._types.query import (
    ComplexityLevel,
    ConfidenceBand,
    QueryIntent,
    QueryUnderstanding,
    TaskType,
)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType
    complexity_level: ComplexityLevel
    runtime_mode: RuntimeMode
    source_scope: list[str] = Field(default_factory=list)
    web_search_allowed: bool = False
    graph_expansion_allowed: bool = False
    rerank_required: bool = True


_DEEP_TASK_TYPES = {
    TaskType.COMPARISON,
    TaskType.SYNTHESIS,
    TaskType.TIMELINE,
    TaskType.RESEARCH,
}


class RoutingService:
    def __init__(self, thresholds: RoutingThresholds | None = None) -> None:
        self._thresholds = thresholds or RoutingThresholds()

    def route(
        self,
        query: str,
        *,
        query_understanding: QueryUnderstanding,
        source_scope: Sequence[str] = (),
        access_policy: AccessPolicy | None = None,
    ) -> RoutingDecision:
        del query, access_policy
        task_type = self._classify_task_type(query_understanding, source_scope)
        complexity_level = self._classify_complexity(task_type, source_scope, query_understanding)
        runtime_mode = self._choose_runtime(task_type, complexity_level, query_understanding)
        web_search_allowed = self._web_search_allowed(task_type, source_scope, query_understanding)
        graph_expansion_allowed = self._graph_expansion_allowed(query_understanding, runtime_mode)
        rerank_required = self._rerank_required(query_understanding)
        return RoutingDecision(
            task_type=task_type,
            complexity_level=complexity_level,
            runtime_mode=runtime_mode,
            source_scope=list(source_scope),
            web_search_allowed=web_search_allowed,
            graph_expansion_allowed=graph_expansion_allowed,
            rerank_required=rerank_required,
        )

    @staticmethod
    def _classify_task_type(
        query_understanding: QueryUnderstanding,
        source_scope: Sequence[str],
    ) -> TaskType:
        intent = query_understanding.intent
        if intent is QueryIntent.COMPARISON_REQUEST:
            return TaskType.COMPARISON
        if intent is QueryIntent.SUMMARY_REQUEST:
            return TaskType.RESEARCH if query_understanding.should_decompose_query else TaskType.SYNTHESIS
        if intent is QueryIntent.FLOW_PROCESS_REQUEST:
            return TaskType.RESEARCH if query_understanding.needs_graph_expansion else TaskType.SYNTHESIS
        if query_understanding.should_decompose_query:
            return TaskType.RESEARCH
        if len(source_scope) == 1 or query_understanding.metadata_filters.has_constraints():
            return TaskType.SINGLE_DOC_QA
        return TaskType.LOOKUP

    @staticmethod
    def _classify_complexity(
        task_type: TaskType,
        source_scope: Sequence[str],
        query_understanding: QueryUnderstanding,
    ) -> ComplexityLevel:
        if task_type is TaskType.COMPARISON:
            return ComplexityLevel.L3_COMPARATIVE
        if task_type in {TaskType.TIMELINE, TaskType.RESEARCH, TaskType.SYNTHESIS}:
            return ComplexityLevel.L4_RESEARCH
        if len(source_scope) == 1 or query_understanding.metadata_filters.has_constraints():
            return ComplexityLevel.L2_SCOPED
        return ComplexityLevel.L1_DIRECT

    @staticmethod
    def _choose_runtime(
        task_type: TaskType,
        complexity_level: ComplexityLevel,
        query_understanding: QueryUnderstanding,
    ) -> RuntimeMode:
        if query_understanding.metadata_filters.has_constraints() and not query_understanding.should_decompose_query:
            return RuntimeMode.FAST
        if task_type in _DEEP_TASK_TYPES:
            return RuntimeMode.DEEP
        if complexity_level in {ComplexityLevel.L3_COMPARATIVE, ComplexityLevel.L4_RESEARCH}:
            return RuntimeMode.DEEP
        if query_understanding.confidence_band is ConfidenceBand.LOW and query_understanding.should_rewrite_query:
            return RuntimeMode.DEEP
        return RuntimeMode.FAST

    @staticmethod
    def _web_search_allowed(
        task_type: TaskType,
        source_scope: Sequence[str],
        query_understanding: QueryUnderstanding,
    ) -> bool:
        del query_understanding
        if source_scope:
            return False
        if task_type not in {TaskType.COMPARISON, TaskType.SYNTHESIS, TaskType.TIMELINE, TaskType.RESEARCH}:
            return False
        return True

    @staticmethod
    def _graph_expansion_allowed(
        query_understanding: QueryUnderstanding,
        runtime_mode: RuntimeMode,
    ) -> bool:
        if runtime_mode is not RuntimeMode.DEEP:
            return False
        if not query_understanding.needs_graph_expansion:
            return False
        return query_understanding.confidence_band is not ConfidenceBand.LOW

    @staticmethod
    def _rerank_required(query_understanding: QueryUnderstanding) -> bool:
        if len(query_understanding.routing_hints.primary_channels) >= 2:
            return True
        return query_understanding.confidence_band is not ConfidenceBand.HIGH


__all__ = [
    "RoutingDecision",
    "RoutingService",
]
