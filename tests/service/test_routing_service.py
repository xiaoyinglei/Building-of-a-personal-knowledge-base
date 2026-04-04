from __future__ import annotations

from rag.query.routing import RoutingService
from rag.query.understanding import QueryUnderstandingService
from rag.schema._types.access import RuntimeMode
from rag.schema._types.query import ComplexityLevel, TaskType


def test_routing_service_routes_comparison_queries_to_deep_research() -> None:
    understanding = QueryUnderstandingService().analyze("比较 Alpha 和 Beta 的检索链路差异。")
    service = RoutingService()

    decision = service.route(
        "比较 Alpha 和 Beta 的检索链路差异。",
        query_understanding=understanding,
        source_scope=["doc-alpha", "doc-beta"],
    )

    assert decision.task_type is TaskType.COMPARISON
    assert decision.complexity_level is ComplexityLevel.L3_COMPARATIVE
    assert decision.runtime_mode is RuntimeMode.DEEP


def test_routing_service_keeps_high_confidence_scoped_queries_on_fast_path() -> None:
    understanding = QueryUnderstandingService().analyze("第2页讲了什么风险？")
    service = RoutingService()

    decision = service.route(
        "第2页讲了什么风险？",
        query_understanding=understanding,
        source_scope=["doc-alpha"],
    )

    assert decision.task_type is TaskType.SINGLE_DOC_QA
    assert decision.complexity_level is ComplexityLevel.L2_SCOPED
    assert decision.runtime_mode is RuntimeMode.FAST
    assert decision.web_search_allowed is False
    assert decision.graph_expansion_allowed is False


def test_routing_service_uses_confidence_to_avoid_overcommitting_to_deep_path() -> None:
    understanding = QueryUnderstandingService().analyze("这个呢？")
    service = RoutingService()

    decision = service.route(
        "这个呢？",
        query_understanding=understanding,
    )

    assert understanding.should_rewrite_query is True
    assert decision.runtime_mode is RuntimeMode.DEEP
    assert decision.web_search_allowed is False
    assert decision.rerank_required is True
