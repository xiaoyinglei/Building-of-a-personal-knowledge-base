from pkp.service.routing_service import RoutingService
from pkp.types.access import RuntimeMode
from pkp.types.query import ComplexityLevel, TaskType


def test_routing_service_routes_compare_queries_to_deep_research() -> None:
    service = RoutingService()

    decision = service.route(
        "Compare the tradeoffs between Alpha and Beta",
        source_scope=["doc-alpha", "doc-beta"],
    )

    assert decision.task_type is TaskType.COMPARISON
    assert decision.complexity_level is ComplexityLevel.L3_COMPARATIVE
    assert decision.runtime_mode is RuntimeMode.DEEP


def test_routing_service_keeps_scoped_lookup_queries_on_fast_path() -> None:
    service = RoutingService()

    decision = service.route(
        "What does this document say about retention?",
        source_scope=["doc-alpha"],
    )

    assert decision.task_type is TaskType.SINGLE_DOC_QA
    assert decision.complexity_level is ComplexityLevel.L2_SCOPED
    assert decision.runtime_mode is RuntimeMode.FAST
