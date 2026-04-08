from __future__ import annotations

import json

from rag.assembly import ChatCapabilityBinding
from rag.retrieval.analysis import QueryUnderstandingService, RoutingService
from rag.schema.query import ComplexityLevel, TaskType
from rag.schema.runtime import RuntimeMode


class FakeRoutingBackend:
    chat_model_name = "fake-routing"

    def chat(self, prompt: str) -> str:
        query = prompt.split("Query: ", 1)[1].rsplit("\nJSON only.", 1)[0]
        payload = {
            "比较 Alpha 和 Beta 的检索链路差异。": {
                "task_type": "comparison",
                "complexity_level": "L3_comparative",
                "query_type": "comparison",
            },
            "第2页讲了什么风险？": {
                "task_type": "single_doc_qa",
                "complexity_level": "L2_scoped",
                "query_type": "scoped_lookup",
                "needs_metadata": True,
                "metadata_filters": {"page_numbers": [2]},
            },
            "这个系统的处理流程是怎样的？": {
                "task_type": "research",
                "complexity_level": "L4_research",
                "query_type": "process",
                "needs_graph_expansion": True,
            },
        }[query]
        return json.dumps(payload, ensure_ascii=False)


def _service() -> QueryUnderstandingService:
    binding = ChatCapabilityBinding(backend=FakeRoutingBackend(), location="local")
    return QueryUnderstandingService(chat_bindings=(binding,))


def test_routing_service_routes_comparison_queries_to_deep_research() -> None:
    understanding = _service().analyze("比较 Alpha 和 Beta 的检索链路差异。")
    decision = RoutingService().route(
        "比较 Alpha 和 Beta 的检索链路差异。",
        query_understanding=understanding,
        source_scope=["doc-alpha", "doc-beta"],
    )

    assert decision.task_type is TaskType.COMPARISON
    assert decision.complexity_level is ComplexityLevel.L3_COMPARATIVE
    assert decision.runtime_mode is RuntimeMode.DEEP


def test_routing_service_keeps_scoped_queries_on_fast_path() -> None:
    understanding = _service().analyze("第2页讲了什么风险？")
    decision = RoutingService().route(
        "第2页讲了什么风险？",
        query_understanding=understanding,
        source_scope=["doc-alpha"],
    )

    assert decision.task_type is TaskType.SINGLE_DOC_QA
    assert decision.complexity_level is ComplexityLevel.L2_SCOPED
    assert decision.runtime_mode is RuntimeMode.FAST
    assert decision.web_search_allowed is False
    assert decision.graph_expansion_allowed is False


def test_routing_service_uses_explicit_process_signal_for_deep_path() -> None:
    understanding = _service().analyze("这个系统的处理流程是怎样的？")
    decision = RoutingService().route(
        "这个系统的处理流程是怎样的？",
        query_understanding=understanding,
    )

    assert decision.task_type is TaskType.RESEARCH
    assert decision.complexity_level is ComplexityLevel.L4_RESEARCH
    assert decision.runtime_mode is RuntimeMode.DEEP
    assert decision.web_search_allowed is True
    assert decision.graph_expansion_allowed is True
