from __future__ import annotations

import json

from rag.agent.schema import AgentTaskRequest
from rag.agent.understanding import TaskUnderstandingService
from rag.assembly import ChatCapabilityBinding
from rag.schema.query import TaskType


class _FakeTaskUnderstandingBackend:
    chat_model_name = "fake-task-understanding"

    def chat(self, prompt: str) -> str:
        query = prompt.split("User query: ", 1)[1].splitlines()[0]
        payload = {
            "Compare Alpha and Beta, then recommend which system is better for enterprise deployment.": {
                "task_type": "comparison",
                "complexity_level": "L3_comparative",
                "deliverable_type": "decision_report",
                "decomposition_required": True,
                "needs_external_evidence": False,
                "needs_comparison": True,
                "needs_timeline": False,
                "success_criteria": [
                    "Cover both systems",
                    "Compare tradeoffs",
                    "Produce a grounded recommendation",
                ],
            }
        }.get(
            query,
            {
                "task_type": "research",
                "complexity_level": "L4_research",
                "deliverable_type": "analysis_report",
                "decomposition_required": True,
                "needs_external_evidence": False,
                "needs_comparison": False,
                "needs_timeline": False,
                "success_criteria": ["Collect evidence", "State unknowns"],
            },
        )
        return json.dumps(payload, ensure_ascii=False)


def test_task_understanding_service_extracts_task_level_requirements() -> None:
    binding = ChatCapabilityBinding(backend=_FakeTaskUnderstandingBackend(), location="local")
    service = TaskUnderstandingService(chat_bindings=(binding,))
    request = AgentTaskRequest(
        user_query="Compare Alpha and Beta, then recommend which system is better for enterprise deployment."
    )

    understanding = service.analyze(request)

    assert understanding.task_type is TaskType.COMPARISON
    assert understanding.deliverable_type == "decision_report"
    assert understanding.needs_comparison is True
    assert "Produce a grounded recommendation" in understanding.success_criteria


def test_task_understanding_service_has_conservative_fallback_when_llm_disabled() -> None:
    service = TaskUnderstandingService(enable_llm=False)
    request = AgentTaskRequest(
        user_query="Summarize Alpha system risks and open questions.",
        expected_output="structured_analysis_report",
        allow_web=True,
    )

    understanding = service.analyze(request)

    assert understanding.deliverable_type == "structured_analysis_report"
    assert understanding.task_type in {TaskType.SYNTHESIS, TaskType.RESEARCH}
    assert understanding.decomposition_required is True
    assert len(understanding.success_criteria) >= 2


def test_task_understanding_service_records_query_understanding_diagnostics() -> None:
    service = TaskUnderstandingService(enable_llm=False)
    request = AgentTaskRequest(user_query="Build a timeline of Alpha incidents.")

    service.analyze(request)
    diagnostics = service.diagnostics_payload()

    assert "query_understanding" in diagnostics
    assert diagnostics["fallback_used"] is True
