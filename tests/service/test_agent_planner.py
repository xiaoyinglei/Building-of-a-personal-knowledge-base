from __future__ import annotations

from rag.agent.planner import AgentPlanner
from rag.agent.schema import AgentTaskRequest, TaskUnderstanding
from rag.schema.query import ComplexityLevel, TaskType


def test_agent_planner_breaks_complex_comparison_into_information_gaps() -> None:
    planner = AgentPlanner(enable_llm=False)
    request = AgentTaskRequest(
        user_query="Compare Alpha and Beta and recommend which system fits regulated deployment.",
        max_subtasks=5,
    )
    understanding = TaskUnderstanding(
        task_type=TaskType.COMPARISON,
        complexity_level=ComplexityLevel.L3_COMPARATIVE,
        deliverable_type="decision_report",
        decomposition_required=True,
        needs_external_evidence=False,
        needs_comparison=True,
        needs_timeline=False,
        success_criteria=["Cover both systems", "Compare tradeoffs", "Give grounded recommendation"],
    )

    subtasks = planner.plan(request=request, understanding=understanding)

    assert 3 <= len(subtasks) <= 5
    assert subtasks[0].priority <= subtasks[-1].priority
    assert all(subtask.objective for subtask in subtasks)
    assert all(subtask.expected_evidence for subtask in subtasks)


def test_agent_planner_respects_subtask_cap_in_fallback_mode() -> None:
    planner = AgentPlanner(enable_llm=False)
    request = AgentTaskRequest(
        user_query="Research Alpha architecture, timeline, risks, and open questions.",
        max_subtasks=3,
    )
    understanding = TaskUnderstanding(
        task_type=TaskType.RESEARCH,
        complexity_level=ComplexityLevel.L4_RESEARCH,
        deliverable_type="analysis_report",
        decomposition_required=True,
        needs_external_evidence=False,
        needs_comparison=False,
        needs_timeline=True,
        success_criteria=["Cover chronology", "Explain architecture", "List unknowns"],
    )

    subtasks = planner.plan(request=request, understanding=understanding)

    assert len(subtasks) == 3
    assert any("timeline" in subtask.objective.lower() or "chronology" in subtask.objective.lower() for subtask in subtasks)
