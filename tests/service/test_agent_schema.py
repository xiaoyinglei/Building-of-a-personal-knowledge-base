from __future__ import annotations

from rag.agent.schema import (
    AgentFinalReport,
    AgentTaskRequest,
    CriticAction,
    EvidenceAssessment,
    ExecutionStepTrace,
    ExecutionSummary,
    RetrievalIntent,
    SubTask,
    SubTaskResult,
    SubTaskStatus,
    TaskUnderstanding,
)
from rag.agent.state import AgentFailureEvent, AgentFailureKind, AgentRunState
from rag.schema.query import ComplexityLevel, MetadataFilters, PolicyHints, TaskType


def test_agent_task_request_defaults_are_job_ready() -> None:
    request = AgentTaskRequest(user_query="Compare Alpha and Beta architectures.")

    assert request.task_goal == "Produce an evidence-grounded analysis report."
    assert request.source_scope == []
    assert request.allow_web is False
    assert request.expected_output == "structured_analysis_report"
    assert request.response_style == "formal"
    assert request.max_subtasks == 5
    assert request.retry_budget == 2


def test_task_understanding_and_retrieval_intent_use_existing_query_contracts() -> None:
    understanding = TaskUnderstanding(
        task_type=TaskType.COMPARISON,
        complexity_level=ComplexityLevel.L3_COMPARATIVE,
        deliverable_type="comparison_report",
        decomposition_required=True,
        needs_external_evidence=False,
        needs_comparison=True,
        needs_timeline=False,
        success_criteria=["Cover both systems", "Cite evidence"],
    )
    intent = RetrievalIntent(
        needs_special=False,
        needs_structure=True,
        needs_metadata=True,
        needs_graph_expansion=False,
        preferred_sections=["Architecture", "Tradeoffs"],
        metadata_filters=MetadataFilters(page_numbers=[2]),
        policy_hints=PolicyHints(local_only=True),
    )

    assert understanding.task_type is TaskType.COMPARISON
    assert understanding.complexity_level is ComplexityLevel.L3_COMPARATIVE
    assert intent.metadata_filters.page_numbers == [2]
    assert intent.policy_hints.local_only is True


def test_subtask_trace_result_and_report_are_structured() -> None:
    subtask = SubTask(
        subtask_id="s1",
        objective="Identify Alpha architecture claims.",
        instruction="Retrieve architecture evidence for Alpha.",
        expected_evidence=["Architecture layers", "Interfaces"],
        retrieval_hint="Prefer architecture sections.",
        allow_web=False,
        stop_condition="Two independent supporting chunks found.",
        priority=1,
    )
    trace = ExecutionStepTrace(
        subtask_id="s1",
        attempt_index=1,
        retrieval_query="Alpha architecture layers interfaces",
        selected_mode="naive",
        branch_hits={"vector": 4},
        evidence_count=3,
        evidence_sufficient=False,
        action_taken=CriticAction.RETRY_REWRITE_QUERY,
        notes=["Missing interface evidence"],
    )
    assessment = EvidenceAssessment(
        sufficient=False,
        confidence=0.42,
        missing_dimensions=["Interfaces"],
        conflicts=[],
        recommended_action=CriticAction.RETRY_REWRITE_QUERY,
    )
    result = SubTaskResult(
        subtask=subtask,
        status=SubTaskStatus.RETRY_EXHAUSTED,
        findings=["Alpha appears layered."],
        evidence_summary=["Two chunks describe ingestion and storage layers."],
        evidence_assessment=assessment,
        traces=[trace],
        unresolved_questions=["How does Alpha expose external APIs?"],
    )
    report = AgentFinalReport(
        executive_summary="Evidence supports a layered Alpha architecture, but interface coverage is incomplete.",
        key_findings=["Alpha uses layered ingestion and retrieval components."],
        evidence_map=[],
        risks=["Interface boundary remains weakly supported."],
        unknowns=["External API contract"],
        recommendations=["Expand retrieval to interface-specific sections."],
        citations=[],
        execution_summary=ExecutionSummary(
            subtasks_count=1,
            completed_subtasks=0,
            retries_count=1,
            web_used=False,
            unresolved_items_count=1,
        ),
    )

    assert result.evidence_assessment.recommended_action is CriticAction.RETRY_REWRITE_QUERY
    assert result.traces[0].branch_hits["vector"] == 4
    assert report.execution_summary.retries_count == 1


def test_agent_run_state_tracks_failures_and_trace_summary() -> None:
    request = AgentTaskRequest(user_query="Summarize Alpha risks.")
    state = AgentRunState(request=request)

    state.record_failure(
        AgentFailureEvent(
            kind=AgentFailureKind.PLANNER_FAILURE,
            message="planner produced no subtasks",
            stage="planner",
        )
    )

    assert state.failures[0].kind is AgentFailureKind.PLANNER_FAILURE
    assert state.has_failures is True
