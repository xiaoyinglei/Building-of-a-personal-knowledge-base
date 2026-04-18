"""Top-level service that orchestrates the full agent workflow."""

from __future__ import annotations

from rag.agent.critic import EvidenceCritic
from rag.agent.executor import AgentExecutor
from rag.agent.planner import AgentPlanner
from rag.agent.report import AgentReportBuilder
from rag.agent.schema import AgentRunStatus, AgentTaskRequest, CriticAction, SubTaskStatus
from rag.agent.state import AgentFailureEvent, AgentFailureKind, AgentRunState
from rag.agent.synthesizer import AgentSynthesizer
from rag.agent.understanding import TaskUnderstandingService
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class AnalysisAgentService:
    """Formal entrypoint for structured multi-step analysis tasks."""

    def __init__(
        self,
        *,
        task_understanding_service: TaskUnderstandingService | None = None,
        planner: AgentPlanner | None = None,
        executor: AgentExecutor | None = None,
        synthesizer: AgentSynthesizer | None = None,
        report_builder: AgentReportBuilder | None = None,
    ) -> None:
        self._task_understanding_service = task_understanding_service or TaskUnderstandingService(enable_llm=False)
        self._planner = planner or AgentPlanner(enable_llm=False)
        self._executor = executor
        self._report_builder = report_builder or AgentReportBuilder()
        self._synthesizer = synthesizer or AgentSynthesizer(report_builder=self._report_builder)
        self.last_state: AgentRunState | None = None

    def run_task(
        self,
        request: AgentTaskRequest,
        *,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST,
    ) -> AgentRunState:
        if self._executor is None:
            raise RuntimeError("AnalysisAgentService requires an executor before run_task can be called")
        state = AgentRunState(request=request, status=AgentRunStatus.RUNNING)
        understanding = self._task_understanding_service.analyze(request)
        state.task_understanding = understanding
        subtasks = self._planner.plan(request=request, understanding=understanding)
        if not subtasks:
            state.record_failure(
                AgentFailureEvent(
                    kind=AgentFailureKind.PLANNER_FAILURE,
                    message="Planner produced no subtasks.",
                    stage="planner",
                )
            )
            state.status = AgentRunStatus.FAILED
            state.final_report = self._synthesizer.synthesize(
                state=state,
                access_policy=access_policy,
                execution_location_preference=execution_location_preference,
            )
            self.last_state = state
            return state

        state.subtasks = list(subtasks)
        for subtask in subtasks:
            result = self._executor.execute_subtask(
                request=request,
                subtask=subtask,
                access_policy=access_policy,
                execution_location_preference=execution_location_preference,
            )
            state.record_result(result)
            if result.status is SubTaskStatus.RETRY_EXHAUSTED:
                state.record_failure(
                    AgentFailureEvent(
                        kind=AgentFailureKind.RETRY_EXHAUSTED,
                        message=f"Retry budget exhausted for {subtask.subtask_id}.",
                        stage="executor",
                    )
                )
            elif result.status is SubTaskStatus.ABSTAINED:
                failure_kind = (
                    AgentFailureKind.EVIDENCE_CONFLICT
                    if result.evidence_assessment.conflicts
                    else AgentFailureKind.RETRIEVAL_MISS
                )
                state.record_failure(
                    AgentFailureEvent(
                        kind=failure_kind,
                        message=f"Subtask {subtask.subtask_id} could not be completed with current evidence.",
                        stage="critic",
                    )
                )
            state.web_used = state.web_used or any(
                trace.action_taken is CriticAction.ENABLE_WEB for trace in result.traces
            )

        state.final_report = self._synthesizer.synthesize(
            state=state,
            access_policy=access_policy,
            execution_location_preference=execution_location_preference,
        )
        if not state.subtask_results:
            state.status = AgentRunStatus.FAILED
        elif state.failures:
            state.status = AgentRunStatus.PARTIAL
        else:
            state.status = AgentRunStatus.COMPLETED
        self.last_state = state
        return state


__all__ = ["AnalysisAgentService"]
