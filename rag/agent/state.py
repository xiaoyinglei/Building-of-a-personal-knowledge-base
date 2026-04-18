"""Mutable run-state objects for agent execution and failure tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.schema import (
    AgentFinalReport,
    AgentRunStatus,
    AgentTaskRequest,
    ExecutionStepTrace,
    SubTask,
    SubTaskResult,
    TaskUnderstanding,
)


class AgentFailureKind(StrEnum):
    PLANNER_FAILURE = "planner_failure"
    RETRIEVAL_MISS = "retrieval_miss"
    EVIDENCE_CONFLICT = "evidence_conflict"
    RETRY_EXHAUSTED = "retry_exhausted"
    EXTERNAL_DISABLED = "external_disabled"
    SOURCE_CONSTRAINED = "source_constrained"
    SYNTHESIS_INFORMATION_GAP = "synthesis_information_gap"


class AgentFailureEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: AgentFailureKind
    message: str
    stage: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentRunState(BaseModel):
    request: AgentTaskRequest
    status: AgentRunStatus = AgentRunStatus.PENDING
    task_understanding: TaskUnderstanding | None = None
    subtasks: list[SubTask] = Field(default_factory=list)
    subtask_results: list[SubTaskResult] = Field(default_factory=list)
    traces: list[ExecutionStepTrace] = Field(default_factory=list)
    failures: list[AgentFailureEvent] = Field(default_factory=list)
    web_used: bool = False
    retries_used: int = 0
    final_report: AgentFinalReport | None = None

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)

    def record_trace(self, trace: ExecutionStepTrace) -> None:
        self.traces.append(trace)

    def record_result(self, result: SubTaskResult) -> None:
        self.subtask_results.append(result)
        self.traces.extend(result.traces)

    def record_failure(self, failure: AgentFailureEvent) -> None:
        self.failures.append(failure)


__all__ = [
    "AgentFailureEvent",
    "AgentFailureKind",
    "AgentRunState",
]
