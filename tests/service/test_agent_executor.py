from __future__ import annotations

from collections import deque

from rag.agent.critic import EvidenceCritic
from rag.agent.executor import AgentExecutor
from rag.agent.schema import AgentTaskRequest, SubTask, SubTaskStatus
from rag.retrieval.analysis import RoutingDecision
from rag.retrieval.evidence import EvidenceBundle, SelfCheckResult
from rag.retrieval.models import RetrievalResult
from rag.schema.query import ComplexityLevel, EvidenceItem, TaskType
from rag.schema.runtime import AccessPolicy, RetrievalDiagnostics, RuntimeMode


def _item(chunk_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id="doc-1",
        citation_anchor=f"sec-{chunk_id}",
        text=text,
        score=0.9,
    )


def _retrieval_result(*items: EvidenceItem, sufficient: bool, retrieve_more: bool = False) -> RetrievalResult:
    return RetrievalResult(
        decision=RoutingDecision(
            task_type=TaskType.RESEARCH,
            complexity_level=ComplexityLevel.L4_RESEARCH,
            runtime_mode=RuntimeMode.DEEP,
        ),
        evidence=EvidenceBundle(internal=list(items)),
        self_check=SelfCheckResult(
            retrieve_more=retrieve_more,
            evidence_sufficient=sufficient,
            claim_supported=sufficient,
        ),
        diagnostics=RetrievalDiagnostics(branch_hits={"vector": len(items)}),
    )


class _StubRetrievalService:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = deque(results)
        self.queries: list[str] = []

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        del kwargs
        self.queries.append(query)
        return self._results.popleft()


def test_agent_executor_retries_with_rewritten_query_and_collects_trace() -> None:
    retrieval_service = _StubRetrievalService(
        [
            _retrieval_result(sufficient=False, retrieve_more=True),
            _retrieval_result(
                _item("c1", "Alpha exposes an API gateway interface for client integrations."),
                _item("c2", "The interface contract is documented in the API section."),
                sufficient=True,
            ),
        ]
    )
    executor = AgentExecutor(retrieval_service=retrieval_service, critic=EvidenceCritic())
    request = AgentTaskRequest(user_query="Find Alpha interfaces.", retry_budget=1)
    subtask = SubTask(
        subtask_id="s1",
        objective="Find Alpha interfaces",
        instruction="Retrieve Alpha interface evidence",
        expected_evidence=["interface", "api"],
        retrieval_hint="Prefer API sections",
        allow_web=False,
        stop_condition="Two interface mentions found",
        priority=1,
    )

    result = executor.execute_subtask(
        request=request,
        subtask=subtask,
        access_policy=AccessPolicy.default(),
    )

    assert result.status is SubTaskStatus.COMPLETED
    assert len(result.traces) == 2
    assert retrieval_service.queries[0] != retrieval_service.queries[1]
    assert "api" in retrieval_service.queries[1].lower()


def test_agent_executor_marks_retry_exhaustion_when_evidence_never_recovers() -> None:
    retrieval_service = _StubRetrievalService(
        [
            _retrieval_result(sufficient=False, retrieve_more=True),
            _retrieval_result(sufficient=False, retrieve_more=True),
        ]
    )
    executor = AgentExecutor(retrieval_service=retrieval_service, critic=EvidenceCritic())
    request = AgentTaskRequest(user_query="Find Alpha interfaces.", retry_budget=1)
    subtask = SubTask(
        subtask_id="s1",
        objective="Find Alpha interfaces",
        instruction="Retrieve Alpha interface evidence",
        expected_evidence=["interface", "api"],
        retrieval_hint="Prefer API sections",
        allow_web=False,
        stop_condition="Two interface mentions found",
        priority=1,
    )

    result = executor.execute_subtask(
        request=request,
        subtask=subtask,
        access_policy=AccessPolicy.default(),
    )

    assert result.status is SubTaskStatus.RETRY_EXHAUSTED
    assert len(result.traces) == 2
    assert result.unresolved_questions
