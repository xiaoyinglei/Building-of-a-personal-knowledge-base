from __future__ import annotations

from rag.agent.critic import EvidenceCritic
from rag.agent.executor import AgentExecutor
from rag.agent.planner import AgentPlanner
from rag.agent.schema import AgentTaskRequest
from rag.agent.service import AnalysisAgentService
from rag.agent.understanding import TaskUnderstandingService
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


class _RuleBasedRetrievalService:
    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        del kwargs
        lowered = query.lower()
        evidence: list[EvidenceItem] = []
        if "alpha" in lowered:
            evidence.append(_item("c1", "Alpha uses a layered ingestion and retrieval pipeline."))
        if "beta" in lowered:
            evidence.append(_item("c2", "Beta favors a compact serving architecture with fewer modules."))
        if "tradeoff" in lowered or "recommendation" in lowered or "difference" in lowered:
            evidence.append(_item("c3", "Alpha trades simplicity for stronger evidence controls."))
            evidence.append(_item("c4", "Beta is simpler to operate but exposes fewer diagnostics."))
        sufficient = len(evidence) >= 2
        return RetrievalResult(
            decision=RoutingDecision(
                task_type=TaskType.RESEARCH,
                complexity_level=ComplexityLevel.L4_RESEARCH,
                runtime_mode=RuntimeMode.DEEP,
            ),
            evidence=EvidenceBundle(internal=evidence),
            self_check=SelfCheckResult(
                retrieve_more=not sufficient,
                evidence_sufficient=sufficient,
                claim_supported=sufficient,
            ),
            diagnostics=RetrievalDiagnostics(branch_hits={"vector": len(evidence)}),
        )


class _EmptyRetrievalService:
    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        del query, kwargs
        return RetrievalResult(
            decision=RoutingDecision(
                task_type=TaskType.RESEARCH,
                complexity_level=ComplexityLevel.L4_RESEARCH,
                runtime_mode=RuntimeMode.DEEP,
            ),
            evidence=EvidenceBundle(),
            self_check=SelfCheckResult(
                retrieve_more=True,
                evidence_sufficient=False,
                claim_supported=False,
            ),
            diagnostics=RetrievalDiagnostics(branch_hits={}),
        )


def _service(retrieval_service: object) -> AnalysisAgentService:
    return AnalysisAgentService(
        task_understanding_service=TaskUnderstandingService(enable_llm=False),
        planner=AgentPlanner(enable_llm=False),
        executor=AgentExecutor(retrieval_service=retrieval_service, critic=EvidenceCritic()),
    )


def test_analysis_agent_service_runs_full_pipeline_and_returns_report() -> None:
    service = _service(_RuleBasedRetrievalService())
    request = AgentTaskRequest(
        user_query="Compare Alpha and Beta and recommend which system is easier to operate.",
        retry_budget=1,
    )

    state = service.run_task(request, access_policy=AccessPolicy.default())

    assert state.final_report is not None
    assert state.final_report.execution_summary.subtasks_count >= 3
    assert state.final_report.key_findings
    assert state.traces


def test_analysis_agent_service_surfaces_failures_when_no_subtask_can_be_supported() -> None:
    service = _service(_EmptyRetrievalService())
    request = AgentTaskRequest(
        user_query="Summarize Alpha operational risks.",
        retry_budget=1,
    )

    state = service.run_task(request, access_policy=AccessPolicy.default())

    assert state.failures
    assert state.final_report is not None
    assert state.final_report.unknowns
