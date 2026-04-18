from __future__ import annotations

from rag.agent.critic import EvidenceCritic
from rag.agent.schema import CriticAction, SubTask
from rag.retrieval.analysis import RoutingDecision
from rag.retrieval.evidence import EvidenceBundle, SelfCheckResult
from rag.retrieval.models import RetrievalResult
from rag.schema.query import ComplexityLevel, EvidenceItem, TaskType
from rag.schema.runtime import RetrievalDiagnostics, RuntimeMode


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


def _item(chunk_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id="doc-1",
        citation_anchor=f"sec-{chunk_id}",
        text=text,
        score=0.9,
    )


def test_evidence_critic_accepts_supported_subtask_results() -> None:
    critic = EvidenceCritic()
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
    retrieval = _retrieval_result(
        _item("c1", "Alpha exposes an API gateway interface for clients."),
        _item("c2", "The API contract is documented in the integration section."),
        sufficient=True,
    )

    assessment = critic.assess(
        subtask=subtask,
        retrieval=retrieval,
        attempt_index=1,
        retry_budget_remaining=1,
        allow_web=False,
    )

    assert assessment.sufficient is True
    assert assessment.recommended_action is CriticAction.ACCEPT
    assert assessment.confidence >= 0.6


def test_evidence_critic_requests_query_rewrite_when_evidence_is_missing() -> None:
    critic = EvidenceCritic()
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
    retrieval = _retrieval_result(sufficient=False, retrieve_more=True)

    assessment = critic.assess(
        subtask=subtask,
        retrieval=retrieval,
        attempt_index=1,
        retry_budget_remaining=2,
        allow_web=False,
    )

    assert assessment.sufficient is False
    assert assessment.recommended_action is CriticAction.RETRY_REWRITE_QUERY
    assert assessment.missing_dimensions == ["interface", "api"]


def test_evidence_critic_abstains_when_conflict_remains_after_budget_is_exhausted() -> None:
    critic = EvidenceCritic()
    subtask = SubTask(
        subtask_id="s1",
        objective="Determine whether Alpha supports local deployment",
        instruction="Retrieve deployment evidence",
        expected_evidence=["local deployment support"],
        retrieval_hint="Prefer deployment sections",
        allow_web=False,
        stop_condition="Support stance is clear",
        priority=1,
    )
    retrieval = _retrieval_result(
        _item("c1", "Alpha supports local deployment in regulated environments."),
        _item("c2", "Alpha does not support local deployment for customer installations."),
        sufficient=False,
        retrieve_more=True,
    )

    assessment = critic.assess(
        subtask=subtask,
        retrieval=retrieval,
        attempt_index=2,
        retry_budget_remaining=0,
        allow_web=False,
    )

    assert assessment.conflicts
    assert assessment.recommended_action is CriticAction.ABSTAIN
