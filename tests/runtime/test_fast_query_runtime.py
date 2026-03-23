from dataclasses import dataclass

from pkp.runtime.fast_query_runtime import FastQueryRuntime
from pkp.types import (
    AccessPolicy,
    EvidenceItem,
    ExecutionLocationPreference,
    ExecutionPolicy,
    PreservationSuggestion,
    QueryResponse,
    RuntimeMode,
    TaskType,
)
from pkp.types.query import ComplexityLevel


def make_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        effective_access_policy=AccessPolicy.default(),
        task_type=TaskType.LOOKUP,
        complexity_level=ComplexityLevel.L1_DIRECT,
        latency_budget=10,
        cost_budget=1.0,
        execution_location_preference=ExecutionLocationPreference.CLOUD_FIRST,
        fallback_allowed=True,
    )


@dataclass
class FakeRetrievalService:
    hits: list[EvidenceItem]

    def retrieve(self, query: str, policy: ExecutionPolicy, mode: RuntimeMode, round_index: int) -> list[EvidenceItem]:
        assert mode is RuntimeMode.FAST
        assert round_index == 1
        return self.hits

    def rerank(self, query: str, hits: list[EvidenceItem], policy: ExecutionPolicy) -> list[EvidenceItem]:
        return hits


class FakeEvidenceService:
    def __init__(self, sufficient: bool, claim_aligned: bool, conflicts: list[str] | None = None) -> None:
        self._sufficient = sufficient
        self._claim_aligned = claim_aligned
        self._conflicts = conflicts or []

    def fast_path_sufficient(self, hits: list[EvidenceItem], policy: ExecutionPolicy) -> bool:
        return self._sufficient

    def detect_conflicts(self, hits: list[EvidenceItem]) -> list[str]:
        return self._conflicts

    def build_fast_response(self, query: str, hits: list[EvidenceItem]) -> QueryResponse:
        return QueryResponse(
            conclusion=f"answer for {query}",
            evidence=hits,
            differences_or_conflicts=self._conflicts,
            uncertainty="low",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=RuntimeMode.FAST,
        )

    def claim_citation_aligned(self, response: QueryResponse) -> bool:
        return self._claim_aligned


class FakeDeepRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, query: str, policy: ExecutionPolicy) -> QueryResponse:
        self.calls.append(query)
        return QueryResponse(
            conclusion="deep answer",
            evidence=[],
            differences_or_conflicts=[],
            uncertainty="medium",
            preservation_suggestion=PreservationSuggestion(suggested=True, artifact_type="topic_page"),
            runtime_mode=RuntimeMode.DEEP,
        )


def make_hit() -> EvidenceItem:
    return EvidenceItem(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_id="src-1",
        citation_anchor="Section 1",
        text="evidence",
        score=0.95,
    )


def test_fast_query_runtime_returns_evidence_backed_answer_when_checks_pass() -> None:
    runtime = FastQueryRuntime(
        retrieval_service=FakeRetrievalService(hits=[make_hit()]),
        evidence_service=FakeEvidenceService(sufficient=True, claim_aligned=True),
        deep_runtime=FakeDeepRuntime(),
    )

    response = runtime.run("what is this", make_policy())

    assert response.runtime_mode is RuntimeMode.FAST
    assert response.conclusion == "answer for what is this"


def test_fast_query_runtime_escalates_when_evidence_is_insufficient() -> None:
    deep_runtime = FakeDeepRuntime()
    runtime = FastQueryRuntime(
        retrieval_service=FakeRetrievalService(hits=[make_hit()]),
        evidence_service=FakeEvidenceService(sufficient=False, claim_aligned=True),
        deep_runtime=deep_runtime,
    )

    response = runtime.run("complex question", make_policy())

    assert deep_runtime.calls == ["complex question"]
    assert response.runtime_mode is RuntimeMode.DEEP


def test_fast_query_runtime_escalates_on_conflict_or_failed_claim_alignment() -> None:
    deep_runtime = FakeDeepRuntime()
    runtime = FastQueryRuntime(
        retrieval_service=FakeRetrievalService(hits=[make_hit()]),
        evidence_service=FakeEvidenceService(
            sufficient=True,
            claim_aligned=False,
            conflicts=["doc-1 disagrees with doc-2"],
        ),
        deep_runtime=deep_runtime,
    )

    response = runtime.run("why conflict", make_policy())

    assert deep_runtime.calls == ["why conflict"]
    assert response.runtime_mode is RuntimeMode.DEEP
