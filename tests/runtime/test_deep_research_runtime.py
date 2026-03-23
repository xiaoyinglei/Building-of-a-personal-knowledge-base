from dataclasses import dataclass, field

from pkp.runtime.deep_research_runtime import DeepResearchRuntime
from pkp.runtime.session_runtime import SessionRuntime
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
        task_type=TaskType.RESEARCH,
        complexity_level=ComplexityLevel.L4_RESEARCH,
        latency_budget=60,
        cost_budget=3.0,
        execution_location_preference=ExecutionLocationPreference.CLOUD_FIRST,
        fallback_allowed=True,
    )


def hit(chunk_id: str, doc_id: str, score: float) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_id=f"src-{doc_id}",
        citation_anchor=f"{doc_id}#1",
        text=f"evidence from {doc_id}",
        score=score,
    )


@dataclass
class FakeRoutingService:
    expansions: list[list[str]] = field(default_factory=lambda: [["sub-q-1", "sub-q-2"], ["sub-q-3"]])

    def decompose(self, query: str) -> list[str]:
        return list(self.expansions[0])

    def expand(self, query: str, evidence_matrix: list[dict[str, object]], round_index: int) -> list[str]:
        if round_index >= len(self.expansions):
            return []
        return list(self.expansions[round_index])


@dataclass
class FakeRetrievalService:
    batches: list[list[EvidenceItem]]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def retrieve(self, query: str, policy: ExecutionPolicy, mode: RuntimeMode, round_index: int) -> list[EvidenceItem]:
        self.calls.append((query, round_index))
        return self.batches[round_index - 1]


class FakeEvidenceService:
    def __init__(self, sufficient_after_round: int) -> None:
        self._sufficient_after_round = sufficient_after_round

    def build_evidence_matrix(self, hits: list[EvidenceItem]) -> list[dict[str, object]]:
        return [{"claim": item.text, "sources": [item.doc_id]} for item in hits]

    def evidence_sufficient(self, evidence_matrix: list[dict[str, object]], round_index: int) -> bool:
        return round_index >= self._sufficient_after_round

    def build_deep_response(self, query: str, evidence_matrix: list[dict[str, object]]) -> QueryResponse:
        return QueryResponse(
            conclusion=f"deep answer for {query}",
            evidence=[
                EvidenceItem(
                    chunk_id=f"chunk-{index}",
                    doc_id=row["sources"][0],
                    citation_anchor="matrix",
                    text=row["claim"],
                    score=1.0,
                )
                for index, row in enumerate(evidence_matrix, start=1)
            ],
            differences_or_conflicts=[],
            uncertainty="low",
            preservation_suggestion=PreservationSuggestion(
                suggested=True,
                artifact_type="topic_page",
                title="Research summary",
            ),
            runtime_mode=RuntimeMode.DEEP,
        )

    def build_retrieval_only_response(self, query: str, hits: list[EvidenceItem]) -> QueryResponse:
        return QueryResponse(
            conclusion=f"retrieval only for {query}",
            evidence=hits,
            differences_or_conflicts=[],
            uncertainty="high",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=RuntimeMode.DEEP,
        )


def test_deep_research_runtime_builds_evidence_matrix_and_stops_when_sufficient() -> None:
    runtime = DeepResearchRuntime(
        routing_service=FakeRoutingService(),
        retrieval_service=FakeRetrievalService(batches=[[hit("a", "doc-1", 0.9)], [hit("b", "doc-2", 0.8)]]),
        evidence_service=FakeEvidenceService(sufficient_after_round=2),
        session_runtime=SessionRuntime(),
        max_rounds=4,
    )

    response = runtime.run("compare docs", make_policy(), session_id="session-1")

    session = runtime.session_runtime.get("session-1")
    assert session.sub_questions == ["sub-q-1", "sub-q-2"]
    assert len(session.evidence_matrix) == 1
    assert response.runtime_mode is RuntimeMode.DEEP
    assert response.preservation_suggestion.suggested is True
