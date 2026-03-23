from dataclasses import dataclass

from pkp.runtime.deep_research_runtime import DeepResearchRuntime
from pkp.runtime.session_runtime import SessionRuntime
from pkp.types import (
    AccessPolicy,
    EvidenceItem,
    ExecutionLocationPreference,
    ExecutionPolicy,
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


@dataclass
class FakeRoutingService:
    def decompose(self, query: str) -> list[str]:
        return [query]

    def expand(self, query: str, evidence_matrix: list[dict[str, object]], round_index: int) -> list[str]:
        return []


@dataclass
class FakeRetrievalService:
    def retrieve(self, query: str, policy: ExecutionPolicy, mode: RuntimeMode, round_index: int) -> list[EvidenceItem]:
        return [
            EvidenceItem(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_id="src-1",
                citation_anchor="Section",
                text="retrieved evidence",
                score=0.8,
            )
        ]


class FakeEvidenceService:
    def build_evidence_matrix(self, hits: list[EvidenceItem]) -> list[dict[str, object]]:
        return [{"claim": item.text, "sources": [item.doc_id]} for item in hits]

    def evidence_sufficient(self, evidence_matrix: list[dict[str, object]], round_index: int) -> bool:
        return True

    def build_deep_response(self, query: str, evidence_matrix: list[dict[str, object]], *, location: str) -> object:
        raise RuntimeError(f"{location} synthesis failed")

    def build_retrieval_only_response(self, query: str, hits: list[EvidenceItem]):
        from pkp.types import PreservationSuggestion, QueryResponse

        return QueryResponse(
            conclusion="retrieval fallback",
            evidence=hits,
            differences_or_conflicts=[],
            uncertainty="high",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=RuntimeMode.DEEP,
        )


def test_deep_research_runtime_falls_back_to_retrieval_only_when_synthesis_locations_fail() -> None:
    runtime = DeepResearchRuntime(
        routing_service=FakeRoutingService(),
        retrieval_service=FakeRetrievalService(),
        evidence_service=FakeEvidenceService(),
        session_runtime=SessionRuntime(),
        max_rounds=2,
    )

    response = runtime.run("research", make_policy(), session_id="fallback-session")

    assert response.conclusion == "retrieval fallback"
    assert response.runtime_mode is RuntimeMode.DEEP
