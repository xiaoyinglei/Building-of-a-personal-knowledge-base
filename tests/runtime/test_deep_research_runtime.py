from dataclasses import dataclass, field

from pkp.interfaces._runtime.deep_research_runtime import DeepResearchRuntime
from pkp.interfaces._runtime.session_runtime import SessionRuntime
from pkp.schema._types import (
    AccessPolicy,
    EvidenceItem,
    ExecutionLocationPreference,
    ExecutionPolicy,
    PreservationSuggestion,
    QueryResponse,
    RuntimeMode,
    TaskType,
)
from pkp.schema._types.query import ComplexityLevel


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


@dataclass
class FakeMemoryService:
    recalled_hints: list[str]
    recall_calls: list[tuple[str, list[str]]] = field(default_factory=list)
    recorded_episodes: list[dict[str, object]] = field(default_factory=list)

    def recall(self, query: str, source_scope: list[str]) -> list[str]:
        self.recall_calls.append((query, list(source_scope)))
        return list(self.recalled_hints)

    def record_episode(
        self,
        *,
        session_id: str,
        query: str,
        response: QueryResponse,
        evidence_matrix: list[dict[str, object]],
        source_scope: list[str],
    ) -> str:
        episode_id = f"episode-{len(self.recorded_episodes) + 1}"
        self.recorded_episodes.append(
            {
                "episode_id": episode_id,
                "session_id": session_id,
                "query": query,
                "response": response.conclusion,
                "evidence_matrix": list(evidence_matrix),
                "source_scope": list(source_scope),
            }
        )
        return episode_id


@dataclass
class ClockedRetrievalService:
    clock: "FakeClock"
    batches: list[list[EvidenceItem]]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def retrieve(self, query: str, policy: ExecutionPolicy, mode: RuntimeMode, round_index: int) -> list[EvidenceItem]:
        self.calls.append((query, round_index))
        self.clock.advance(2.0)
        return self.batches[round_index - 1]


@dataclass
class FakeClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


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
    assert len(session.evidence_matrix) == 2
    assert response.runtime_mode is RuntimeMode.DEEP
    assert response.preservation_suggestion.suggested is True


def test_deep_research_runtime_stops_after_token_budget_is_exhausted() -> None:
    runtime = DeepResearchRuntime(
        routing_service=FakeRoutingService(),
        retrieval_service=FakeRetrievalService(
            batches=[
                [hit("a", "doc-1", 0.9)],
                [hit("b", "doc-2", 0.8)],
            ]
        ),
        evidence_service=FakeEvidenceService(sufficient_after_round=99),
        session_runtime=SessionRuntime(),
        max_rounds=4,
    )
    policy = make_policy().model_copy(update={"token_budget": 3})

    response = runtime.run("compare docs", policy, session_id="token-budget")

    session = runtime.session_runtime.get("token-budget")
    assert session.sub_questions == ["sub-q-1", "sub-q-2"]
    assert len(session.evidence_matrix) == 1
    assert response.runtime_mode is RuntimeMode.DEEP


def test_deep_research_runtime_stops_after_wall_clock_budget_is_exhausted() -> None:
    clock = FakeClock()
    retrieval = ClockedRetrievalService(
        clock=clock,
        batches=[
            [hit("a", "doc-1", 0.9)],
            [hit("b", "doc-2", 0.8)],
        ],
    )
    runtime = DeepResearchRuntime(
        routing_service=FakeRoutingService(),
        retrieval_service=retrieval,
        evidence_service=FakeEvidenceService(sufficient_after_round=99),
        session_runtime=SessionRuntime(),
        max_rounds=4,
        clock=clock,
    )
    policy = make_policy().model_copy(update={"latency_budget": 1})

    response = runtime.run("compare docs", policy, session_id="wall-clock")

    assert retrieval.calls == [("sub-q-1", 1), ("sub-q-2", 1)]
    assert response.runtime_mode is RuntimeMode.DEEP


def test_deep_research_runtime_recalls_memory_hints_and_records_episode() -> None:
    memory_service = FakeMemoryService(recalled_hints=["fast path", "deep path"])
    runtime = DeepResearchRuntime(
        routing_service=FakeRoutingService(),
        retrieval_service=FakeRetrievalService(batches=[[hit("a", "doc-1", 0.9)], [hit("b", "doc-2", 0.8)]]),
        evidence_service=FakeEvidenceService(sufficient_after_round=2),
        session_runtime=SessionRuntime(),
        memory_service=memory_service,
        max_rounds=4,
    )
    policy = make_policy().model_copy(update={"source_scope": ["doc-1", "doc-2"]})

    response = runtime.run("compare docs", policy, session_id="memory-session")

    session = runtime.session_runtime.get("memory-session")
    assert response.runtime_mode is RuntimeMode.DEEP
    assert memory_service.recall_calls == [("compare docs", ["doc-1", "doc-2"])]
    assert session.memory_hints == ["fast path", "deep path"]
    assert session.episode_id == "episode-1"
    assert memory_service.recorded_episodes == [
        {
            "episode_id": "episode-1",
            "session_id": "memory-session",
            "query": "compare docs",
            "response": "deep answer for compare docs",
            "evidence_matrix": [
                {"claim": "evidence from doc-1", "sources": ["doc-1"]},
                {"claim": "evidence from doc-2", "sources": ["doc-2"]},
            ],
            "source_scope": ["doc-1", "doc-2"],
        }
    ]


def test_deep_research_runtime_respects_recursive_depth_limit() -> None:
    routing = FakeRoutingService(expansions=[["sub-q-1"], ["sub-q-2"], ["sub-q-3"], ["sub-q-4"]])
    retrieval = FakeRetrievalService(
        batches=[
            [hit("a", "doc-1", 0.9)],
            [hit("b", "doc-2", 0.8)],
            [hit("c", "doc-3", 0.7)],
            [hit("d", "doc-4", 0.6)],
        ]
    )
    runtime = DeepResearchRuntime(
        routing_service=routing,
        retrieval_service=retrieval,
        evidence_service=FakeEvidenceService(sufficient_after_round=99),
        session_runtime=SessionRuntime(),
        max_rounds=4,
        max_recursive_depth=1,
    )

    runtime.run("compare docs", make_policy(), session_id="depth-limited")

    assert retrieval.calls == [("sub-q-1", 1), ("sub-q-2", 2)]
