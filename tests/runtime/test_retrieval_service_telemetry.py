from dataclasses import dataclass

from pkp.query.context import RoutingDecision
from pkp.query.retrieve import RetrievalService
from pkp.utils._telemetry import TelemetryService
from pkp.schema._types import AccessPolicy, RuntimeMode, TaskType
from pkp.schema._types.query import ComplexityLevel


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str = "internal"
    source_id: str | None = None
    section_path: tuple[str, ...] = ()


@dataclass
class StubRoutingService:
    def route(self, query: str, *, source_scope=(), access_policy=None) -> RoutingDecision:
        del query, source_scope, access_policy
        return RoutingDecision(
            task_type=TaskType.RESEARCH,
            complexity_level=ComplexityLevel.L4_RESEARCH,
            runtime_mode=RuntimeMode.DEEP,
            graph_expansion_allowed=True,
        )


def test_retrieval_service_records_branch_usage_and_graph_expansion() -> None:
    telemetry = TelemetryService.create_in_memory()
    retrieval_service = RetrievalService(
        full_text_retriever=lambda _query, _scope: [
            Candidate(
                chunk_id="full-1",
                doc_id="doc-1",
                text="Alpha",
                citation_anchor="doc-1#1",
                score=0.9,
                rank=1,
            )
        ],
        vector_retriever=lambda _query, _scope: [
            Candidate(
                chunk_id="vector-1",
                doc_id="doc-1",
                text="Beta",
                citation_anchor="doc-1#2",
                score=0.8,
                rank=1,
            )
        ],
        section_retriever=lambda _query, _scope: [],
        graph_expander=lambda _query, _scope, _evidence: [
            Candidate(
                chunk_id="graph-1",
                doc_id="doc-2",
                text="Gamma",
                citation_anchor="doc-2#1",
                score=0.7,
                rank=1,
                source_kind="graph",
            )
        ],
        reranker=lambda _query, candidates: candidates,
        routing_service=StubRoutingService(),
        telemetry_service=telemetry,
    )

    result = retrieval_service.retrieve(
        "research alpha",
        access_policy=AccessPolicy.default(),
        source_scope=[],
    )

    assert result.graph_expanded is True
    events = telemetry.list_events()
    assert [event.name for event in events] == [
        "retrieval.branch_used",
        "retrieval.branch_used",
        "retrieval.branch_used",
        "retrieval.branch_used",
        "retrieval.rrf_fused",
        "retrieval.rerank_effectiveness",
        "retrieval.graph_expanded",
        "artifact.preservation_suggested",
    ]
    assert [event.payload["branch"] for event in events[:4]] == ["local", "global", "vector", "full_text"]
    assert [event.payload["count"] for event in events[:4]] == [0, 0, 1, 1]
    assert events[4].payload["candidate_count"] == 2
    assert events[5].payload["reordered"] is False
    assert events[6].payload["added_count"] == 1
    assert events[7].payload["artifact_type"] == "topic_page"


def test_retrieval_service_records_rrf_rerank_and_preservation_telemetry() -> None:
    telemetry = TelemetryService.create_in_memory()
    retrieval_service = RetrievalService(
        full_text_retriever=lambda _query, _scope: [
            Candidate(
                chunk_id="full-1",
                doc_id="doc-a",
                text="Alpha",
                citation_anchor="doc-a#1",
                score=0.9,
                rank=1,
            ),
            Candidate(
                chunk_id="full-2",
                doc_id="doc-b",
                text="Beta",
                citation_anchor="doc-b#1",
                score=0.8,
                rank=2,
            ),
        ],
        vector_retriever=lambda _query, _scope: [
            Candidate(
                chunk_id="full-2",
                doc_id="doc-b",
                text="Beta",
                citation_anchor="doc-b#1",
                score=0.95,
                rank=1,
            ),
            Candidate(
                chunk_id="full-1",
                doc_id="doc-a",
                text="Alpha",
                citation_anchor="doc-a#1",
                score=0.85,
                rank=2,
            ),
        ],
        section_retriever=lambda _query, _scope: [],
        graph_expander=lambda _query, _scope, _evidence: [],
        reranker=lambda _query, candidates: list(reversed(candidates)),
        routing_service=StubRoutingService(),
        telemetry_service=telemetry,
    )

    result = retrieval_service.retrieve(
        "Compare Alpha and Beta",
        access_policy=AccessPolicy.default(),
        source_scope=[],
    )

    assert result.preservation_suggestion.suggested is True
    events = telemetry.list_events()
    assert [event.name for event in events] == [
        "retrieval.branch_used",
        "retrieval.branch_used",
        "retrieval.branch_used",
        "retrieval.branch_used",
        "retrieval.rrf_fused",
        "retrieval.rerank_effectiveness",
        "artifact.preservation_suggested",
    ]
    assert [event.payload["branch"] for event in events[:4]] == ["local", "global", "vector", "full_text"]
    assert [event.payload["count"] for event in events[:4]] == [0, 0, 2, 2]
    assert events[4].payload["duplicate_count"] == 2
    assert events[5].payload["reordered"] is True
    assert events[6].payload["artifact_type"] == "comparison_page"
