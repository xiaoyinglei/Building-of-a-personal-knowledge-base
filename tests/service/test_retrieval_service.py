from __future__ import annotations

from dataclasses import dataclass

from pkp.service.retrieval_service import RetrievalService
from pkp.types.access import AccessPolicy, ExternalRetrievalPolicy, Residency


@dataclass(frozen=True)
class FakeCandidate:
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str = "internal"
    source_id: str | None = None
    section_path: tuple[str, ...] = ()


def _build_service(
    *,
    full_text_candidates: list[FakeCandidate] | None = None,
    vector_candidates: list[FakeCandidate] | None = None,
    section_candidates: list[FakeCandidate] | None = None,
    graph_candidates: list[FakeCandidate] | None = None,
    web_candidates: list[FakeCandidate] | None = None,
) -> tuple[RetrievalService, dict[str, object]]:
    calls: dict[str, object] = {
        "rerank_inputs": [],
        "graph_calls": 0,
        "web_calls": 0,
    }

    def full_text_retriever(query: str, source_scope: list[str]) -> list[FakeCandidate]:
        calls["full_text_query"] = query
        calls["full_text_scope"] = list(source_scope)
        return list(full_text_candidates or [])

    def vector_retriever(query: str, source_scope: list[str]) -> list[FakeCandidate]:
        calls["vector_query"] = query
        calls["vector_scope"] = list(source_scope)
        return list(vector_candidates or [])

    def section_retriever(query: str, source_scope: list[str]) -> list[FakeCandidate]:
        calls["section_query"] = query
        calls["section_scope"] = list(source_scope)
        return list(section_candidates or [])

    def graph_expander(
        query: str, source_scope: list[str], non_graph_evidence: list[FakeCandidate]
    ) -> list[FakeCandidate]:
        calls["graph_calls"] = int(calls["graph_calls"]) + 1
        calls["graph_scope"] = list(source_scope)
        calls["graph_non_graph_ids"] = [candidate.chunk_id for candidate in non_graph_evidence]
        return list(graph_candidates or [])

    def web_retriever(query: str, source_scope: list[str]) -> list[FakeCandidate]:
        calls["web_calls"] = int(calls["web_calls"]) + 1
        calls["web_scope"] = list(source_scope)
        return list(web_candidates or [])

    def reranker(query: str, candidates: list[FakeCandidate]) -> list[FakeCandidate]:
        calls["rerank_inputs"].append([candidate.chunk_id for candidate in candidates])
        return list(candidates)

    service = RetrievalService(
        full_text_retriever=full_text_retriever,
        vector_retriever=vector_retriever,
        section_retriever=section_retriever,
        graph_expander=graph_expander,
        web_retriever=web_retriever,
        reranker=reranker,
    )
    return service, calls


def test_retrieval_service_rrf_fuses_branches_before_rerank_and_honors_source_scope() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate("chunk-a", "doc-a", "alpha", "#a", 0.9, 1),
            FakeCandidate("chunk-b", "doc-b", "beta", "#b", 0.8, 2),
        ],
        vector_candidates=[
            FakeCandidate("chunk-b", "doc-b", "beta", "#b", 0.95, 1),
            FakeCandidate("chunk-c", "doc-c", "gamma", "#c", 0.7, 2),
        ],
    )

    result = service.retrieve(
        "Compare the tradeoffs between Alpha and Beta",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a", "doc-b"],
    )

    assert calls["rerank_inputs"] == [["chunk-b", "chunk-a"]]
    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-b", "chunk-a"]


def test_retrieval_service_separates_external_evidence_and_expands_graph() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate("chunk-a", "doc-a", "alpha", "#a", 0.9, 1),
        ],
        web_candidates=[
            FakeCandidate(
                "chunk-web",
                "web-doc",
                "external alpha",
                "#web",
                0.7,
                1,
                source_kind="external",
                source_id="web",
            )
        ],
        graph_candidates=[
            FakeCandidate(
                "chunk-graph",
                "doc-a",
                "graph alpha",
                "#g",
                0.6,
                1,
                source_kind="graph",
                source_id="graph",
            )
        ],
    )

    result = service.retrieve(
        "Compare the tradeoffs between Alpha and Beta",
        access_policy=AccessPolicy.default(),
    )

    assert calls["web_calls"] == 1
    assert [item.evidence_kind for item in result.evidence.internal] == ["internal"]
    assert [item.evidence_kind for item in result.evidence.external] == ["external"]
    assert [item.evidence_kind for item in result.evidence.graph] == ["graph"]
    assert calls["graph_calls"] == 1
    assert calls["graph_non_graph_ids"] == ["chunk-a"]


def test_retrieval_service_blocks_web_search_when_policy_denies() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate("chunk-a", "doc-a", "alpha", "#a", 0.9, 1),
        ],
        web_candidates=[
            FakeCandidate(
                "chunk-web",
                "web-doc",
                "external alpha",
                "#web",
                0.7,
                1,
                source_kind="external",
                source_id="web",
            )
        ],
    )

    result = service.retrieve(
        "Compare the tradeoffs between Alpha and Beta",
        access_policy=AccessPolicy(
            residency=Residency.CLOUD_ALLOWED,
            external_retrieval=ExternalRetrievalPolicy.DENY,
        ),
        source_scope=["doc-a"],
    )

    assert calls["web_calls"] == 0
    assert result.self_check.retrieve_more is True
    assert result.self_check.evidence_sufficient is False
    assert result.evidence.external == []


def test_retrieval_service_skips_web_search_when_internal_evidence_is_already_sufficient() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate("chunk-a", "doc-a", "alpha", "#a", 0.9, 1),
            FakeCandidate("chunk-b", "doc-b", "beta", "#b", 0.85, 2),
            FakeCandidate("chunk-c", "doc-c", "gamma", "#c", 0.8, 3),
            FakeCandidate("chunk-d", "doc-d", "delta", "#d", 0.75, 4),
        ],
        web_candidates=[
            FakeCandidate(
                "chunk-web",
                "web-doc",
                "external alpha",
                "#web",
                0.7,
                1,
                source_kind="external",
                source_id="web",
            )
        ],
    )

    result = service.retrieve(
        "compare Alpha and Beta",
        access_policy=AccessPolicy.default(),
    )

    assert result.self_check.evidence_sufficient is True
    assert calls["web_calls"] == 0
    assert result.evidence.external == []
