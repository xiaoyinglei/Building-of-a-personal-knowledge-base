from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pkp.service.retrieval_service import RetrievalService
from pkp.types.access import AccessPolicy, ExternalRetrievalPolicy, Residency
from pkp.types.content import ChunkRole


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
    chunk_role: ChunkRole = ChunkRole.CHILD
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    parent_text: str | None = None


def _build_service(
    *,
    full_text_candidates: list[FakeCandidate] | None = None,
    vector_candidates: list[FakeCandidate] | None = None,
    section_candidates: list[FakeCandidate] | None = None,
    special_candidates: list[FakeCandidate] | None = None,
    graph_candidates: list[FakeCandidate] | None = None,
    web_candidates: list[FakeCandidate] | None = None,
) -> tuple[RetrievalService, dict[str, object]]:
    calls: dict[str, Any] = {
        "rerank_inputs": [],
        "graph_calls": 0,
        "web_calls": 0,
        "special_calls": 0,
        "structure_calls": 0,
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
        calls["structure_calls"] = cast(int, calls["structure_calls"]) + 1
        return list(section_candidates or [])

    def special_retriever(query: str, source_scope: list[str]) -> list[FakeCandidate]:
        calls["special_query"] = query
        calls["special_scope"] = list(source_scope)
        calls["special_calls"] = cast(int, calls["special_calls"]) + 1
        return list(special_candidates or [])

    def graph_expander(
        query: str, source_scope: list[str], non_graph_evidence: list[FakeCandidate]
    ) -> list[FakeCandidate]:
        calls["graph_calls"] = cast(int, calls["graph_calls"]) + 1
        calls["graph_scope"] = list(source_scope)
        calls["graph_non_graph_ids"] = [candidate.chunk_id for candidate in non_graph_evidence]
        return list(graph_candidates or [])

    def web_retriever(query: str, source_scope: list[str]) -> list[FakeCandidate]:
        calls["web_calls"] = cast(int, calls["web_calls"]) + 1
        calls["web_scope"] = list(source_scope)
        return list(web_candidates or [])

    def reranker(query: str, candidates: list[FakeCandidate]) -> list[FakeCandidate]:
        cast(list[list[str]], calls["rerank_inputs"]).append([candidate.chunk_id for candidate in candidates])
        return list(candidates)

    service = RetrievalService(
        full_text_retriever=cast(Any, full_text_retriever),
        vector_retriever=cast(Any, vector_retriever),
        section_retriever=cast(Any, section_retriever),
        special_retriever=cast(Any, special_retriever),
        graph_expander=cast(Any, graph_expander),
        web_retriever=cast(Any, web_retriever),
        reranker=cast(Any, reranker),
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


def test_retrieval_service_prefers_definition_chunk_when_vector_and_sparse_signals_disagree() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate(
                "chunk-cli",
                "doc-a",
                'uv run python -m pkp.ui.cli query --mode fast --query "这个项目做什么？"',
                "#query",
                1.0,
                1,
                section_path=("查询",),
            ),
            FakeCandidate(
                "chunk-desc",
                "doc-a",
                "一个以可靠性为优先的个人知识平台，用来完成资料接入、索引构建、检索问答和知识沉淀。",
                "#intro",
                0.9,
                2,
                section_path=("个人知识平台",),
            ),
        ],
        vector_candidates=[
            FakeCandidate(
                "chunk-desc",
                "doc-a",
                "一个以可靠性为优先的个人知识平台，用来完成资料接入、索引构建、检索问答和知识沉淀。",
                "#intro",
                0.61,
                1,
                section_path=("个人知识平台",),
            ),
            FakeCandidate(
                "chunk-cli",
                "doc-a",
                'uv run python -m pkp.ui.cli query --mode fast --query "这个项目做什么？"',
                "#query",
                0.51,
                2,
                section_path=("查询",),
            ),
        ],
    )

    result = service.retrieve(
        "这个项目做什么？",
        access_policy=AccessPolicy.default(),
    )

    assert calls["rerank_inputs"] == [["chunk-desc", "chunk-cli"]]
    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-desc", "chunk-cli"]


def test_retrieval_service_uses_query_understanding_to_enable_special_branch() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate("chunk-text", "doc-a", "普通段落", "#a", 0.8, 1),
        ],
        special_candidates=[
            FakeCandidate(
                "chunk-table",
                "doc-a",
                "| 指标 | 数值 |",
                "#table",
                0.9,
                1,
                chunk_role=ChunkRole.SPECIAL,
                special_chunk_type="table",
            ),
        ],
    )

    result = service.retrieve(
        "表格里指标数值是多少？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
    )

    assert calls["special_calls"] == 1
    assert result.diagnostics.query_understanding is not None
    assert result.diagnostics.query_understanding.needs_special is True
    assert result.diagnostics.branch_hits["special"] == 1


def test_retrieval_service_applies_parent_backfill_to_child_evidence() -> None:
    service, _calls = _build_service(
        full_text_candidates=[
            FakeCandidate(
                "chunk-child",
                "doc-a",
                "命中的子块只有前半句。",
                "#child",
                0.9,
                1,
                parent_chunk_id="parent-a",
                parent_text="命中的子块只有前半句。这里是完整的父块上下文，包含后半句和结论。",
            ),
        ],
    )

    result = service.retrieve(
        "这个问题的完整上下文是什么？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
    )

    assert result.evidence.internal[0].text == "命中的子块只有前半句。这里是完整的父块上下文，包含后半句和结论。"
    assert result.evidence.internal[0].parent_chunk_id == "parent-a"
    assert result.diagnostics.parent_backfilled_count == 1


def test_retrieval_service_uses_structure_constraints_for_heading_queries() -> None:
    service, calls = _build_service(
        section_candidates=[
            FakeCandidate(
                "chunk-structure",
                "doc-a",
                "系统架构分为三层。",
                "#arch",
                0.8,
                1,
                section_path=("系统架构",),
            ),
        ],
    )

    result = service.retrieve(
        "系统架构分为哪几层？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
    )

    assert calls["structure_calls"] == 1
    assert result.diagnostics.query_understanding is not None
    assert result.diagnostics.query_understanding.structure_constraints["preferred_section_terms"] == ["系统架构"]
