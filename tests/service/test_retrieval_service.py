from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from rag.assembly import ChatCapabilityBinding
from rag.retrieval.analysis import QueryUnderstandingService
from rag.retrieval.models import QueryMode
from rag.retrieval.orchestrator import RetrievalService
from rag.schema.core import ChunkRole
from rag.schema.runtime import AccessPolicy, ExternalRetrievalPolicy, Residency


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
    metadata: dict[str, str] | None = None


class FakeRetrievalBackend:
    chat_model_name = "fake-retrieval-query-understanding"

    def chat(self, prompt: str) -> str:
        query = prompt.split("Query: ", 1)[1].rsplit("\nJSON only.", 1)[0]
        payload = {
            "Compare the tradeoffs between Alpha and Beta": {
                "task_type": "comparison",
                "complexity_level": "L3_comparative",
                "query_type": "comparison",
            },
            "Compare how Alpha depends on Beta": {
                "task_type": "research",
                "complexity_level": "L4_research",
                "query_type": "process",
                "needs_graph_expansion": True,
            },
            "compare Alpha and Beta": {
                "task_type": "comparison",
                "complexity_level": "L3_comparative",
                "query_type": "comparison",
            },
            "What is Alpha Engine?": {
                "task_type": "lookup",
                "complexity_level": "L1_direct",
                "query_type": "lookup",
            },
            "这个项目做什么？": {
                "task_type": "lookup",
                "complexity_level": "L1_direct",
                "query_type": "lookup",
            },
            "表格里指标数值是多少？": {
                "task_type": "single_doc_qa",
                "complexity_level": "L2_scoped",
                "query_type": "special_lookup",
                "needs_special": True,
                "special_targets": ["table"],
            },
            "pptx 第2页表格里的系统架构是什么？": {
                "task_type": "single_doc_qa",
                "complexity_level": "L2_scoped",
                "query_type": "special_lookup",
                "needs_special": True,
                "needs_structure": True,
                "needs_metadata": True,
                "structure_constraints": {
                    "match_strategy": "heading",
                    "requires_structure_match": True,
                    "prefer_heading_match": True,
                    "semantic_section_families": ["architecture"],
                    "preferred_section_terms": ["系统架构"],
                    "heading_hints": ["系统架构"],
                },
                "metadata_filters": {
                    "source_types": ["pptx"],
                    "page_numbers": [2],
                },
                "special_targets": ["table"],
                "preferred_section_terms": ["系统架构"],
            },
            "这个问题的完整上下文是什么？": {
                "task_type": "lookup",
                "complexity_level": "L1_direct",
                "query_type": "lookup",
            },
            "系统架构分为哪几层？": {
                "task_type": "lookup",
                "complexity_level": "L2_scoped",
                "query_type": "structure_lookup",
                "needs_structure": True,
                "structure_constraints": {
                    "match_strategy": "semantic",
                    "requires_structure_match": True,
                    "semantic_section_families": ["architecture"],
                    "preferred_section_terms": ["系统架构"],
                },
                "preferred_section_terms": ["系统架构"],
            },
            "第2页讲了什么风险？": {
                "task_type": "single_doc_qa",
                "complexity_level": "L2_scoped",
                "query_type": "scoped_lookup",
                "needs_metadata": True,
                "metadata_filters": {"page_numbers": [2]},
            },
            "这个文档里提到了哪些内容？": {
                "task_type": "synthesis",
                "complexity_level": "L4_research",
                "query_type": "summary",
            },
            "What is Alpha?": {
                "task_type": "lookup",
                "complexity_level": "L1_direct",
                "query_type": "lookup",
            },
            "Alpha engine details": {
                "task_type": "lookup",
                "complexity_level": "L1_direct",
                "query_type": "lookup",
            },
            "How does Alpha depend on Beta?": {
                "task_type": "research",
                "complexity_level": "L4_research",
                "query_type": "process",
                "needs_graph_expansion": True,
            },
            "How are Alpha and Beta related?": {
                "task_type": "comparison",
                "complexity_level": "L3_comparative",
                "query_type": "comparison",
            },
            "pptx 第2页表格里的 Alpha 和 Beta 有什么关系？": {
                "task_type": "comparison",
                "complexity_level": "L3_comparative",
                "query_type": "comparison",
                "needs_special": True,
                "needs_metadata": True,
                "metadata_filters": {
                    "source_types": ["pptx"],
                    "page_numbers": [2],
                },
                "special_targets": ["table"],
            },
        }.get(query, {"task_type": "lookup", "complexity_level": "L1_direct", "query_type": "lookup"})
        return json.dumps(payload, ensure_ascii=False)


def _query_understanding_service() -> QueryUnderstandingService:
    binding = ChatCapabilityBinding(backend=FakeRetrievalBackend(), location="local")
    return QueryUnderstandingService(chat_bindings=(binding,))


def _build_service(
    *,
    full_text_candidates: list[FakeCandidate] | None = None,
    vector_candidates: list[FakeCandidate] | None = None,
    local_candidates: list[FakeCandidate] | None = None,
    global_candidates: list[FakeCandidate] | None = None,
    section_candidates: list[FakeCandidate] | None = None,
    special_candidates: list[FakeCandidate] | None = None,
    metadata_candidates: list[FakeCandidate] | None = None,
    graph_candidates: list[FakeCandidate] | None = None,
    web_candidates: list[FakeCandidate] | None = None,
) -> tuple[RetrievalService, dict[str, object]]:
    calls: dict[str, Any] = {
        "rerank_inputs": [],
        "graph_calls": 0,
        "web_calls": 0,
        "special_calls": 0,
        "structure_calls": 0,
        "metadata_calls": 0,
        "full_text_calls": 0,
        "vector_calls": 0,
        "local_calls": 0,
        "global_calls": 0,
    }

    def full_text_retriever(query: str, source_scope: list[str], query_understanding: object) -> list[FakeCandidate]:
        calls["full_text_query"] = query
        calls["full_text_scope"] = list(source_scope)
        calls["full_text_understanding"] = query_understanding
        calls["full_text_calls"] = cast(int, calls["full_text_calls"]) + 1
        return list(full_text_candidates or [])

    def vector_retriever(query: str, source_scope: list[str], query_understanding: object) -> list[FakeCandidate]:
        calls["vector_query"] = query
        calls["vector_scope"] = list(source_scope)
        calls["vector_understanding"] = query_understanding
        calls["vector_calls"] = cast(int, calls["vector_calls"]) + 1
        return list(vector_candidates or [])

    def local_retriever(query: str, source_scope: list[str], query_understanding: object) -> list[FakeCandidate]:
        calls["local_query"] = query
        calls["local_scope"] = list(source_scope)
        calls["local_understanding"] = query_understanding
        calls["local_calls"] = cast(int, calls["local_calls"]) + 1
        return list(local_candidates or [])

    def global_retriever(query: str, source_scope: list[str], query_understanding: object) -> list[FakeCandidate]:
        calls["global_query"] = query
        calls["global_scope"] = list(source_scope)
        calls["global_understanding"] = query_understanding
        calls["global_calls"] = cast(int, calls["global_calls"]) + 1
        return list(global_candidates or [])

    def section_retriever(query: str, source_scope: list[str], query_understanding: object) -> list[FakeCandidate]:
        calls["section_query"] = query
        calls["section_scope"] = list(source_scope)
        calls["section_understanding"] = query_understanding
        calls["structure_calls"] = cast(int, calls["structure_calls"]) + 1
        return list(section_candidates or [])

    def special_retriever(query: str, source_scope: list[str], query_understanding: object) -> list[FakeCandidate]:
        calls["special_query"] = query
        calls["special_scope"] = list(source_scope)
        calls["special_understanding"] = query_understanding
        calls["special_calls"] = cast(int, calls["special_calls"]) + 1
        return list(special_candidates or [])

    def metadata_retriever(query: str, source_scope: list[str], query_understanding: object) -> list[FakeCandidate]:
        calls["metadata_query"] = query
        calls["metadata_scope"] = list(source_scope)
        calls["metadata_understanding"] = query_understanding
        calls["metadata_calls"] = cast(int, calls["metadata_calls"]) + 1
        return list(metadata_candidates or [])

    def graph_expander(
        query: str, source_scope: list[str], non_graph_evidence: list[FakeCandidate]
    ) -> list[FakeCandidate]:
        calls["graph_calls"] = cast(int, calls["graph_calls"]) + 1
        calls["graph_scope"] = list(source_scope)
        calls["graph_non_graph_ids"] = [candidate.chunk_id for candidate in non_graph_evidence]
        return list(graph_candidates or [])

    def web_retriever(query: str, source_scope: list[str], query_understanding: object) -> list[FakeCandidate]:
        calls["web_calls"] = cast(int, calls["web_calls"]) + 1
        calls["web_scope"] = list(source_scope)
        calls["web_understanding"] = query_understanding
        return list(web_candidates or [])

    def reranker(query: str, candidates: list[FakeCandidate]) -> list[FakeCandidate]:
        cast(list[list[str]], calls["rerank_inputs"]).append([candidate.chunk_id for candidate in candidates])
        return list(candidates)

    service = RetrievalService(
        full_text_retriever=cast(Any, full_text_retriever),
        vector_retriever=cast(Any, vector_retriever),
        local_retriever=cast(Any, local_retriever),
        global_retriever=cast(Any, global_retriever),
        section_retriever=cast(Any, section_retriever),
        special_retriever=cast(Any, special_retriever),
        metadata_retriever=cast(Any, metadata_retriever),
        graph_expander=cast(Any, graph_expander),
        web_retriever=cast(Any, web_retriever),
        reranker=cast(Any, reranker),
        query_understanding_service=_query_understanding_service(),
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
        "Compare how Alpha depends on Beta",
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


def test_retrieval_service_bypass_mode_skips_retrieval_and_graph_expansion() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate("chunk-a", "doc-a", "alpha", "#a", 0.8, 1),
        ],
        vector_candidates=[
            FakeCandidate("chunk-b", "doc-b", "beta", "#b", 0.9, 1),
        ],
        section_candidates=[
            FakeCandidate("chunk-section", "doc-a", "section", "#s", 0.95, 1),
        ],
        special_candidates=[
            FakeCandidate("chunk-special", "doc-a", "special", "#sp", 0.95, 1),
        ],
        metadata_candidates=[
            FakeCandidate("chunk-meta", "doc-a", "meta", "#m", 0.95, 1),
        ],
        graph_candidates=[
            FakeCandidate("chunk-graph", "doc-a", "graph", "#g", 0.95, 1, source_kind="graph"),
        ],
        web_candidates=[
            FakeCandidate("chunk-web", "web-doc", "web", "#w", 0.95, 1, source_kind="external"),
        ],
    )

    result = service.retrieve(
        "What is Alpha Engine?",
        access_policy=AccessPolicy.default(),
        query_mode=QueryMode.BYPASS,
    )

    assert result.evidence.internal == []
    assert result.reranked_chunk_ids == []
    assert calls["vector_calls"] == 0
    assert calls["full_text_calls"] == 0
    assert calls["structure_calls"] == 0
    assert calls["special_calls"] == 0
    assert calls["metadata_calls"] == 0
    assert calls["graph_calls"] == 0
    assert calls["web_calls"] == 0


def test_retrieval_service_keeps_definition_and_cli_candidates_for_downstream_rerank() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate(
                "chunk-cli",
                "doc-a",
                'uv run python -m rag.cli query --mode fast --query "这个项目做什么？"',
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
                'uv run python -m rag.cli query --mode fast --query "这个项目做什么？"',
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

    assert len(calls["rerank_inputs"]) == 1
    assert set(calls["rerank_inputs"][0]) == {"chunk-desc", "chunk-cli"}
    assert {item.chunk_id for item in result.evidence.internal} == {"chunk-desc", "chunk-cli"}


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


def test_retrieval_service_special_query_prioritizes_special_evidence_in_unified_fusion() -> None:
    service, calls = _build_service(
        full_text_candidates=[
            FakeCandidate("chunk-text", "doc-a", "本段讲了总体背景。", "#a", 0.98, 1),
        ],
        special_candidates=[
            FakeCandidate(
                "chunk-table",
                "doc-a",
                "| 指标 | 数值 |\n|---|---|\n| 收入 | 120 |",
                "#table",
                0.72,
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
    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-table", "chunk-text"]


def test_retrieval_service_hybrid_mode_keeps_structure_special_and_metadata_branches() -> None:
    service, calls = _build_service(
        local_candidates=[FakeCandidate("chunk-local", "doc-a", "local hit", "#l", 0.9, 1)],
        global_candidates=[FakeCandidate("chunk-global", "doc-a", "global hit", "#g", 0.88, 1)],
        section_candidates=[FakeCandidate("chunk-section", "doc-a", "系统架构分三层。", "#s", 0.92, 1)],
        special_candidates=[
            FakeCandidate(
                "chunk-table",
                "doc-a",
                "| 指标 | 数值 |",
                "#table",
                0.91,
                1,
                chunk_role=ChunkRole.SPECIAL,
                special_chunk_type="table",
            )
        ],
        metadata_candidates=[
            FakeCandidate("chunk-page", "doc-a", "第二页介绍了架构。", "#page-2", 0.89, 1, metadata={"page_no": "2"})
        ],
    )

    result = service.retrieve(
        "pptx 第2页表格里的系统架构是什么？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
        query_mode=QueryMode.HYBRID,
    )

    assert calls["local_calls"] == 1
    assert calls["global_calls"] == 1
    assert calls["structure_calls"] == 1
    assert calls["special_calls"] == 1
    assert calls["metadata_calls"] == 1
    assert result.diagnostics.branch_hits["section"] == 1
    assert result.diagnostics.branch_hits["special"] == 1
    assert result.diagnostics.branch_hits["metadata"] == 1


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
    assert "系统架构" in result.diagnostics.query_understanding.preferred_section_terms
    assert result.diagnostics.query_understanding.structure_constraints.semantic_section_families == ["architecture"]


def test_retrieval_service_uses_metadata_branch_for_page_constrained_queries() -> None:
    service, calls = _build_service(
        metadata_candidates=[
            FakeCandidate(
                "chunk-page",
                "doc-a",
                "第二页记录了主要风险与缓解计划。",
                "#page-2",
                0.9,
                1,
                metadata={"page_no": "2"},
            ),
        ],
    )

    result = service.retrieve(
        "第2页讲了什么风险？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
    )

    assert calls["metadata_calls"] == 1
    assert result.diagnostics.query_understanding is not None
    assert result.diagnostics.query_understanding.metadata_filters.page_numbers == [2]
    assert result.diagnostics.branch_hits["metadata"] == 1


def test_retrieval_service_filters_out_candidates_that_violate_explicit_page_constraints() -> None:
    service, _calls = _build_service(
        full_text_candidates=[
            FakeCandidate(
                "chunk-page-5",
                "doc-a",
                "第五页记录了其他内容。",
                "#page-5",
                0.95,
                1,
                metadata={"page_no": "5"},
            ),
            FakeCandidate(
                "chunk-page-2",
                "doc-a",
                "第二页记录了主要风险与缓解计划。",
                "#page-2",
                0.85,
                2,
                metadata={"page_no": "2"},
            ),
        ],
    )

    result = service.retrieve(
        "第2页讲了什么风险？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
    )

    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-page-2"]


def test_retrieval_service_filters_out_candidates_that_violate_explicit_section_constraints() -> None:
    service, _calls = _build_service(
        full_text_candidates=[
            FakeCandidate(
                "chunk-intro",
                "doc-a",
                "系统由多个模块组成。",
                "#intro",
                0.96,
                1,
                section_path=("项目介绍",),
            ),
            FakeCandidate(
                "chunk-arch",
                "doc-a",
                "系统架构分为接入层、检索层、生成层。",
                "#arch",
                0.88,
                2,
                section_path=("系统架构",),
            ),
        ],
    )

    result = service.retrieve(
        "系统架构分为哪几层？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
    )

    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-arch"]


def test_retrieval_service_collapses_duplicate_child_hits_from_same_parent() -> None:
    service, _calls = _build_service(
        full_text_candidates=[
            FakeCandidate(
                "chunk-a1",
                "doc-a",
                "这是父块里的第一小段。",
                "#a1",
                0.95,
                1,
                parent_chunk_id="parent-a",
            ),
            FakeCandidate(
                "chunk-a2",
                "doc-a",
                "这是父块里的第二小段。",
                "#a2",
                0.92,
                2,
                parent_chunk_id="parent-a",
            ),
            FakeCandidate(
                "chunk-b1",
                "doc-a",
                "这是另一个父块里的内容。",
                "#b1",
                0.88,
                3,
                parent_chunk_id="parent-b",
            ),
        ],
    )

    result = service.retrieve(
        "这个文档里提到了哪些内容？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
    )

    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-a1", "chunk-b1"]
    assert result.diagnostics.collapsed_candidate_count == 1


def test_retrieval_service_naive_mode_uses_only_vector_branch() -> None:
    service, calls = _build_service(
        vector_candidates=[FakeCandidate("chunk-vector", "doc-a", "vector hit", "#v", 0.92, 1)],
        local_candidates=[FakeCandidate("chunk-local", "doc-a", "local hit", "#l", 0.95, 1)],
        global_candidates=[FakeCandidate("chunk-global", "doc-a", "global hit", "#g", 0.91, 1)],
        full_text_candidates=[FakeCandidate("chunk-sparse", "doc-a", "sparse hit", "#s", 0.89, 1)],
    )

    result = service.retrieve(
        "What is Alpha?",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
        query_mode=QueryMode.NAIVE,
    )

    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-vector"]
    assert calls["vector_calls"] == 1
    assert calls["local_calls"] == 0
    assert calls["global_calls"] == 0
    assert calls["full_text_calls"] == 0


def test_retrieval_service_local_mode_uses_only_entity_local_branch() -> None:
    service, calls = _build_service(
        local_candidates=[FakeCandidate("chunk-local", "doc-a", "local hit", "#l", 0.95, 1)],
        vector_candidates=[FakeCandidate("chunk-vector", "doc-a", "vector hit", "#v", 0.92, 1)],
        global_candidates=[FakeCandidate("chunk-global", "doc-a", "global hit", "#g", 0.91, 1)],
    )

    result = service.retrieve(
        "Alpha engine details",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
        query_mode="local",
    )

    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-local"]
    assert calls["local_calls"] == 1
    assert calls["global_calls"] == 0
    assert calls["vector_calls"] == 0
    assert calls["full_text_calls"] == 0


def test_retrieval_service_global_mode_uses_only_relation_global_branch() -> None:
    service, calls = _build_service(
        global_candidates=[FakeCandidate("chunk-global", "doc-a", "global hit", "#g", 0.97, 1)],
        local_candidates=[FakeCandidate("chunk-local", "doc-a", "local hit", "#l", 0.95, 1)],
        vector_candidates=[FakeCandidate("chunk-vector", "doc-a", "vector hit", "#v", 0.92, 1)],
    )

    result = service.retrieve(
        "How does Alpha depend on Beta?",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
        query_mode=QueryMode.GLOBAL,
    )

    assert [item.chunk_id for item in result.evidence.internal] == ["chunk-global"]
    assert calls["global_calls"] == 1
    assert calls["local_calls"] == 0
    assert calls["vector_calls"] == 0
    assert calls["full_text_calls"] == 0


def test_retrieval_service_hybrid_mode_combines_local_and_global_without_vector() -> None:
    service, calls = _build_service(
        local_candidates=[FakeCandidate("chunk-local", "doc-a", "local hit", "#l", 0.95, 1)],
        global_candidates=[FakeCandidate("chunk-global", "doc-a", "global hit", "#g", 0.94, 1)],
        vector_candidates=[FakeCandidate("chunk-vector", "doc-a", "vector hit", "#v", 0.99, 1)],
    )

    result = service.retrieve(
        "How are Alpha and Beta related?",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
        query_mode=QueryMode.HYBRID,
    )

    assert {item.chunk_id for item in result.evidence.internal} == {"chunk-local", "chunk-global"}
    assert calls["local_calls"] == 1
    assert calls["global_calls"] == 1
    assert calls["vector_calls"] == 0
    assert calls["full_text_calls"] == 0


def test_retrieval_service_hybrid_and_mix_publish_distinct_mode_executor_budgets() -> None:
    service, _calls = _build_service(
        local_candidates=[FakeCandidate("chunk-local", "doc-a", "local hit", "#l", 0.95, 1)],
        global_candidates=[FakeCandidate("chunk-global", "doc-a", "global hit", "#g", 0.94, 1)],
        vector_candidates=[FakeCandidate("chunk-vector", "doc-a", "vector hit", "#v", 0.91, 1)],
        full_text_candidates=[FakeCandidate("chunk-sparse", "doc-a", "sparse hit", "#s", 0.90, 1)],
        section_candidates=[FakeCandidate("chunk-section", "doc-a", "section hit", "#sec", 0.89, 1)],
        special_candidates=[
            FakeCandidate(
                "chunk-table",
                "doc-a",
                "| 指标 | 数值 |",
                "#table",
                0.88,
                1,
                chunk_role=ChunkRole.SPECIAL,
                special_chunk_type="table",
            )
        ],
        metadata_candidates=[
            FakeCandidate("chunk-page", "doc-a", "第二页信息。", "#page-2", 0.87, 1, metadata={"page_no": "2"})
        ],
    )

    hybrid = service.retrieve(
        "pptx 第2页表格里的 Alpha 和 Beta 有什么关系？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
        query_mode=QueryMode.HYBRID,
    )
    mix = service.retrieve(
        "pptx 第2页表格里的 Alpha 和 Beta 有什么关系？",
        access_policy=AccessPolicy.default(),
        source_scope=["doc-a"],
        query_mode=QueryMode.MIX,
    )

    assert hybrid.diagnostics.mode_executor == "hybrid"
    assert mix.diagnostics.mode_executor == "mix"
    assert "vector" not in hybrid.diagnostics.branch_limits
    assert "full_text" not in hybrid.diagnostics.branch_limits
    assert mix.diagnostics.branch_limits["local"] > 0
    assert mix.diagnostics.branch_limits["global"] > 0
    assert mix.diagnostics.branch_limits["vector"] > 0
    assert mix.diagnostics.branch_limits["full_text"] > 0
