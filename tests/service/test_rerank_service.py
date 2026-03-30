from dataclasses import dataclass
from typing import Any, cast

from pkp.llm.rerank import HeuristicRerankService


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    text: str
    score: float
    section_path: tuple[str, ...] = ()
    special_chunk_type: str | None = None
    chunk_role: str | None = None
    metadata: dict[str, str] | None = None


def test_rerank_service_prefers_explanatory_text_over_command_snippets_for_natural_language_queries() -> None:
    service = HeuristicRerankService()
    candidates = [
        Candidate(
            chunk_id="chunk-cli",
            text='uv run pkp query --mode fast --query "这个项目做什么？"',
            score=1.2,
            section_path=("查询",),
        ),
        Candidate(
            chunk_id="chunk-desc",
            text="一个以可靠性为优先的个人知识平台，用来完成资料接入、索引构建、检索问答、深度研究和知识沉淀。",
            score=1.0,
            section_path=("个人知识平台",),
        ),
    ]

    reranked = service.rerank("这个项目做什么？", cast(Any, candidates))

    assert [candidate.chunk_id for candidate in reranked][:2] == ["chunk-desc", "chunk-cli"]


def test_rerank_service_penalizes_readme_cli_examples_for_definition_queries() -> None:
    service = HeuristicRerankService()
    candidates = [
        Candidate(
            chunk_id="chunk-cli",
            text='uv run python -m pkp.interfaces._ui.cli query --mode fast --query "这个项目做什么？" '
            'uv run python -m pkp.interfaces._ui.cli query --mode deep --query "比较 Fast Path 和 Deep Path"',
            score=1.0,
            section_path=("查询",),
        ),
        Candidate(
            chunk_id="chunk-desc",
            text="一个以可靠性为优先的个人知识平台，用来完成资料接入、索引构建、检索问答、深度研究和知识沉淀。",
            score=0.9,
            section_path=("个人知识平台",),
        ),
    ]

    reranked = service.rerank("这个项目做什么？", cast(Any, candidates))

    assert [candidate.chunk_id for candidate in reranked][:2] == ["chunk-desc", "chunk-cli"]


def test_rerank_service_prefers_table_special_chunks_for_table_queries() -> None:
    service = HeuristicRerankService()
    candidates = [
        Candidate(
            chunk_id="chunk-figure",
            text="图 2 展示了月返流程示意图。",
            score=1.1,
            section_path=("专项工作",),
            special_chunk_type="figure",
            chunk_role="special",
        ),
        Candidate(
            chunk_id="chunk-table",
            text="| 指标 | 数值 |\n| --- | --- |\n| 告警 | 7 |",
            score=0.95,
            section_path=("专项工作", "统计表"),
            special_chunk_type="table",
            chunk_role="special",
        ),
    ]

    reranked = service.rerank("表格里告警数值是多少？", cast(Any, candidates))

    assert [candidate.chunk_id for candidate in reranked][:2] == ["chunk-table", "chunk-figure"]


def test_rerank_service_prefers_exact_structure_path_matches_for_heading_queries() -> None:
    service = HeuristicRerankService()
    candidates = [
        Candidate(
            chunk_id="chunk-general",
            text="系统由多个模块组成。",
            score=1.05,
            section_path=("项目介绍",),
        ),
        Candidate(
            chunk_id="chunk-arch",
            text="系统架构分为接入层、检索层、生成层。",
            score=0.98,
            section_path=("系统架构",),
        ),
    ]

    reranked = service.rerank("系统架构分为哪几层？", cast(Any, candidates))

    assert [candidate.chunk_id for candidate in reranked][:2] == ["chunk-arch", "chunk-general"]
