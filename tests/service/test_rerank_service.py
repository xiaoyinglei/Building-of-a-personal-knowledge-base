from dataclasses import dataclass
from typing import Any, cast

import pytest

from rag.providers.rerank import ModelBackedRerankService


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    text: str
    score: float
    section_path: tuple[str, ...] = ()
    special_chunk_type: str | None = None
    chunk_role: str | None = None
    metadata: dict[str, str] | None = None


class _FakeModelRerankProvider:
    provider_name = "fake-model-rerank"
    rerank_model_name = "fake-cross-encoder"
    is_rerank_configured = True

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        lowered = query.lower()
        preferred: tuple[str, ...]
        if "表格" in query or "数值" in query:
            preferred = ("table", "指标")
        elif "架构" in query or "哪几层" in query:
            preferred = ("架构", "system architecture")
        else:
            preferred = ("个人知识平台", "可靠性", "supports beta service")
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: (
                0 if any(token.lower() in candidates[index].lower() for token in preferred) else 1,
                -len(candidates[index]),
            ),
        )
        if "项目做什么" in lowered:
            return ranked
        return ranked


def test_model_backed_rerank_service_uses_provider_ranking() -> None:
    service = ModelBackedRerankService(provider=_FakeModelRerankProvider())
    candidates = [
        Candidate(
            chunk_id="chunk-cli",
            text='uv run rag query --mode fast --query "这个项目做什么？"',
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


def test_model_backed_rerank_service_ranks_definition_chunks_ahead_of_cli_examples() -> None:
    service = ModelBackedRerankService(provider=_FakeModelRerankProvider())
    candidates = [
        Candidate(
            chunk_id="chunk-cli",
            text='uv run python -m rag.cli query --mode fast --query "这个项目做什么？" '
            'uv run python -m rag.cli query --mode deep --query "比较 Fast Path 和 Deep Path"',
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


def test_model_backed_rerank_service_prefers_table_chunks_when_provider_scores_them_higher() -> None:
    service = ModelBackedRerankService(provider=_FakeModelRerankProvider())
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


def test_model_backed_rerank_service_prefers_structure_chunks_when_provider_scores_them_higher() -> None:
    service = ModelBackedRerankService(provider=_FakeModelRerankProvider())
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


def test_model_backed_rerank_service_requires_real_rerank_backend() -> None:
    service = ModelBackedRerankService()

    with pytest.raises(RuntimeError, match="No model-backed reranker is configured"):
        service.rerank(
            "这个项目做什么？",
            cast(
                Any,
                [
                    Candidate(
                        chunk_id="chunk-desc",
                        text="一个以可靠性为优先的个人知识平台。",
                        score=1.0,
                    )
                ],
            ),
        )
