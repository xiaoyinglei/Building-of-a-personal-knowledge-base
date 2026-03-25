from dataclasses import dataclass

from pkp.service.rerank_service import HeuristicRerankService


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    text: str
    score: float
    section_path: tuple[str, ...] = ()


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

    reranked = service.rerank("这个项目做什么？", candidates)

    assert [candidate.chunk_id for candidate in reranked][:2] == ["chunk-desc", "chunk-cli"]
