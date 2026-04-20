from __future__ import annotations

from dataclasses import dataclass, field

from rag.retrieval.rerank_service import IndustrialRerankService


@dataclass(frozen=True)
class _FakeCandidate:
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str = "internal"
    source_id: str | None = None
    section_path: tuple[str, ...] = ()
    chunk_role: object | None = None
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    benchmark_doc_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class _CapturingReranker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(self, query: str, candidates: list[_FakeCandidate]) -> list[_FakeCandidate]:
        del query
        self.calls.append([candidate.chunk_id for candidate in candidates])
        return list(candidates)


def test_industrial_rerank_service_applies_hard_cap_before_model_rerank() -> None:
    reranker = _CapturingReranker()
    service = IndustrialRerankService(max_model_candidates=50)

    result = service.rank(
        query="Alpha",
        fused_candidates=[
            _FakeCandidate(
                chunk_id=f"chunk-{index}",
                doc_id="doc-a",
                text=f"candidate {index}",
                citation_anchor=f"#c{index}",
                score=1.0 - index * 0.001,
                rank=index + 1,
                metadata={"section_id": index},
            )
            for index in range(80)
        ],
        reranker=reranker,
        rerank_required=True,
        rerank_pool_k=None,
        allow_asset_fallback=False,
    )

    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 50
    assert result.diagnostics.output_count == 50
    assert len(result.ranked_candidates) == 50

