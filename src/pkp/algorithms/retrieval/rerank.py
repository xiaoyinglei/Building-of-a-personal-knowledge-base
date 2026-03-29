from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pkp.algorithms.retrieval.contracts import Reranker
from pkp.query.context import CandidateLike


@dataclass(slots=True)
class UnifiedReranker:
    reranker: Reranker | None = None

    def rerank(self, query: str, candidates: list[CandidateLike]) -> list[CandidateLike]:
        if self.reranker is None:
            return list(candidates)
        return list(self.reranker(query, candidates))
