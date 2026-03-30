from __future__ import annotations

from dataclasses import dataclass

from rag.query._retrieval.contracts import Reranker
from rag.query.context import CandidateLike


@dataclass(slots=True)
class UnifiedReranker:
    reranker: Reranker | None = None

    def rerank(self, query: str, candidates: list[CandidateLike]) -> list[CandidateLike]:
        if self.reranker is None:
            return list(candidates)
        return list(self.reranker(query, candidates))
