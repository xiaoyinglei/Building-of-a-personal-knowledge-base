from __future__ import annotations

from dataclasses import dataclass

from pkp.algorithms.retrieval.contracts import RetrieverFn
from pkp.algorithms.retrieval.mode_planner import RetrievalPlan
from pkp.query.context import CandidateLike


@dataclass(slots=True)
class BranchRetrieverRegistry:
    full_text_retriever: RetrieverFn
    vector_retriever: RetrieverFn
    section_retriever: RetrieverFn
    special_retriever: RetrieverFn
    metadata_retriever: RetrieverFn
    local_retriever: RetrieverFn
    global_retriever: RetrieverFn
    web_retriever: RetrieverFn

    def collect_internal(
        self,
        *,
        plan: RetrievalPlan,
        query: str,
        source_scope: list[str],
    ) -> dict[str, list[CandidateLike]]:
        results: dict[str, list[CandidateLike]] = {}
        for branch in plan.internal_branches:
            candidates = list(self._call(branch, query=query, source_scope=source_scope))
            results[branch] = candidates
        return results

    def collect_web(self, *, query: str, source_scope: list[str]) -> list[CandidateLike]:
        return list(self.web_retriever(query, source_scope))

    def _call(self, branch: str, *, query: str, source_scope: list[str]) -> Sequence[CandidateLike]:
        return {
            "full_text": self.full_text_retriever,
            "vector": self.vector_retriever,
            "section": self.section_retriever,
            "special": self.special_retriever,
            "metadata": self.metadata_retriever,
            "local": self.local_retriever,
            "global": self.global_retriever,
        }.get(branch, self.vector_retriever)(query, source_scope)
