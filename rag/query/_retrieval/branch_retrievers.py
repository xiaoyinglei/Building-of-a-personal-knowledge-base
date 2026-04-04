from __future__ import annotations

from dataclasses import dataclass

from rag.query._retrieval.contracts import RetrieverFn
from rag.query.context import CandidateLike
from rag.schema._types.query import QueryUnderstanding


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

    def collect_web(
        self,
        *,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
    ) -> list[CandidateLike]:
        return list(self.web_retriever(query, source_scope, query_understanding))

    def get(self, branch: str) -> RetrieverFn:
        return {
            "full_text": self.full_text_retriever,
            "vector": self.vector_retriever,
            "section": self.section_retriever,
            "special": self.special_retriever,
            "metadata": self.metadata_retriever,
            "local": self.local_retriever,
            "global": self.global_retriever,
        }.get(branch, self.vector_retriever)
