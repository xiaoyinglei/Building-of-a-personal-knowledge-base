from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.query.context import CandidateLike
from rag.schema._types.query import QueryUnderstanding


class RetrieverFn(Protocol):
    def __call__(
        self,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
    ) -> Sequence[CandidateLike]: ...


class GraphExpander(Protocol):
    def __call__(
        self,
        query: str,
        source_scope: list[str],
        evidence: list[CandidateLike],
    ) -> Sequence[CandidateLike]: ...


class Reranker(Protocol):
    def __call__(self, query: str, candidates: list[CandidateLike]) -> Sequence[CandidateLike]: ...
