from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.query.context import CandidateLike


class RetrieverFn(Protocol):
    def __call__(self, query: str, source_scope: list[str]) -> Sequence[CandidateLike]: ...


class GraphExpander(Protocol):
    def __call__(
        self,
        query: str,
        source_scope: list[str],
        evidence: list[CandidateLike],
    ) -> Sequence[CandidateLike]: ...


class Reranker(Protocol):
    def __call__(self, query: str, candidates: list[CandidateLike]) -> Sequence[CandidateLike]: ...
