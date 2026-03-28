from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from pkp.service.evidence_service import CandidateLike


RetrieverFn = Callable[[str, list[str]], Sequence[CandidateLike]]
GraphExpander = Callable[[str, list[str], list[CandidateLike]], Sequence[CandidateLike]]


class Reranker(Protocol):
    def __call__(self, query: str, candidates: list[CandidateLike]) -> Sequence[CandidateLike]: ...
