from __future__ import annotations

from collections.abc import Callable, Sequence

from rag.utils._contracts import SearchResult, WebSearchRepo


class DeterministicWebSearchRepo(WebSearchRepo):
    def __init__(self, index: dict[str, list[SearchResult]] | None = None) -> None:
        self._index = index or {}

    def register(self, query: str, results: list[SearchResult]) -> None:
        self._index[query] = list(results)

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        return list(self._index.get(query, []))[:limit]


class CallableWebSearchRepo(WebSearchRepo):
    def __init__(self, search_fn: Callable[[str, int], Sequence[SearchResult]]) -> None:
        self._search_fn = search_fn

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        return list(self._search_fn(query, limit))
