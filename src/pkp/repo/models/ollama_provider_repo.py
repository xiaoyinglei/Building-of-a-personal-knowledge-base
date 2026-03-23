from __future__ import annotations

from collections.abc import Sequence

from pkp.repo.interfaces import ModelProviderRepo
from pkp.repo.models.fallback_embedding_repo import FallbackEmbeddingRepo


class OllamaProviderRepo(ModelProviderRepo):
    def __init__(self, *, embedding_fallback: FallbackEmbeddingRepo | None = None) -> None:
        self._fallback = embedding_fallback or FallbackEmbeddingRepo()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._fallback.embed(texts)

    def chat(self, prompt: str) -> str:
        return prompt

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        return self._fallback.rerank(query, candidates)
