from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from rag.utils._contracts import ModelProviderRepo


class FallbackEmbeddingRepo(ModelProviderRepo):
    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def chat(self, prompt: str) -> str:
        return prompt

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        del query, candidates
        raise RuntimeError("FallbackEmbeddingRepo does not implement rerank")

    @property
    def is_rerank_configured(self) -> bool:
        return False

    def _embed_text(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        for index in range(self._dimension):
            values.append(digest[index] / 255.0)
        return values
