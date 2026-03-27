from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from math import sqrt

from pkp.repo.interfaces import ModelProviderRepo
from pkp.types.text import search_terms


class LexicalEmbeddingRepo(ModelProviderRepo):
    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension
        self.provider_name = "offline-lexical"
        self.embedding_model_name = f"lexical-{dimension}"
        self.chat_model_name = "offline-lexical-chat"
        self.is_chat_configured = False
        self.is_embed_configured = True
        self.is_rerank_configured = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def chat(self, prompt: str) -> str:
        return prompt

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        del query
        return list(range(len(candidates)))

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for term in search_terms(text):
            bucket = self._bucket(term)
            vector[bucket] += 1.0
        norm = sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def _bucket(self, term: str) -> int:
        digest = sha256(term.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self._dimension
