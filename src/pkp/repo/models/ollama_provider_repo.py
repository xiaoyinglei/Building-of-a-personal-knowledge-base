from __future__ import annotations

from collections.abc import Sequence

import httpx

from pkp.repo.interfaces import ModelProviderRepo
from pkp.repo.models.fallback_embedding_repo import FallbackEmbeddingRepo


class OllamaProviderRepo(ModelProviderRepo):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        chat_model: str = "llama3.1:8b",
        embedding_model: str = "nomic-embed-text",
        embedding_fallback: FallbackEmbeddingRepo | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._fallback = embedding_fallback or FallbackEmbeddingRepo()
        self._http_client = http_client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client().post(
            f"{self._base_url}/api/embed",
            json={
                "model": self._embedding_model,
                "input": list(texts),
            },
        )
        response.raise_for_status()
        payload = response.json()
        return [list(vector) for vector in payload["embeddings"]]

    def chat(self, prompt: str) -> str:
        response = self._client().post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["message"]["content"])

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        return self._fallback.rerank(query, candidates)

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client()
        return self._http_client
