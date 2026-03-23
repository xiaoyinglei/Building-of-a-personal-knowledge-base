from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pkp.repo.interfaces import ModelProviderRepo
from pkp.repo.models.fallback_embedding_repo import FallbackEmbeddingRepo


class OpenAIProviderRepo(ModelProviderRepo):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_fallback: FallbackEmbeddingRepo | None = None,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._embedding_model = embedding_model
        self._fallback = embedding_fallback or FallbackEmbeddingRepo()
        self._client = client
        self._client_factory = client_factory

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client_instance().embeddings.create(
            model=self._embedding_model,
            input=list(texts),
        )
        return [list(item.embedding) for item in response.data]

    def chat(self, prompt: str) -> str:
        response = self._client_instance().responses.create(
            model=self._model,
            input=prompt,
        )
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        return str(output_text or "")

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        return self._fallback.rerank(query, candidates)

    def _client_instance(self) -> Any:
        if self._client is None:
            factory = self._client_factory or self._default_client_factory
            self._client = factory()
        return self._client

    def _default_client_factory(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
