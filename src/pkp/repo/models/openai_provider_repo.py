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
        self._chat_api_mode = self._initial_chat_api_mode(base_url)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = self._client_instance().embeddings.create(
                model=self._embedding_model,
                input=list(texts),
            )
            return [list(item.embedding) for item in response.data]
        except Exception as exc:  # pragma: no cover - provider-specific exception tree
            raise RuntimeError(f"OpenAI embedding request failed: {exc}") from exc

    def chat(self, prompt: str) -> str:
        last_error: Exception | None = None
        for mode in self._chat_api_order():
            try:
                if mode == "responses":
                    text = self._chat_via_responses(prompt)
                else:
                    text = self._chat_via_chat_completions(prompt)
                self._chat_api_mode = mode
                return text
            except Exception as exc:  # pragma: no cover - provider-specific exception tree
                last_error = exc
                if not self._should_try_chat_fallback(mode, exc):
                    break
                continue
        raise RuntimeError(f"OpenAI chat request failed: {last_error}") from last_error

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        return self._fallback.rerank(query, candidates)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def chat_model_name(self) -> str:
        return self._model

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model

    @property
    def is_chat_configured(self) -> bool:
        return bool(self._api_key and self._api_key.strip() and self._model)

    @property
    def is_embed_configured(self) -> bool:
        return bool(self._api_key and self._api_key.strip() and self._embedding_model)

    @property
    def is_rerank_configured(self) -> bool:
        return False

    def _client_instance(self) -> Any:
        if self._client is None:
            factory = self._client_factory or self._default_client_factory
            self._client = factory()
        return self._client

    def _chat_api_order(self) -> tuple[str, ...]:
        if self._chat_api_mode == "chat_completions":
            return ("chat_completions", "responses")
        return ("responses", "chat_completions")

    def _chat_via_responses(self, prompt: str) -> str:
        response = self._client_instance().responses.create(
            model=self._model,
            input=prompt,
        )
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        return str(output_text or "")

    def _chat_via_chat_completions(self, prompt: str) -> str:
        response = self._client_instance().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return "\n".join(part for part in parts if part)
        return str(content or "")

    @staticmethod
    def _initial_chat_api_mode(base_url: str) -> str:
        lowered = base_url.lower()
        if "generativelanguage.googleapis.com" in lowered:
            return "chat_completions"
        return "responses"

    @staticmethod
    def _should_try_chat_fallback(mode: str, exc: Exception) -> bool:
        message = str(exc).lower()
        fallback_markers = (
            "404",
            "not found",
            "unsupported",
            "attributeerror",
        )
        if mode == "responses":
            return any(marker in message for marker in fallback_markers) or isinstance(exc, AttributeError)
        if mode == "chat_completions":
            return any(marker in message for marker in fallback_markers) or isinstance(exc, AttributeError)
        return False

    def _default_client_factory(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
