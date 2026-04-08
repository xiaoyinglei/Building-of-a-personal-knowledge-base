from __future__ import annotations

import importlib
from collections.abc import Callable, MutableMapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import httpx

from rag.schema.runtime import ModelProviderRepo

_FAST_TOKENIZER_PADDING_WARNING = "Asking-to-pad-a-fast-tokenizer"


def suppress_backend_fast_tokenizer_padding_warning(backend: object) -> object:
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        return backend
    if not _looks_like_fast_tokenizer(tokenizer):
        return backend

    deprecation_warnings = getattr(tokenizer, "deprecation_warnings", None)
    if isinstance(deprecation_warnings, MutableMapping):
        deprecation_warnings[_FAST_TOKENIZER_PADDING_WARNING] = True
    return backend


def _looks_like_fast_tokenizer(tokenizer: object) -> bool:
    if bool(getattr(tokenizer, "is_fast", False)):
        return True
    return tokenizer.__class__.__name__.endswith("Fast")


def expand_optional_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    if isinstance(raw, Path):
        return raw.expanduser()
    normalized = raw.strip()
    if not normalized:
        return None
    return Path(normalized).expanduser()


def resolve_local_model_reference(model_name: str, model_path: str | Path | None) -> str:
    expanded = expand_optional_path(model_path)
    if expanded is None:
        return model_name
    return str(resolve_huggingface_snapshot_path(expanded))


def resolve_huggingface_snapshot_path(model_root: str | Path) -> Path:
    path = Path(model_root).expanduser()
    if _looks_like_model_dir(path):
        return path

    main_ref = path / "refs" / "main"
    if main_ref.exists():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = path / "snapshots" / revision
        if _looks_like_model_dir(snapshot):
            return snapshot

    snapshots_root = path / "snapshots"
    if snapshots_root.exists():
        candidates = sorted(
            candidate
            for candidate in snapshots_root.iterdir()
            if candidate.is_dir() and _looks_like_model_dir(candidate)
        )
        if len(candidates) == 1:
            return candidates[0]

    return path


def _looks_like_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "tokenizer_config.json").exists()


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
        return [digest[index] / 255.0 for index in range(self._dimension)]


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
        except Exception as exc:  # pragma: no cover
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
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if not self._should_try_chat_fallback(mode, exc):
                    break
        raise RuntimeError(f"OpenAI chat request failed: {last_error}") from last_error

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        del query, candidates
        raise RuntimeError(
            "OpenAIProviderRepo does not provide rerank. "
            "Configure a dedicated reranker via RAG_RERANK_MODEL or a provider with is_rerank_configured=true."
        )

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
        response = self._client_instance().responses.create(model=self._model, input=prompt)
        output_text = getattr(response, "output_text", None)
        return output_text if isinstance(output_text, str) else str(output_text or "")

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
        return "chat_completions" if "generativelanguage.googleapis.com" in base_url.lower() else "responses"

    @staticmethod
    def _should_try_chat_fallback(mode: str, exc: Exception) -> bool:
        message = str(exc).lower()
        fallback_markers = ("404", "not found", "unsupported", "attributeerror")
        if mode in {"responses", "chat_completions"}:
            return any(marker in message for marker in fallback_markers) or isinstance(exc, AttributeError)
        return False

    def _default_client_factory(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self._base_url)


class OllamaProviderRepo(ModelProviderRepo):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        chat_model: str = "qwen3.5:9b",
        embedding_model: str | None = "qwen3-embedding:8b",
        embedding_fallback: FallbackEmbeddingRepo | None = None,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._fallback = embedding_fallback or FallbackEmbeddingRepo()
        self._http_client = http_client
        self._timeout_seconds = timeout_seconds

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.is_embed_configured or not self._embedding_model:
            raise RuntimeError("Ollama embedding model is not configured")
        try:
            response = self._client().post(
                f"{self._base_url}/api/embed",
                json={"model": self._embedding_model, "input": list(texts)},
            )
            response.raise_for_status()
            payload = response.json()
            return [list(vector) for vector in payload["embeddings"]]
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

    def chat(self, prompt: str) -> str:
        try:
            response = self._client().post(
                f"{self._base_url}/api/chat",
                json={"model": self._chat_model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["message"]["content"])
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama chat request failed: {exc}") from exc

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        del query, candidates
        raise RuntimeError(
            "OllamaProviderRepo does not provide rerank. "
            "Configure a dedicated reranker via RAG_RERANK_MODEL or a provider with is_rerank_configured=true."
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def chat_model_name(self) -> str:
        return self._chat_model

    @property
    def embedding_model_name(self) -> str | None:
        return self._embedding_model

    @property
    def is_chat_configured(self) -> bool:
        return bool(self._base_url and self._chat_model)

    @property
    def is_embed_configured(self) -> bool:
        return bool(self._base_url and self._embedding_model)

    @property
    def is_rerank_configured(self) -> bool:
        return False

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=httpx.Timeout(self._timeout_seconds))
        return self._http_client


class LocalBgeProviderRepo:
    def __init__(
        self,
        *,
        embedding_model: str = "BAAI/bge-m3",
        embedding_model_path: str | None = None,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        rerank_model_path: str | None = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        batch_size: int = 8,
        max_length: int = 1024,
        rerank_batch_size: int = 8,
        rerank_max_length: int = 512,
    ) -> None:
        self._embedding_model = embedding_model
        self._embedding_model_ref = resolve_local_model_reference(embedding_model, embedding_model_path)
        self._rerank_model = rerank_model
        self._rerank_model_ref = resolve_local_model_reference(rerank_model, rerank_model_path)
        self._normalize_embeddings = normalize_embeddings
        self._use_fp16 = use_fp16
        self._batch_size = batch_size
        self._max_length = max_length
        self._rerank_batch_size = rerank_batch_size
        self._rerank_max_length = rerank_max_length
        self._embedding_backend: object | None = None
        self._rerank_backend: object | None = None

    @property
    def provider_name(self) -> str:
        return "local-bge"

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model

    @property
    def rerank_model_name(self) -> str:
        return self._rerank_model

    @property
    def is_embed_configured(self) -> bool:
        return bool(self._embedding_model_ref)

    @property
    def is_rerank_configured(self) -> bool:
        return bool(self._rerank_model_ref)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        backend = self._embedding_backend or self._load_embedding_backend()
        self._embedding_backend = backend
        encoder = cast(Any, backend)
        payload = encoder.encode(
            list(texts),
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vectors = payload.get("dense_vecs") if isinstance(payload, dict) else payload
        if dense_vectors is None:
            raise RuntimeError("Local BGE embedding backend returned no dense vectors")
        return [list(vector) for vector in dense_vectors]

    def rerank(self, query: str, candidates: Sequence[object]) -> list[int]:
        if not candidates:
            return []
        backend = self._rerank_backend or self._load_rerank_backend()
        self._rerank_backend = backend
        reranker = cast(Any, backend)
        candidate_texts = [self._candidate_text(candidate) for candidate in candidates]
        pairs = [[query, candidate_text] for candidate_text in candidate_texts]
        scores = reranker.compute_score(
            pairs,
            batch_size=self._rerank_batch_size,
            max_length=self._rerank_max_length,
        )
        indexed = list(enumerate(float(score) for score in scores))
        indexed.sort(key=lambda item: item[1], reverse=True)
        return [index for index, _score in indexed]

    def _load_embedding_backend(self) -> object:
        try:
            module = importlib.import_module("FlagEmbedding")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("FlagEmbedding is required for local BGE embeddings") from exc
        encoder_cls = getattr(module, "BGEM3FlagModel", None)
        if encoder_cls is None:
            raise RuntimeError("FlagEmbedding BGEM3FlagModel is unavailable")
        try:
            backend = cast(
                object,
                encoder_cls(
                    self._embedding_model_ref,
                    normalize_embeddings=self._normalize_embeddings,
                    use_fp16=self._use_fp16,
                    batch_size=self._batch_size,
                    query_max_length=self._max_length,
                    passage_max_length=self._max_length,
                    return_sparse=False,
                    return_colbert_vecs=False,
                ),
            )
            return suppress_backend_fast_tokenizer_padding_warning(backend)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Local BGE embedding load failed: {exc}") from exc

    def _load_rerank_backend(self) -> object:
        try:
            module = importlib.import_module("FlagEmbedding")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("FlagEmbedding is required for local BGE rerank") from exc
        reranker_cls = getattr(module, "FlagReranker", None)
        if reranker_cls is None:
            raise RuntimeError("FlagEmbedding FlagReranker is unavailable")
        try:
            backend = cast(
                object,
                reranker_cls(
                    self._rerank_model_ref,
                    use_fp16=self._use_fp16,
                    batch_size=self._rerank_batch_size,
                    max_length=self._rerank_max_length,
                    normalize=False,
                ),
            )
            return suppress_backend_fast_tokenizer_padding_warning(backend)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Local BGE rerank load failed: {exc}") from exc

    @staticmethod
    def _candidate_text(candidate: object) -> str:
        if isinstance(candidate, str):
            return candidate
        text = getattr(candidate, "text", None)
        if isinstance(text, str):
            return text
        return str(candidate)


suppress_fast_tokenizer_padding_warning = suppress_backend_fast_tokenizer_padding_warning

__all__ = [
    "FallbackEmbeddingRepo",
    "LocalBgeProviderRepo",
    "OllamaProviderRepo",
    "OpenAIProviderRepo",
    "expand_optional_path",
    "resolve_huggingface_snapshot_path",
    "resolve_local_model_reference",
    "suppress_backend_fast_tokenizer_padding_warning",
    "suppress_fast_tokenizer_padding_warning",
]
