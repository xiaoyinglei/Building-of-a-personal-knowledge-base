from __future__ import annotations

import importlib
import io
import logging
import re
import time
from collections.abc import Callable, MutableMapping, Sequence
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import httpx

from rag.schema.runtime import ModelProviderRepo

_FAST_TOKENIZER_PADDING_WARNING = "Asking-to-pad-a-fast-tokenizer"
_LOGGER = logging.getLogger(__name__)
_DECODER_ONLY_RERANKER_MARKERS = ("qwen", "gemma", "minicpm", "llm-reranker")


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


def _patch_transformers_import_utils_for_flagembedding() -> None:
    try:
        import transformers.utils.import_utils as import_utils
    except Exception:
        return
    if hasattr(import_utils, "is_torch_fx_available"):
        return
    import_utils.is_torch_fx_available = import_utils.is_torch_available  # type: ignore[attr-defined]


def _prepare_for_model_fallback(
    tokenizer: object,
    ids: Sequence[int],
    pair_ids: Sequence[int] | None = None,
    *,
    add_special_tokens: bool = True,
    truncation: str | bool | None = None,
    max_length: int | None = None,
    padding: bool | str = False,
    return_attention_mask: bool = True,
    return_token_type_ids: bool | None = None,
    **_: object,
) -> dict[str, list[int]]:
    del padding
    first = list(ids)
    second = list(pair_ids) if pair_ids is not None else None

    def special_token_count(has_pair: bool) -> int:
        counter = getattr(tokenizer, "num_special_tokens_to_add", None)
        if callable(counter) and add_special_tokens:
            try:
                return int(counter(pair=has_pair))
            except Exception:
                return 0
        return 0

    def apply_truncation() -> tuple[list[int], list[int] | None]:
        local_first = list(first)
        local_second = list(second) if second is not None else None
        if max_length is None:
            return local_first, local_second

        budget = max(0, int(max_length) - special_token_count(local_second is not None))
        if local_second is None:
            return local_first[:budget], None

        strategy = truncation
        if strategy in (None, False):
            if len(local_first) + len(local_second) <= budget:
                return local_first, local_second
            strategy = "only_second"

        if strategy == "only_first":
            keep = max(0, budget - len(local_second))
            return local_first[:keep], local_second
        if strategy == "only_second":
            keep = max(0, budget - len(local_first))
            return local_first, local_second[:keep]

        # Fall back to longest-first style trimming.
        while len(local_first) + len(local_second) > budget and (local_first or local_second):
            if len(local_second) >= len(local_first) and local_second:
                local_second.pop()
                continue
            if local_first:
                local_first.pop()
                continue
            break
        return local_first, local_second

    def build_input_ids(local_first: list[int], local_second: list[int] | None) -> list[int]:
        if not add_special_tokens:
            return local_first + (local_second or [])

        cls_id = getattr(tokenizer, "cls_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        class_name = tokenizer.__class__.__name__.lower()

        if cls_id is not None or sep_id is not None:
            prefix = [int(cls_id)] if cls_id is not None else []
            sep = [int(sep_id)] if sep_id is not None else ([int(eos_id)] if eos_id is not None else [])
            if local_second is None:
                return prefix + local_first + sep
            middle = sep + sep if "roberta" in class_name and sep else sep
            return prefix + local_first + middle + local_second + sep

        prefix = [int(bos_id)] if bos_id is not None else []
        suffix = [int(eos_id)] if eos_id is not None else []
        if local_second is None:
            return prefix + local_first + suffix
        return prefix + local_first + suffix + local_second + suffix

    truncated_first, truncated_second = apply_truncation()
    input_ids = build_input_ids(truncated_first, truncated_second)
    encoded: dict[str, list[int]] = {"input_ids": input_ids}
    if return_attention_mask:
        encoded["attention_mask"] = [1] * len(input_ids)
    if return_token_type_ids:
        if truncated_second is None:
            encoded["token_type_ids"] = [0] * len(input_ids)
        else:
            encoded["token_type_ids"] = [0] * len(input_ids)
    return encoded


def _patch_transformers_tokenizer_prepare_for_model() -> None:
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
    except Exception:
        return
    if hasattr(TokenizersBackend, "prepare_for_model"):
        return

    def prepare_for_model(
        self: object,
        ids: Sequence[int],
        pair_ids: Sequence[int] | None = None,
        **kwargs: object,
    ) -> dict[str, list[int]]:
        return _prepare_for_model_fallback(self, ids, pair_ids, **kwargs)

    TokenizersBackend.prepare_for_model = prepare_for_model  # type: ignore[attr-defined]


def _load_flagembedding_module() -> object:
    _patch_transformers_import_utils_for_flagembedding()
    _patch_transformers_tokenizer_prepare_for_model()
    return cast(object, importlib.import_module("FlagEmbedding"))


def _infer_flagembedding_reranker_model_class(model_ref: str) -> str:
    normalized = model_ref.strip().lower()
    if any(marker in normalized for marker in _DECODER_ONLY_RERANKER_MARKERS):
        return "decoder-only-base"
    return "encoder-only-base"


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
        batch_size: int = 8,
        log_embedding_calls: bool = False,
        show_backend_progress: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._fallback = embedding_fallback or FallbackEmbeddingRepo()
        self._http_client = http_client
        self._timeout_seconds = timeout_seconds
        self._batch_size = batch_size
        self._preferred_device: str | None = None
        self._reported_device = "ollama-managed"
        self._log_embedding_calls = log_embedding_calls
        self._show_backend_progress = show_backend_progress
        self._embedding_call_count = 0
        self._embedded_text_count = 0
        self._embedding_total_duration_ms = 0.0
        self._last_embedding_duration_ms = 0.0
        self._last_embedding_request_size = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.is_embed_configured or not self._embedding_model:
            raise RuntimeError("Ollama embedding model is not configured")
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            started = time.perf_counter()
            try:
                response = self._client().post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._embedding_model, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc
            duration_ms = (time.perf_counter() - started) * 1000.0
            batch_vectors = payload.get("embeddings")
            if not isinstance(batch_vectors, list):
                raise RuntimeError("Ollama embedding response did not contain embeddings")
            vectors.extend([list(vector) for vector in batch_vectors])
            self._embedding_call_count += 1
            self._embedded_text_count += len(batch)
            self._embedding_total_duration_ms += duration_ms
            self._last_embedding_duration_ms = duration_ms
            self._last_embedding_request_size = len(batch)
            if self._log_embedding_calls:
                _LOGGER.info(
                    "embedding_call provider=%s model=%s device=%s "
                    "encode_batch_size=%s request_size=%s duration_ms=%.3f",
                    self.provider_name,
                    self.embedding_model_name,
                    self.embedding_device,
                    self.embedding_batch_size,
                    len(batch),
                    duration_ms,
                )
        return vectors

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

    @property
    def embedding_batch_size(self) -> int:
        return self._batch_size

    @property
    def embedding_device(self) -> str:
        return self._reported_device

    def set_embedding_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        self._batch_size = batch_size

    def set_device_preference(self, device: str | None) -> None:
        self._preferred_device = device.strip() if isinstance(device, str) and device.strip() else None

    def set_embedding_call_logging(self, enabled: bool) -> None:
        self._log_embedding_calls = enabled

    def set_backend_progress_enabled(self, enabled: bool) -> None:
        self._show_backend_progress = enabled

    def reset_embedding_stats(self) -> None:
        self._embedding_call_count = 0
        self._embedded_text_count = 0
        self._embedding_total_duration_ms = 0.0
        self._last_embedding_duration_ms = 0.0
        self._last_embedding_request_size = 0

    def embedding_runtime_info(self) -> dict[str, object]:
        return {
            "provider": self.provider_name,
            "model_name": self.embedding_model_name,
            "device": self.embedding_device,
            "encode_batch_size": self.embedding_batch_size,
            "preferred_device": self._preferred_device,
        }

    def embedding_stats(self) -> dict[str, object]:
        return {
            **self.embedding_runtime_info(),
            "call_count": self._embedding_call_count,
            "text_count": self._embedded_text_count,
            "total_duration_ms": round(self._embedding_total_duration_ms, 3),
            "last_duration_ms": round(self._last_embedding_duration_ms, 3),
            "last_request_size": self._last_embedding_request_size,
        }

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=httpx.Timeout(self._timeout_seconds), trust_env=False)
        return self._http_client


class LocalHfChatProviderRepo(ModelProviderRepo):
    def __init__(
        self,
        *,
        chat_model: str | None = None,
        chat_model_path: str | Path | None = None,
        backend: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self._chat_model = chat_model.strip() if isinstance(chat_model, str) else ""
        self._chat_model_path = None if chat_model_path is None else str(expand_optional_path(chat_model_path) or "")
        self._backend_preference = (backend or "auto").strip().lower() or "auto"
        self._preferred_device = None if device is None else device.strip() or None
        self._max_new_tokens = max(1, int(max_new_tokens))
        self._temperature = max(0.0, float(temperature))
        self._top_p = min(max(float(top_p), 0.0), 1.0)
        self._resolved_model_ref = self._resolve_chat_model_reference(
            chat_model=self._chat_model,
            chat_model_path=chat_model_path,
        )
        self._backend_name: str | None = None
        self._backend_impl: object | None = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        raise RuntimeError("LocalHfChatProviderRepo does not provide embeddings")

    def chat(self, prompt: str) -> str:
        backend = self._backend()
        chat = getattr(backend, "chat", None)
        if not callable(chat):
            raise RuntimeError("Local HF chat backend is not available")
        return _normalize_chat_text(str(chat(prompt)))

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        del query, candidates
        raise RuntimeError("LocalHfChatProviderRepo does not provide rerank")

    @property
    def provider_name(self) -> str:
        return "local-hf"

    @property
    def chat_model_name(self) -> str:
        return self._chat_model or self._resolved_model_ref

    @property
    def is_chat_configured(self) -> bool:
        return bool(self._resolved_model_ref)

    @property
    def is_embed_configured(self) -> bool:
        return False

    @property
    def is_rerank_configured(self) -> bool:
        return False

    @property
    def backend_name(self) -> str | None:
        return self._backend_name

    def _backend(self) -> object:
        if self._backend_impl is not None:
            return self._backend_impl
        errors: list[str] = []
        for backend_name in self._backend_candidates():
            try:
                backend = self._load_backend(backend_name)
            except Exception as exc:
                errors.append(f"{backend_name}: {exc}")
                continue
            self._backend_name = backend_name
            self._backend_impl = backend
            return backend
        detail = "; ".join(errors) if errors else "no backend candidates were available"
        raise RuntimeError(f"Failed to initialize local HF chat backend for {self._resolved_model_ref}: {detail}")

    def _backend_candidates(self) -> tuple[str, ...]:
        if self._backend_preference in {"mlx", "transformers"}:
            return (self._backend_preference,)
        likely_mlx = any(
            marker in value.lower()
            for marker in ("mlx-community", "/mlx", "models--mlx-community--", "mlx_community")
            for value in filter(None, (self._chat_model, self._chat_model_path, self._resolved_model_ref))
        )
        return ("mlx", "transformers") if likely_mlx else ("transformers", "mlx")

    def _load_backend(self, backend_name: str) -> object:
        if backend_name == "mlx":
            return self._load_mlx_backend()
        if backend_name == "transformers":
            return self._load_transformers_backend()
        raise RuntimeError(f"Unsupported local HF chat backend: {backend_name}")

    def _load_mlx_backend(self) -> object:
        from mlx_lm import generate, load

        model, tokenizer = load(self._resolved_model_ref)

        class _MlxBackend:
            def __init__(
                self,
                *,
                model: object,
                tokenizer: object,
                max_new_tokens: int,
                temperature: float,
                top_p: float,
            ) -> None:
                self._model = model
                self._tokenizer = tokenizer
                self._max_new_tokens = max_new_tokens
                self._temperature = temperature
                self._top_p = top_p

            def chat(self, prompt: str) -> str:
                rendered = _render_chat_prompt(self._tokenizer, prompt)
                kwargs: dict[str, object] = {
                    "verbose": False,
                    "max_tokens": self._max_new_tokens,
                }
                if self._temperature > 0.0:
                    kwargs["temp"] = self._temperature
                    kwargs["top_p"] = self._top_p
                return str(generate(self._model, self._tokenizer, rendered, **kwargs)).strip()

        return _MlxBackend(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
        )

    def _load_transformers_backend(self) -> object:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_transformers_device(self._preferred_device)
        dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self._resolved_model_ref, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self._resolved_model_ref,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()

        class _TransformersBackend:
            def __init__(
                self,
                *,
                model: object,
                tokenizer: object,
                device: str,
                max_new_tokens: int,
                temperature: float,
                top_p: float,
            ) -> None:
                self._model = model
                self._tokenizer = tokenizer
                self._device = device
                self._max_new_tokens = max_new_tokens
                self._temperature = temperature
                self._top_p = top_p

            def chat(self, prompt: str) -> str:
                import torch

                rendered = _render_chat_prompt(self._tokenizer, prompt)
                encoded = self._tokenizer(rendered, return_tensors="pt")
                encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
                generation_kwargs: dict[str, object] = {
                    "max_new_tokens": self._max_new_tokens,
                    "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                }
                if self._temperature > 0.0:
                    generation_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": self._temperature,
                            "top_p": self._top_p,
                        }
                    )
                else:
                    generation_kwargs["do_sample"] = False
                with torch.inference_mode():
                    outputs = self._model.generate(**encoded, **generation_kwargs)
                prompt_tokens = int(encoded["input_ids"].shape[-1])
                generated = outputs[0][prompt_tokens:]
                return str(self._tokenizer.decode(generated, skip_special_tokens=True)).strip()

        return _TransformersBackend(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
        )

    @staticmethod
    def _resolve_chat_model_reference(*, chat_model: str, chat_model_path: str | Path | None) -> str:
        explicit_path = expand_optional_path(chat_model_path)
        if explicit_path is not None:
            return str(resolve_huggingface_snapshot_path(explicit_path))
        return chat_model.strip()

    @staticmethod
    def _resolve_transformers_device(device: str | None) -> str:
        if isinstance(device, str) and device.strip():
            normalized = device.strip().lower()
            if normalized != "auto":
                return normalized
        import torch

        if getattr(torch.cuda, "is_available", None) and torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"


def _render_chat_prompt(tokenizer: object, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        preferred_calls = (
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
            {"tokenize": False, "add_generation_prompt": True},
        )
        for kwargs in preferred_calls:
            try:
                rendered = apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **kwargs,
                )
            except Exception:
                continue
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        try:
            rendered = apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            return prompt
    return prompt


def _normalize_chat_text(text: str) -> str:
    normalized = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL).strip()
    return normalized or text.strip()


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
        devices: str | None = None,
        log_embedding_calls: bool = False,
        show_backend_progress: bool = False,
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
        self._preferred_device = devices
        self._resolved_device = self._resolve_device(devices)
        self._log_embedding_calls = log_embedding_calls
        self._show_backend_progress = show_backend_progress
        self._embedding_backend: object | None = None
        self._rerank_backend: object | None = None
        self._embedding_call_count = 0
        self._embedded_text_count = 0
        self._embedding_total_duration_ms = 0.0
        self._last_embedding_duration_ms = 0.0
        self._last_embedding_request_size = 0
        self._actual_embedding_device: str | None = None

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
        dense_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch_texts = list(texts[start : start + self._batch_size])
            started = time.perf_counter()
            with self._backend_output_context():
                payload = encoder.encode(
                    batch_texts,
                    batch_size=self._batch_size,
                    max_length=self._max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            duration_ms = (time.perf_counter() - started) * 1000.0
            batch_vectors = payload.get("dense_vecs") if isinstance(payload, dict) else payload
            if batch_vectors is None:
                raise RuntimeError("Local BGE embedding backend returned no dense vectors")
            self._embedding_call_count += 1
            self._embedded_text_count += len(batch_texts)
            self._embedding_total_duration_ms += duration_ms
            self._last_embedding_duration_ms = duration_ms
            self._last_embedding_request_size = len(batch_texts)
            if self._log_embedding_calls:
                _LOGGER.info(
                    "embedding_call provider=%s model=%s device=%s "
                    "encode_batch_size=%s request_size=%s duration_ms=%.3f",
                    self.provider_name,
                    self.embedding_model_name,
                    self.embedding_device,
                    self.embedding_batch_size,
                    len(batch_texts),
                    duration_ms,
                )
            dense_vectors.extend(list(vector) for vector in batch_vectors)
        return dense_vectors

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        backend = self._embedding_backend or self._load_embedding_backend()
        self._embedding_backend = backend
        encoder = cast(Any, backend)
        sparse_vectors: list[dict[int, float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch_texts = list(texts[start : start + self._batch_size])
            with self._backend_output_context():
                payload = encoder.encode(
                    batch_texts,
                    batch_size=self._batch_size,
                    max_length=self._max_length,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
            sparse_payload = payload.get("lexical_weights") if isinstance(payload, dict) else None
            if sparse_payload is None and isinstance(payload, dict):
                sparse_payload = payload.get("sparse_vecs") or payload.get("sparse_weights")
            if sparse_payload is None:
                raise RuntimeError("Local BGE embedding backend returned no sparse vectors")
            sparse_vectors.extend(self._normalize_sparse_payload(item) for item in sparse_payload)
        return sparse_vectors

    def rerank(self, query: str, candidates: Sequence[object]) -> list[int]:
        if not candidates:
            return []
        backend = self._rerank_backend or self._load_rerank_backend()
        self._rerank_backend = backend
        reranker = cast(Any, backend)
        candidate_texts = [self._candidate_text(candidate) for candidate in candidates]
        pairs = [[query, candidate_text] for candidate_text in candidate_texts]
        with self._backend_output_context():
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
            module = _load_flagembedding_module()
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("FlagEmbedding is required for local BGE embeddings") from exc
        encoder_cls = getattr(module, "BGEM3FlagModel", None)
        if encoder_cls is None:
            raise RuntimeError("FlagEmbedding BGEM3FlagModel is unavailable")
        try:
            with self._backend_output_context():
                backend = cast(
                    object,
                    encoder_cls(
                        self._embedding_model_ref,
                        normalize_embeddings=self._normalize_embeddings,
                        use_fp16=self._use_fp16,
                        devices=self._resolved_device,
                        batch_size=self._batch_size,
                        query_max_length=self._max_length,
                        passage_max_length=self._max_length,
                        return_sparse=True,
                        return_colbert_vecs=False,
                    ),
                )
            backend = suppress_backend_fast_tokenizer_padding_warning(backend)
            self._actual_embedding_device = self._infer_backend_device(backend, fallback=self._resolved_device)
            return backend
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Local BGE embedding load failed: {exc}") from exc

    def _load_rerank_backend(self) -> object:
        try:
            module = _load_flagembedding_module()
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("FlagEmbedding is required for local BGE rerank") from exc
        auto_reranker_cls = getattr(module, "FlagAutoReranker", None)
        if auto_reranker_cls is None:
            raise RuntimeError("FlagEmbedding FlagAutoReranker is unavailable")
        model_class = _infer_flagembedding_reranker_model_class(self._rerank_model_ref)
        try:
            with self._backend_output_context():
                backend = cast(
                    object,
                    auto_reranker_cls.from_finetuned(
                        self._rerank_model_ref,
                        model_class=model_class,
                        use_fp16=self._use_fp16,
                        trust_remote_code=True,
                        devices=self._resolved_device,
                        batch_size=self._rerank_batch_size,
                        max_length=self._rerank_max_length,
                        normalize=False,
                    ),
                )
            return suppress_backend_fast_tokenizer_padding_warning(backend)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Local BGE rerank load failed: {exc}") from exc

    @staticmethod
    def _normalize_sparse_payload(payload: object) -> dict[int, float]:
        if isinstance(payload, dict):
            normalized: dict[int, float] = {}
            for key, value in payload.items():
                try:
                    normalized[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        if isinstance(payload, list):
            normalized = {}
            for item in payload:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    continue
                try:
                    normalized[int(item[0])] = float(item[1])
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        raise RuntimeError(f"Unsupported sparse embedding payload: {type(payload)!r}")

    @staticmethod
    def _candidate_text(candidate: object) -> str:
        if isinstance(candidate, str):
            return candidate
        text = getattr(candidate, "text", None)
        if isinstance(text, str):
            return text
        return str(candidate)

    @property
    def embedding_batch_size(self) -> int:
        return self._batch_size

    @property
    def embedding_device(self) -> str:
        return self._actual_embedding_device or self._resolved_device

    def set_embedding_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        self._batch_size = batch_size

    def set_device_preference(self, device: str | None) -> None:
        self._preferred_device = device
        self._resolved_device = self._resolve_device(device)
        self._embedding_backend = None
        self._rerank_backend = None
        self._actual_embedding_device = None

    def set_embedding_call_logging(self, enabled: bool) -> None:
        self._log_embedding_calls = enabled

    def set_backend_progress_enabled(self, enabled: bool) -> None:
        self._show_backend_progress = enabled

    def reset_embedding_stats(self) -> None:
        self._embedding_call_count = 0
        self._embedded_text_count = 0
        self._embedding_total_duration_ms = 0.0
        self._last_embedding_duration_ms = 0.0
        self._last_embedding_request_size = 0

    def embedding_runtime_info(self) -> dict[str, object]:
        return {
            "provider": self.provider_name,
            "model_name": self.embedding_model_name,
            "device": self.embedding_device,
            "encode_batch_size": self.embedding_batch_size,
            "resolved_device": self._resolved_device,
            "preferred_device": self._preferred_device,
        }

    def embedding_stats(self) -> dict[str, object]:
        return {
            **self.embedding_runtime_info(),
            "call_count": self._embedding_call_count,
            "text_count": self._embedded_text_count,
            "total_duration_ms": round(self._embedding_total_duration_ms, 3),
            "last_duration_ms": round(self._last_embedding_duration_ms, 3),
            "last_request_size": self._last_embedding_request_size,
        }

    @staticmethod
    def _infer_backend_device(backend: object, *, fallback: str) -> str:
        target_devices = getattr(backend, "target_devices", None)
        if isinstance(target_devices, list) and target_devices:
            return str(target_devices[0])
        model = getattr(backend, "model", None)
        device = getattr(model, "device", None)
        if device is not None:
            return str(device)
        return fallback

    def _backend_output_context(self) -> ExitStack | nullcontext[None]:
        if self._show_backend_progress:
            return nullcontext()
        sink = io.StringIO()
        stack = ExitStack()
        stack.enter_context(redirect_stdout(sink))
        stack.enter_context(redirect_stderr(sink))
        return stack

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if isinstance(device, str) and device.strip():
            normalized = device.strip().lower()
            if normalized == "auto":
                device = None
            else:
                return device.strip()
        try:
            import torch
        except Exception:  # pragma: no cover
            return "cpu"
        if getattr(torch.cuda, "is_available", None) and torch.cuda.is_available():
            return "cuda:0"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"


suppress_fast_tokenizer_padding_warning = suppress_backend_fast_tokenizer_padding_warning

__all__ = [
    "FallbackEmbeddingRepo",
    "LocalBgeProviderRepo",
    "LocalHfChatProviderRepo",
    "OllamaProviderRepo",
    "OpenAIProviderRepo",
    "expand_optional_path",
    "resolve_huggingface_snapshot_path",
    "resolve_local_model_reference",
    "suppress_backend_fast_tokenizer_padding_warning",
    "suppress_fast_tokenizer_padding_warning",
]
