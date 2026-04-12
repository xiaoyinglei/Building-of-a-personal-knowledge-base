from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


def _provider_name(provider: object) -> str:
    explicit = getattr(provider, "provider_name", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    fallback = getattr(provider, "name", None)
    if isinstance(fallback, str) and fallback:
        return fallback
    normalized = provider.__class__.__name__.removesuffix("ProviderRepo").removesuffix("Repo")
    return normalized.replace("_", "-").lower() or "unknown"


def _provider_model(provider: object, capability: str) -> str | None:
    attribute_names = {
        "chat": ("chat_model_name", "_chat_model", "_model"),
        "embedding": ("embedding_model_name", "_embedding_model"),
        "rerank": ("rerank_model_name", "_rerank_model"),
    }.get(capability, ())
    for attribute_name in attribute_names:
        value = getattr(provider, attribute_name, None)
        if isinstance(value, str) and value:
            return value
    return None


def _supports_capability(provider: object, capability: str) -> bool:
    method_name = {
        "chat": "chat",
        "embedding": "embed",
        "rerank": "rerank",
    }[capability]
    configured_name = {
        "chat": "is_chat_configured",
        "embedding": "is_embed_configured",
        "rerank": "is_rerank_configured",
    }[capability]
    method = getattr(provider, method_name, None)
    if not callable(method):
        return False
    configured = getattr(provider, configured_name, True)
    return bool(configured)


@dataclass(frozen=True, slots=True)
class EmbeddingCapabilityBinding:
    backend: object
    space: str
    location: str = "runtime"
    provider_name: str = field(init=False)
    model_name: str | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_name", _provider_name(self.backend))
        object.__setattr__(self, "model_name", _provider_model(self.backend, "embedding"))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        embed = getattr(self.backend, "embed", None)
        if not callable(embed):
            raise RuntimeError("Embedding capability is not available")
        return list(embed(list(texts)))

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        embed_query = getattr(self.backend, "embed_query", None)
        if callable(embed_query):
            return list(embed_query(list(texts)))
        return self.embed(texts)


@dataclass(frozen=True, slots=True)
class ChatCapabilityBinding:
    backend: object
    location: str = "runtime"
    provider_name: str = field(init=False)
    model_name: str | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_name", _provider_name(self.backend))
        object.__setattr__(self, "model_name", _provider_model(self.backend, "chat"))

    def chat(self, prompt: str) -> str:
        chat = getattr(self.backend, "chat", None)
        if not callable(chat):
            raise RuntimeError("Chat capability is not available")
        return str(chat(prompt))


@dataclass(frozen=True, slots=True)
class RerankCapabilityBinding:
    backend: object
    location: str = "runtime"
    provider_name: str = field(init=False)
    model_name: str | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_name", _provider_name(self.backend))
        object.__setattr__(self, "model_name", _provider_model(self.backend, "rerank"))

    def rerank(self, query: str, candidates: Sequence[object]) -> list[int]:
        rerank = getattr(self.backend, "rerank", None)
        if not callable(rerank):
            raise RuntimeError("Rerank capability is not available")
        return list(rerank(query, list(candidates)))


CapabilityBinding = EmbeddingCapabilityBinding | ChatCapabilityBinding | RerankCapabilityBinding

__all__ = [
    "CapabilityBinding",
    "ChatCapabilityBinding",
    "EmbeddingCapabilityBinding",
    "RerankCapabilityBinding",
    "_supports_capability",
]
