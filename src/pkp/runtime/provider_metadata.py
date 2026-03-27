from __future__ import annotations


def provider_name(provider: object) -> str:
    explicit_name = getattr(provider, "provider_name", None)
    if isinstance(explicit_name, str) and explicit_name:
        return explicit_name

    fallback_name = getattr(provider, "name", None)
    if isinstance(fallback_name, str) and fallback_name:
        return fallback_name

    class_name = provider.__class__.__name__
    normalized = class_name.removesuffix("ProviderRepo").removesuffix("Repo")
    return normalized.replace("_", "-").lower() or "unknown"


def provider_model(provider: object, capability: str) -> str | None:
    attribute_names = {
        "chat": ("chat_model_name", "_chat_model", "_model"),
        "embed": ("embedding_model_name", "_embedding_model"),
        "rerank": ("rerank_model_name", "_rerank_model"),
    }.get(capability, ())
    for attribute_name in attribute_names:
        value = getattr(provider, attribute_name, None)
        if isinstance(value, str) and value:
            return value
    return None


def capability_configured(provider: object, capability: str) -> bool:
    explicit_attribute = getattr(provider, f"is_{capability}_configured", None)
    if isinstance(explicit_attribute, bool):
        return explicit_attribute
    if capability == "rerank":
        return callable(getattr(provider, "rerank", None))
    return provider_model(provider, capability) is not None and callable(getattr(provider, capability, None))


def embedding_space(provider: object) -> str:
    model = provider_model(provider, "embed") or "default"
    return f"{provider_name(provider)}::{model}"
