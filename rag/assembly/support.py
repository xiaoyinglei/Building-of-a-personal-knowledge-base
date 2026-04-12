from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace

from rag.assembly.models import (
    AssemblyConfig,
    AssemblyOverrides,
    AssemblyProfileSpec,
    CapabilityRequirements,
    ProviderConfig,
    TokenizerConfig,
)
from rag.providers.adapters import (
    FallbackEmbeddingRepo,
    LocalBgeProviderRepo,
    OllamaProviderRepo,
    OpenAIProviderRepo,
)


def first_non_blank(*values: str | None) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def env_int(*names: str) -> int | None:
    value = first_env(*names)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def env_bool(*names: str) -> bool | None:
    value = first_env(*names)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def normalize_gemini_base_url(url: str | None) -> str | None:
    if url is None:
        return None
    normalized = url.rstrip("/")
    if "generativelanguage.googleapis.com" in normalized and not normalized.endswith("/openai"):
        return f"{normalized}/openai"
    return normalized


def compatibility_config_from_environment() -> tuple[AssemblyConfig, dict[str, str | int | bool | None]]:
    compatibility_inputs: dict[str, str | int | bool | None] = {}
    profiles: list[ProviderConfig] = []

    def remember(key: str, value: str | int | bool | None) -> str | int | bool | None:
        if value is not None:
            compatibility_inputs[key] = value
        return value

    openai_api_key = remember(
        "openai_api_key",
        first_env("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "PKP_OPENAI__API_KEY"),
    )
    openai_base_url = remember(
        "openai_base_url",
        normalize_gemini_base_url(first_env("OPENAI_BASE_URL", "GEMINI_BASE_URL", "PKP_OPENAI__BASE_URL")),
    )
    openai_chat_model = remember(
        "openai_chat_model",
        first_env("OPENAI_MODEL", "GEMINI_CHAT_MODEL", "PKP_OPENAI__MODEL"),
    )
    openai_embedding_model = remember(
        "openai_embedding_model",
        first_env(
            "OPENAI_EMBEDDING_MODEL",
            "GEMINI_EMBEDDING_MODEL",
            "PKP_OPENAI__EMBEDDING_MODEL",
        ),
    )
    if openai_api_key and (openai_chat_model or openai_embedding_model):
        label_base = "OpenAI"
        if isinstance(openai_base_url, str) and "generativelanguage.googleapis.com" in openai_base_url:
            label_base = "Gemini (OpenAI compatible)"
        profiles.append(
            ProviderConfig(
                profile_id="openai-compatible",
                provider_kind="openai-compatible",
                location="cloud",
                label=f"{label_base} / {openai_chat_model or openai_embedding_model}",
                api_key=str(openai_api_key),
                base_url=None if openai_base_url is None else str(openai_base_url),
                chat_model=None if openai_chat_model is None else str(openai_chat_model),
                embedding_model=None if openai_embedding_model is None else str(openai_embedding_model),
            )
        )

    ollama_base_url = remember("ollama_base_url", first_env("OLLAMA_BASE_URL", "PKP_OLLAMA__BASE_URL"))
    ollama_chat_model = remember("ollama_chat_model", first_env("OLLAMA_CHAT_MODEL", "PKP_OLLAMA__CHAT_MODEL"))
    ollama_embedding_model = remember(
        "ollama_embedding_model",
        first_env("OLLAMA_EMBEDDING_MODEL", "PKP_OLLAMA__EMBEDDING_MODEL"),
    )
    if ollama_base_url and (ollama_chat_model or ollama_embedding_model):
        profiles.append(
            ProviderConfig(
                profile_id="ollama",
                provider_kind="ollama",
                location="local",
                label=f"Ollama / {ollama_chat_model or ollama_embedding_model}",
                base_url=str(ollama_base_url),
                chat_model=None if ollama_chat_model is None else str(ollama_chat_model),
                embedding_model=None if ollama_embedding_model is None else str(ollama_embedding_model),
            )
        )

    local_bge_enabled = remember(
        "local_bge_enabled",
        first_env("PKP_LOCAL_BGE__ENABLED", "RAG_LOCAL_BGE_ENABLED"),
    )
    local_bge_embedding_model = remember(
        "local_bge_embedding_model",
        first_env("PKP_LOCAL_BGE__EMBEDDING_MODEL", "RAG_LOCAL_BGE_EMBEDDING_MODEL"),
    )
    local_bge_embedding_model_path = remember(
        "local_bge_embedding_model_path",
        first_env("PKP_LOCAL_BGE__EMBEDDING_MODEL_PATH", "RAG_LOCAL_BGE_EMBEDDING_MODEL_PATH"),
    )
    local_bge_embedding_batch_size = remember(
        "local_bge_embedding_batch_size",
        env_int("PKP_LOCAL_BGE__EMBEDDING_BATCH_SIZE", "RAG_LOCAL_BGE_EMBEDDING_BATCH_SIZE"),
    )
    local_bge_device = remember(
        "local_bge_device",
        first_env("PKP_LOCAL_BGE__DEVICE", "RAG_LOCAL_BGE_DEVICE"),
    )
    local_bge_rerank_model = remember(
        "local_bge_rerank_model",
        first_env("RAG_RERANK_MODEL", "PKP_LOCAL_BGE__RERANK_MODEL"),
    )
    local_bge_rerank_model_path = remember(
        "local_bge_rerank_model_path",
        first_env("RAG_RERANK_MODEL_PATH", "PKP_LOCAL_BGE__RERANK_MODEL_PATH"),
    )
    local_bge_rerank_batch_size = remember(
        "local_bge_rerank_batch_size",
        env_int("PKP_LOCAL_BGE__RERANK_BATCH_SIZE", "RAG_LOCAL_BGE_RERANK_BATCH_SIZE"),
    )
    local_bge_allowed = (str(local_bge_enabled).lower() if local_bge_enabled is not None else "") not in {
        "0",
        "false",
        "no",
        "off",
    }
    if local_bge_allowed and (
        local_bge_embedding_model
        or local_bge_embedding_model_path
        or local_bge_rerank_model
        or local_bge_rerank_model_path
    ):
        profiles.append(
            ProviderConfig(
                profile_id="local-bge",
                provider_kind="local-bge",
                location="local",
                label=f"Local BGE / {local_bge_embedding_model or local_bge_rerank_model or 'custom'}",
                embedding_model=None if local_bge_embedding_model is None else str(local_bge_embedding_model),
                embedding_model_path=None
                if local_bge_embedding_model_path is None
                else str(local_bge_embedding_model_path),
                embedding_batch_size=(
                    None if local_bge_embedding_batch_size is None else int(local_bge_embedding_batch_size)
                ),
                device=None if local_bge_device is None else str(local_bge_device),
                rerank_model=None if local_bge_rerank_model is None else str(local_bge_rerank_model),
                rerank_model_path=None if local_bge_rerank_model_path is None else str(local_bge_rerank_model_path),
                rerank_batch_size=(
                    None if local_bge_rerank_batch_size is None else int(local_bge_rerank_batch_size)
                ),
            )
        )

    tokenizer_config = TokenizerConfig(
        embedding_model_name=first_env(
            "RAG_EMBEDDING_MODEL",
            "RAG_INDEX_EMBEDDING_MODEL",
        ),
        tokenizer_model_name=first_env(
            "RAG_TOKENIZER_MODEL",
            "RAG_BUDGET_TOKENIZER_MODEL",
        ),
        chunking_tokenizer_model_name=first_env(
            "RAG_CHUNKING_TOKENIZER_MODEL",
            "RAG_DOCLING_TOKENIZER_MODEL",
        ),
        tokenizer_backend=first_env("RAG_TOKENIZER_BACKEND"),
        chunk_token_size=env_int("RAG_CHUNK_TOKEN_SIZE"),
        chunk_overlap_tokens=env_int("RAG_CHUNK_OVERLAP_TOKENS"),
        max_context_tokens=env_int("RAG_MAX_CONTEXT_TOKENS"),
        prompt_reserved_tokens=env_int("RAG_PROMPT_RESERVED_TOKENS"),
        local_files_only=env_bool("RAG_TOKENIZER_LOCAL_FILES_ONLY"),
    )
    return AssemblyConfig(profiles=tuple(profiles), tokenizer=tokenizer_config), compatibility_inputs


def assembly_profiles(
    *,
    config: AssemblyConfig | None,
    compatibility_config: AssemblyConfig,
) -> tuple[AssemblyProfileSpec, ...]:
    del config
    compat = {profile.profile_id: profile for profile in compatibility_config.profiles}
    compat_openai = compat.get("openai-compatible")
    compat_ollama = compat.get("ollama")
    compat_local_bge = compat.get("local-bge")
    return (
        AssemblyProfileSpec(
            profile_id="local_full",
            label="Local Full",
            description="Prefer local retrieval and local chat. Uses local BGE for retrieval and Ollama for chat.",
            location="local",
            overrides=AssemblyOverrides(
                embedding=local_retrieval_provider(compat_local_bge, compat_ollama),
                rerank=rerank_provider(compat_local_bge),
                chat=_provider_or_default(
                    compat_ollama,
                    fallback_kind="ollama",
                    fallback_location="local",
                    fallback_base_url="http://localhost:11434",
                ),
            ),
        ),
        AssemblyProfileSpec(
            profile_id="local_retrieval_cloud_chat",
            label="Local Retrieval + Cloud Chat",
            description=(
                "Prefer local retrieval with a cloud chat model. Uses local BGE for retrieval and "
                "OpenAI-compatible chat for generation."
            ),
            location="hybrid",
            overrides=AssemblyOverrides(
                embedding=local_retrieval_provider(compat_local_bge, compat_ollama),
                rerank=rerank_provider(compat_local_bge),
                chat=_provider_or_default(
                    compat_openai,
                    fallback_kind="openai-compatible",
                    fallback_location="cloud",
                    fallback_base_url="https://api.openai.com/v1",
                ),
            ),
        ),
        AssemblyProfileSpec(
            profile_id="cloud_full",
            label="Cloud Full",
            description="Prefer a cloud chat + embedding stack, with optional local rerank when available.",
            location="cloud",
            overrides=AssemblyOverrides(
                embedding=_provider_or_default(
                    compat_openai,
                    fallback_kind="openai-compatible",
                    fallback_location="cloud",
                    fallback_base_url="https://api.openai.com/v1",
                ),
                chat=_provider_or_default(
                    compat_openai,
                    fallback_kind="openai-compatible",
                    fallback_location="cloud",
                    fallback_base_url="https://api.openai.com/v1",
                ),
                rerank=rerank_provider(compat_local_bge),
            ),
        ),
        AssemblyProfileSpec(
            profile_id="test_minimal",
            label="Test Minimal",
            description="Minimal test profile. Allows degraded assembly and uses fallback embedding when needed.",
            location="test",
            recommended_requirements=CapabilityRequirements(
                require_embedding=True,
                require_chat=False,
                require_rerank=False,
                allow_degraded=True,
            ),
            overrides=AssemblyOverrides(),
        ),
    )


def assembly_profile_by_id(
    profiles: Sequence[AssemblyProfileSpec],
    profile_id: str | None,
) -> AssemblyProfileSpec | None:
    if profile_id is None:
        return None
    for profile in profiles:
        if profile.profile_id == profile_id:
            return profile
    return None
def merge_provider_config(
    high: ProviderConfig | None,
    low: ProviderConfig | None,
) -> ProviderConfig | None:
    if high is None:
        return low
    if low is None:
        return high
    return ProviderConfig(
        provider_kind=high.provider_kind or low.provider_kind,
        location=first_non_blank(high.location, low.location) or low.location,
        profile_id=first_non_blank(high.profile_id, low.profile_id),
        label=first_non_blank(high.label, low.label),
        api_key=first_non_blank(high.api_key, low.api_key),
        base_url=first_non_blank(high.base_url, low.base_url),
        chat_model=first_non_blank(high.chat_model, low.chat_model),
        embedding_model=first_non_blank(high.embedding_model, low.embedding_model),
        rerank_model=first_non_blank(high.rerank_model, low.rerank_model),
        embedding_model_path=first_non_blank(high.embedding_model_path, low.embedding_model_path),
        rerank_model_path=first_non_blank(high.rerank_model_path, low.rerank_model_path),
        embedding_batch_size=(
            first_positive_int(high.embedding_batch_size, low.embedding_batch_size)
            if high.embedding_batch_size or low.embedding_batch_size
            else None
        ),
        rerank_batch_size=(
            first_positive_int(high.rerank_batch_size, low.rerank_batch_size)
            if high.rerank_batch_size or low.rerank_batch_size
            else None
        ),
        device=first_non_blank(high.device, low.device),
        enabled=high.enabled if high is not None else low.enabled,
    )


def merge_tokenizer_config(
    high: TokenizerConfig | None,
    low: TokenizerConfig | None,
) -> TokenizerConfig | None:
    if high is None:
        return low
    if low is None:
        return high
    return TokenizerConfig(
        embedding_model_name=first_non_blank(high.embedding_model_name, low.embedding_model_name),
        tokenizer_model_name=first_non_blank(high.tokenizer_model_name, low.tokenizer_model_name),
        chunking_tokenizer_model_name=first_non_blank(
            high.chunking_tokenizer_model_name,
            low.chunking_tokenizer_model_name,
        ),
        tokenizer_backend=first_non_blank(high.tokenizer_backend, low.tokenizer_backend),
        chunk_token_size=high.chunk_token_size if high.chunk_token_size is not None else low.chunk_token_size,
        chunk_overlap_tokens=(
            high.chunk_overlap_tokens if high.chunk_overlap_tokens is not None else low.chunk_overlap_tokens
        ),
        max_context_tokens=high.max_context_tokens if high.max_context_tokens is not None else low.max_context_tokens,
        prompt_reserved_tokens=(
            high.prompt_reserved_tokens if high.prompt_reserved_tokens is not None else low.prompt_reserved_tokens
        ),
        local_files_only=high.local_files_only if high.local_files_only is not None else low.local_files_only,
    )


def merge_assembly_config(
    high: AssemblyConfig | None,
    low: AssemblyConfig | None,
) -> AssemblyConfig | None:
    if high is None:
        return low
    if low is None:
        return high
    return AssemblyConfig(
        default_profile_id=first_non_blank(high.default_profile_id, low.default_profile_id),
        profiles=tuple([*high.profiles, *low.profiles]),
        chat=merge_provider_config(high.chat, low.chat),
        embedding=merge_provider_config(high.embedding, low.embedding),
        rerank=merge_provider_config(high.rerank, low.rerank),
        tokenizer=merge_tokenizer_config(high.tokenizer, low.tokenizer),
    )


def merge_assembly_overrides(
    high: AssemblyOverrides | None,
    low: AssemblyOverrides | None,
) -> AssemblyOverrides | None:
    if high is None:
        return low
    if low is None:
        return high
    return AssemblyOverrides(
        chat=merge_provider_config(high.chat, low.chat),
        embedding=merge_provider_config(high.embedding, low.embedding),
        rerank=merge_provider_config(high.rerank, low.rerank),
        tokenizer=merge_tokenizer_config(high.tokenizer, low.tokenizer),
    )


def local_retrieval_provider(
    local_bge: ProviderConfig | None,
    ollama: ProviderConfig | None,
) -> ProviderConfig:
    if local_bge is not None:
        return _strip_profile(local_bge)
    if ollama is not None and ollama.base_url and ollama.embedding_model:
        return _strip_profile(ollama)
    return ProviderConfig(provider_kind="local-bge", location="local")


def rerank_provider(local_bge: ProviderConfig | None) -> ProviderConfig | None:
    return _strip_profile(local_bge) if local_bge is not None else None


def _strip_profile(provider: ProviderConfig) -> ProviderConfig:
    return replace(provider, profile_id=None, label=None)


def _provider_or_default(
    provider: ProviderConfig | None,
    *,
    fallback_kind: str,
    fallback_location: str,
    fallback_base_url: str | None = None,
) -> ProviderConfig:
    if provider is not None:
        return _strip_profile(provider)
    return ProviderConfig(
        provider_kind=fallback_kind,
        location=fallback_location,
        base_url=fallback_base_url,
    )


def build_provider(provider_config: ProviderConfig) -> object:
    kind = provider_config.provider_kind
    if kind == "openai-compatible":
        return OpenAIProviderRepo(
            api_key=provider_config.api_key,
            base_url=normalize_gemini_base_url(provider_config.base_url) or "https://api.openai.com/v1",
            model=provider_config.chat_model or "",
            embedding_model=provider_config.embedding_model or "",
        )
    if kind == "ollama":
        return OllamaProviderRepo(
            base_url=provider_config.base_url or "http://localhost:11434",
            chat_model=provider_config.chat_model or "",
            embedding_model=provider_config.embedding_model,
        )
    if kind == "local-bge":
        return LocalBgeProviderRepo(
            embedding_model=provider_config.embedding_model or "",
            embedding_model_path=provider_config.embedding_model_path,
            rerank_model=provider_config.rerank_model or "",
            rerank_model_path=provider_config.rerank_model_path,
            batch_size=provider_config.embedding_batch_size or 8,
            rerank_batch_size=provider_config.rerank_batch_size or 8,
            devices=provider_config.device,
        )
    if kind in {"fallback", "default-embedding"}:
        return FallbackEmbeddingRepo()
    raise RuntimeError(f"Unsupported provider_kind: {kind}")


def first_positive_int(*values: int | None) -> int:
    for value in values:
        if isinstance(value, int) and value > 0:
            return value
    return 1


def first_non_negative_int(*values: int | None) -> int:
    for value in values:
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def first_bool(*values: bool | None) -> bool:
    for value in values:
        if isinstance(value, bool):
            return value
    return False


__all__ = [
    "assembly_profile_by_id",
    "assembly_profiles",
    "build_provider",
    "compatibility_config_from_environment",
    "env_bool",
    "env_int",
    "first_bool",
    "first_env",
    "first_non_blank",
    "first_non_negative_int",
    "first_positive_int",
    "local_retrieval_provider",
    "merge_assembly_config",
    "merge_assembly_overrides",
    "normalize_gemini_base_url",
    "rerank_provider",
]
