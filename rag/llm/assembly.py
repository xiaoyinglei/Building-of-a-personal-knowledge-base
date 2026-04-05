from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from rag.llm._providers.fallback_embedding_repo import FallbackEmbeddingRepo
from rag.llm._providers.local_bge_provider_repo import LocalBgeProviderRepo
from rag.llm._providers.ollama_provider_repo import OllamaProviderRepo
from rag.llm._providers.openai_provider_repo import OpenAIProviderRepo
from rag.schema._types.text import (
    DEFAULT_TOKENIZER_FALLBACK_MODEL,
    TokenAccountingService,
    TokenizerContract,
    load_env_file,
)

AssemblyStatus = Literal["valid", "degraded", "invalid"]
IssueSeverity = Literal["info", "warning", "error"]
DecisionSource = Literal["explicit", "profile", "config", "compat_env", "default"]

_DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _first_non_blank(*values: str | None) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _env_int(*names: str) -> int | None:
    value = _first_env(*names)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_bool(*names: str) -> bool | None:
    value = _first_env(*names)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_gemini_base_url(url: str | None) -> str | None:
    if url is None:
        return None
    normalized = url.rstrip("/")
    if "generativelanguage.googleapis.com" in normalized and not normalized.endswith("/openai"):
        return f"{normalized}/openai"
    return normalized


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


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    provider_kind: str
    location: str = "runtime"
    profile_id: str | None = None
    label: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    rerank_model: str | None = None
    embedding_model_path: str | None = None
    rerank_model_path: str | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    embedding_model_name: str | None = None
    tokenizer_model_name: str | None = None
    chunking_tokenizer_model_name: str | None = None
    tokenizer_backend: str | None = None
    chunk_token_size: int | None = None
    chunk_overlap_tokens: int | None = None
    max_context_tokens: int | None = None
    prompt_reserved_tokens: int | None = None
    local_files_only: bool | None = None


@dataclass(frozen=True, slots=True)
class AssemblyConfig:
    default_profile_id: str | None = None
    profiles: tuple[ProviderConfig, ...] = ()
    chat: ProviderConfig | None = None
    embedding: ProviderConfig | None = None
    rerank: ProviderConfig | None = None
    tokenizer: TokenizerConfig | None = None


@dataclass(frozen=True, slots=True)
class AssemblyOverrides:
    chat: ProviderConfig | None = None
    embedding: ProviderConfig | None = None
    rerank: ProviderConfig | None = None
    tokenizer: TokenizerConfig | None = None


@dataclass(frozen=True, slots=True)
class CapabilityRequirements:
    require_embedding: bool = True
    require_chat: bool = False
    require_rerank: bool = False
    allow_degraded: bool = True
    default_context_tokens: int = 1024
    default_chunk_token_size: int = 480
    default_chunk_overlap_tokens: int = 64
    default_prompt_reserved_tokens: int = 256


@dataclass(frozen=True, slots=True)
class AssemblyRequest:
    requirements: CapabilityRequirements = field(default_factory=CapabilityRequirements)
    profile_id: str | None = None
    config: AssemblyConfig | None = None
    overrides: AssemblyOverrides | None = None


@dataclass(frozen=True, slots=True)
class AssemblyIssue:
    severity: IssueSeverity
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AssemblyDecision:
    capability: str
    source: DecisionSource
    provider_kind: str
    provider_name: str | None
    model_name: str | None
    location: str | None
    reason: str
    selected: bool = True


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    profile_id: str
    label: str
    provider_kind: str
    location: str
    chat_model: str | None
    embedding_model: str | None
    rerank_model: str | None
    supports_chat: bool
    supports_embedding: bool
    supports_rerank: bool
    provider_config: ProviderConfig = field(repr=False)
    factory: Callable[[], object] = field(repr=False)

    def create_provider(self) -> object:
        return self.factory()


@dataclass(frozen=True, slots=True)
class AssemblyProfileSpec:
    profile_id: str
    label: str
    description: str
    location: str
    config: AssemblyConfig | None = None
    overrides: AssemblyOverrides | None = None
    recommended_requirements: CapabilityRequirements = field(default_factory=CapabilityRequirements)


@dataclass(frozen=True, slots=True)
class CapabilityCatalog:
    profiles: tuple[CapabilityProfile, ...] = ()
    assembly_profiles: tuple[AssemblyProfileSpec, ...] = ()
    diagnostics: tuple[AssemblyIssue, ...] = ()
    compatibility_inputs: dict[str, str | int | bool | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssemblyContracts:
    token_contract: TokenizerContract
    token_accounting: TokenAccountingService
    runtime_contract_payload: dict[str, str | int | bool]


@dataclass(frozen=True, slots=True)
class AssemblyDiagnostics:
    status: AssemblyStatus
    issues: tuple[AssemblyIssue, ...] = ()
    decisions: tuple[AssemblyDecision, ...] = ()
    compatibility_inputs: dict[str, str | int | bool | None] = field(default_factory=dict)

    @property
    def warnings(self) -> tuple[AssemblyIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def errors(self) -> tuple[AssemblyIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    def raise_for_invalid(self) -> None:
        if self.status != "invalid":
            return
        if self.errors:
            detail = "; ".join(issue.message for issue in self.errors)
        else:
            detail = "assembly produced an invalid result"
        raise RuntimeError(detail)


@dataclass(frozen=True, slots=True)
class RuntimeContractGovernance:
    status: AssemblyStatus
    should_persist: bool
    mismatches: dict[str, tuple[Any | None, Any | None]] = field(default_factory=dict)
    issues: tuple[AssemblyIssue, ...] = ()

    def raise_for_invalid(self) -> None:
        if self.status != "invalid":
            return
        detail = "; ".join(issue.message for issue in self.issues) or "runtime contract governance failed"
        raise RuntimeError(detail)


@dataclass(frozen=True, slots=True)
class CapabilityBundle:
    request: AssemblyRequest
    effective_request: AssemblyRequest
    diagnostics: AssemblyDiagnostics
    contracts: AssemblyContracts
    embedding_bindings: tuple[EmbeddingCapabilityBinding, ...]
    chat_bindings: tuple[ChatCapabilityBinding, ...]
    rerank_bindings: tuple[RerankCapabilityBinding, ...]
    selected_profile_id: str | None = None
    profiles: tuple[CapabilityProfile, ...] = ()

    @property
    def status(self) -> AssemblyStatus:
        return self.diagnostics.status

    @property
    def token_contract(self) -> TokenizerContract:
        return self.contracts.token_contract

    @property
    def token_accounting(self) -> TokenAccountingService:
        return self.contracts.token_accounting

    @property
    def runtime_contract_payload(self) -> dict[str, str | int | bool]:
        return self.contracts.runtime_contract_payload


@dataclass(frozen=True, slots=True)
class _CandidateSource:
    source: DecisionSource
    provider_config: ProviderConfig
    cache_key: str
    reason: str


class CapabilityAssemblyService:
    def __init__(self, *, env_path: str = ".env") -> None:
        self._env_path = env_path

    def catalog_from_environment(self, *, config: AssemblyConfig | None = None) -> CapabilityCatalog:
        self._load_env()
        compatibility_config, compatibility_inputs = self._compatibility_config_from_environment()
        profiles = self._catalog_profiles(config=config, compatibility_config=compatibility_config)
        assembly_profiles = self._assembly_profiles(
            config=config,
            compatibility_config=compatibility_config,
        )
        return CapabilityCatalog(
            profiles=profiles,
            assembly_profiles=assembly_profiles,
            diagnostics=(),
            compatibility_inputs=compatibility_inputs,
        )

    def request_for_profile(
        self,
        profile_id: str,
        *,
        requirements: CapabilityRequirements | None = None,
        config: AssemblyConfig | None = None,
        overrides: AssemblyOverrides | None = None,
    ) -> AssemblyRequest:
        self._load_env()
        compatibility_config, _compatibility_inputs = self._compatibility_config_from_environment()
        profile = self._assembly_profile_by_id(
            self._assembly_profiles(config=config, compatibility_config=compatibility_config),
            profile_id,
        )
        if profile is None:
            raise RuntimeError(f"Unknown assembly profile: {profile_id!r}")
        return AssemblyRequest(
            requirements=requirements or profile.recommended_requirements,
            profile_id=profile.profile_id,
            config=config,
            overrides=overrides,
        )

    def evaluate_request(self, request: AssemblyRequest) -> CapabilityBundle:
        self._load_env()
        compatibility_config, compatibility_inputs = self._compatibility_config_from_environment()
        effective_request, assembly_profile = self._effective_request(
            request=request,
            compatibility_config=compatibility_config,
        )
        profiles = self._catalog_profiles(config=effective_request.config, compatibility_config=compatibility_config)
        provider_cache: dict[str, object] = {}
        issues: list[AssemblyIssue] = []
        decisions: list[AssemblyDecision] = []
        selected_profile_id: str | None = None

        embedding_bindings: list[EmbeddingCapabilityBinding] = []
        chat_bindings: list[ChatCapabilityBinding] = []
        rerank_bindings: list[RerankCapabilityBinding] = []

        if assembly_profile is not None:
            selected_profile_id = assembly_profile.profile_id
            decisions.append(
                AssemblyDecision(
                    capability="assembly_profile",
                    source="profile",
                    provider_kind="assembly-profile",
                    provider_name=assembly_profile.profile_id,
                    model_name=None,
                    location=assembly_profile.location,
                    reason=assembly_profile.description,
                )
            )

        explicit_profile = self._profile_by_id(profiles, effective_request.profile_id)
        if effective_request.profile_id and explicit_profile is None:
            issues.append(
                AssemblyIssue(
                    severity="warning" if effective_request.requirements.allow_degraded else "error",
                    code="requested_profile_missing",
                    message=(
                        f"Requested profile {effective_request.profile_id!r} is not available."
                    ),
                )
            )
        if explicit_profile is not None and selected_profile_id is None:
            selected_profile_id = explicit_profile.profile_id

        config_profile: CapabilityProfile | None = None
        if effective_request.config is not None and effective_request.config.default_profile_id:
            config_profile = self._profile_by_id(profiles, effective_request.config.default_profile_id)
            if config_profile is None:
                issues.append(
                    AssemblyIssue(
                        severity="warning" if effective_request.requirements.allow_degraded else "error",
                        code="config_default_profile_missing",
                        message=(
                            "Configured default profile "
                            f"{effective_request.config.default_profile_id!r} is not available."
                        ),
                    )
                )
            elif selected_profile_id is None:
                selected_profile_id = config_profile.profile_id

        embedding_binding = self._resolve_embedding_binding(
            request=effective_request,
            profiles=profiles,
            explicit_profile=explicit_profile,
            config_profile=config_profile,
            compatibility_config=compatibility_config,
            compatibility_inputs=compatibility_inputs,
            provider_cache=provider_cache,
            issues=issues,
            decisions=decisions,
        )
        if embedding_binding is not None:
            embedding_bindings.append(embedding_binding)

        chat_binding = self._resolve_chat_binding(
            request=effective_request,
            profiles=profiles,
            explicit_profile=explicit_profile,
            config_profile=config_profile,
            compatibility_config=compatibility_config,
            provider_cache=provider_cache,
            issues=issues,
            decisions=decisions,
        )
        if chat_binding is not None:
            chat_bindings.append(chat_binding)

        rerank_binding = self._resolve_rerank_binding(
            request=effective_request,
            profiles=profiles,
            explicit_profile=explicit_profile,
            config_profile=config_profile,
            compatibility_config=compatibility_config,
            provider_cache=provider_cache,
            issues=issues,
            decisions=decisions,
        )
        if rerank_binding is not None:
            rerank_bindings.append(rerank_binding)

        if not embedding_bindings:
            fallback = FallbackEmbeddingRepo()
            embedding_binding = EmbeddingCapabilityBinding(backend=fallback, space="default", location="local")
            embedding_bindings.append(embedding_binding)
            decisions.append(
                AssemblyDecision(
                    capability="embedding",
                    source="default",
                    provider_kind="fallback",
                    provider_name=embedding_binding.provider_name,
                    model_name=embedding_binding.model_name,
                    location=embedding_binding.location,
                    reason="No configured embedding provider was available; using fallback embedding backend.",
                )
            )
            issues.append(
                AssemblyIssue(
                    severity="warning",
                    code="fallback_embedding_selected",
                    message="No configured embedding provider was available; using fallback embedding backend.",
                )
            )

        token_contract = self._build_token_contract(
            request=effective_request,
            embedding_binding=embedding_bindings[0],
            compatibility_config=compatibility_config,
            compatibility_inputs=compatibility_inputs,
            issues=issues,
            decisions=decisions,
        )
        token_accounting = TokenAccountingService(token_contract)
        contracts = AssemblyContracts(
            token_contract=token_contract,
            token_accounting=token_accounting,
            runtime_contract_payload=self._runtime_contract_payload(token_contract, token_accounting),
        )
        diagnostics = AssemblyDiagnostics(
            status=self._diagnostics_status(issues),
            issues=tuple(issues),
            decisions=tuple(decisions),
            compatibility_inputs=compatibility_inputs,
        )
        return CapabilityBundle(
            request=request,
            effective_request=effective_request,
            diagnostics=diagnostics,
            contracts=contracts,
            embedding_bindings=tuple(embedding_bindings),
            chat_bindings=tuple(chat_bindings),
            rerank_bindings=tuple(rerank_bindings),
            selected_profile_id=selected_profile_id,
            profiles=profiles,
        )

    def assemble_request(self, request: AssemblyRequest) -> CapabilityBundle:
        bundle = self.evaluate_request(request)
        bundle.diagnostics.raise_for_invalid()
        return bundle

    def evaluate(
        self,
        *,
        profile_id: str | None = None,
        default_context_tokens: int = 1024,
        require_chat: bool = False,
        require_rerank: bool = False,
        allow_degraded: bool = True,
        config: AssemblyConfig | None = None,
        overrides: AssemblyOverrides | None = None,
    ) -> CapabilityBundle:
        """Compatibility-only helper. New code should build an AssemblyRequest and call evaluate_request()."""
        request = AssemblyRequest(
            requirements=CapabilityRequirements(
                require_chat=require_chat,
                require_rerank=require_rerank,
                allow_degraded=allow_degraded,
                default_context_tokens=default_context_tokens,
            ),
            profile_id=profile_id,
            config=config,
            overrides=overrides,
        )
        return self.evaluate_request(request)

    def assemble(
        self,
        *,
        profile_id: str | None = None,
        default_context_tokens: int = 1024,
        require_chat: bool = False,
        require_rerank: bool = False,
        allow_degraded: bool = True,
        config: AssemblyConfig | None = None,
        overrides: AssemblyOverrides | None = None,
    ) -> CapabilityBundle:
        """Compatibility-only helper. New code should build an AssemblyRequest and call assemble_request()."""
        return self.assemble_request(
            AssemblyRequest(
                requirements=CapabilityRequirements(
                    require_chat=require_chat,
                    require_rerank=require_rerank,
                    allow_degraded=allow_degraded,
                    default_context_tokens=default_context_tokens,
                ),
                profile_id=profile_id,
                config=config,
                overrides=overrides,
            )
        )

    def _effective_request(
        self,
        *,
        request: AssemblyRequest,
        compatibility_config: AssemblyConfig,
    ) -> tuple[AssemblyRequest, AssemblyProfileSpec | None]:
        profile = self._assembly_profile_by_id(
            self._assembly_profiles(config=request.config, compatibility_config=compatibility_config),
            request.profile_id,
        )
        if profile is None:
            return request, None
        return (
            AssemblyRequest(
                requirements=request.requirements,
                profile_id=None,
                config=self._merge_assembly_config(profile.config, request.config),
                overrides=self._merge_assembly_overrides(profile.overrides, request.overrides),
            ),
            profile,
        )

    def govern_runtime_contract(
        self,
        *,
        bundle: CapabilityBundle,
        stored_payload: dict[str, Any] | None,
    ) -> RuntimeContractGovernance:
        current_payload = bundle.runtime_contract_payload
        if stored_payload is None:
            return RuntimeContractGovernance(status="valid", should_persist=True)
        mismatches: dict[str, tuple[Any | None, Any | None]] = {}
        for key in (
            "embedding_model_name",
            "tokenizer_model_name",
            "chunking_tokenizer_model_name",
            "tokenizer_backend",
            "chunk_token_size",
            "chunk_overlap_tokens",
        ):
            if stored_payload.get(key) != current_payload.get(key):
                mismatches[key] = (current_payload.get(key), stored_payload.get(key))
        if not mismatches:
            return RuntimeContractGovernance(status="valid", should_persist=False)
        details = ", ".join(
            f"{field}: current={current!r} stored={stored!r}"
            for field, (current, stored) in mismatches.items()
        )
        issue = AssemblyIssue(
            severity="error",
            code="runtime_contract_mismatch",
            message=(
                "RAG runtime contract does not match the existing index. "
                f"Mismatched fields: {details}. Use the same embedding/tokenizer contract or rebuild the index."
            ),
        )
        return RuntimeContractGovernance(
            status="invalid",
            should_persist=False,
            mismatches=mismatches,
            issues=(issue,),
        )

    def _load_env(self) -> None:
        load_env_file(self._env_path)

    def _compatibility_config_from_environment(self) -> tuple[AssemblyConfig, dict[str, str | int | bool | None]]:
        compatibility_inputs: dict[str, str | int | bool | None] = {}
        profiles: list[ProviderConfig] = []

        def remember(key: str, value: str | int | bool | None) -> str | int | bool | None:
            if value is not None:
                compatibility_inputs[key] = value
            return value

        openai_api_key = remember(
            "openai_api_key",
            _first_env("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "PKP_OPENAI__API_KEY"),
        )
        openai_base_url = remember(
            "openai_base_url",
            _normalize_gemini_base_url(_first_env("OPENAI_BASE_URL", "GEMINI_BASE_URL", "PKP_OPENAI__BASE_URL")),
        )
        openai_chat_model = remember(
            "openai_chat_model",
            _first_env("OPENAI_MODEL", "GEMINI_CHAT_MODEL", "PKP_OPENAI__MODEL"),
        )
        openai_embedding_model = remember(
            "openai_embedding_model",
            _first_env(
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

        ollama_base_url = remember("ollama_base_url", _first_env("OLLAMA_BASE_URL", "PKP_OLLAMA__BASE_URL"))
        ollama_chat_model = remember("ollama_chat_model", _first_env("OLLAMA_CHAT_MODEL", "PKP_OLLAMA__CHAT_MODEL"))
        ollama_embedding_model = remember(
            "ollama_embedding_model",
            _first_env("OLLAMA_EMBEDDING_MODEL", "PKP_OLLAMA__EMBEDDING_MODEL"),
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
            _first_env("PKP_LOCAL_BGE__ENABLED", "RAG_LOCAL_BGE_ENABLED"),
        )
        local_bge_embedding_model = remember(
            "local_bge_embedding_model",
            _first_env("PKP_LOCAL_BGE__EMBEDDING_MODEL", "RAG_LOCAL_BGE_EMBEDDING_MODEL"),
        )
        local_bge_embedding_model_path = remember(
            "local_bge_embedding_model_path",
            _first_env("PKP_LOCAL_BGE__EMBEDDING_MODEL_PATH", "RAG_LOCAL_BGE_EMBEDDING_MODEL_PATH"),
        )
        local_bge_rerank_model = remember(
            "local_bge_rerank_model",
            _first_env("RAG_RERANK_MODEL", "PKP_LOCAL_BGE__RERANK_MODEL"),
        )
        local_bge_rerank_model_path = remember(
            "local_bge_rerank_model_path",
            _first_env("RAG_RERANK_MODEL_PATH", "PKP_LOCAL_BGE__RERANK_MODEL_PATH"),
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
                    rerank_model=None if local_bge_rerank_model is None else str(local_bge_rerank_model),
                    rerank_model_path=None if local_bge_rerank_model_path is None else str(local_bge_rerank_model_path),
                )
            )

        tokenizer_config = TokenizerConfig(
            embedding_model_name=_first_env(
                "RAG_EMBEDDING_MODEL",
                "RAG_INDEX_EMBEDDING_MODEL",
            ),
            tokenizer_model_name=_first_env(
                "RAG_TOKENIZER_MODEL",
                "RAG_BUDGET_TOKENIZER_MODEL",
            ),
            chunking_tokenizer_model_name=_first_env(
                "RAG_CHUNKING_TOKENIZER_MODEL",
                "RAG_DOCLING_TOKENIZER_MODEL",
            ),
            tokenizer_backend=_first_env("RAG_TOKENIZER_BACKEND"),
            chunk_token_size=_env_int("RAG_CHUNK_TOKEN_SIZE"),
            chunk_overlap_tokens=_env_int("RAG_CHUNK_OVERLAP_TOKENS"),
            max_context_tokens=_env_int("RAG_MAX_CONTEXT_TOKENS"),
            prompt_reserved_tokens=_env_int("RAG_PROMPT_RESERVED_TOKENS"),
            local_files_only=_env_bool("RAG_TOKENIZER_LOCAL_FILES_ONLY"),
        )
        return AssemblyConfig(profiles=tuple(profiles), tokenizer=tokenizer_config), compatibility_inputs

    def _assembly_profiles(
        self,
        *,
        config: AssemblyConfig | None,
        compatibility_config: AssemblyConfig,
    ) -> tuple[AssemblyProfileSpec, ...]:
        del config
        compat_openai = self._provider_config_by_id(compatibility_config.profiles, "openai-compatible")
        compat_ollama = self._provider_config_by_id(compatibility_config.profiles, "ollama")
        compat_local_bge = self._provider_config_by_id(compatibility_config.profiles, "local-bge")
        return (
            AssemblyProfileSpec(
                profile_id="local_full",
                label="Local Full",
                description="Prefer local retrieval and local chat. Uses local BGE for retrieval and Ollama for chat.",
                location="local",
                overrides=AssemblyOverrides(
                    embedding=self._local_retrieval_provider(compat_local_bge, compat_ollama),
                    rerank=self._rerank_provider(compat_local_bge),
                    chat=self._chat_provider(
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
                    embedding=self._local_retrieval_provider(compat_local_bge, compat_ollama),
                    rerank=self._rerank_provider(compat_local_bge),
                    chat=self._chat_provider(
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
                    embedding=self._embedding_provider(
                        compat_openai,
                        fallback_kind="openai-compatible",
                        fallback_location="cloud",
                        fallback_base_url="https://api.openai.com/v1",
                    ),
                    chat=self._chat_provider(
                        compat_openai,
                        fallback_kind="openai-compatible",
                        fallback_location="cloud",
                        fallback_base_url="https://api.openai.com/v1",
                    ),
                    rerank=self._rerank_provider(compat_local_bge),
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

    def _catalog_profiles(
        self,
        *,
        config: AssemblyConfig | None,
        compatibility_config: AssemblyConfig,
    ) -> tuple[CapabilityProfile, ...]:
        profiles: list[CapabilityProfile] = []
        for provider_config in [*(config.profiles if config is not None else ()), *compatibility_config.profiles]:
            profile_id = provider_config.profile_id
            if not profile_id or not provider_config.enabled:
                continue
            profiles.append(self._profile_from_provider_config(provider_config))
        return tuple(profiles)

    @staticmethod
    def _assembly_profile_by_id(
        profiles: Sequence[AssemblyProfileSpec],
        profile_id: str | None,
    ) -> AssemblyProfileSpec | None:
        if profile_id is None:
            return None
        for profile in profiles:
            if profile.profile_id == profile_id:
                return profile
        return None

    @staticmethod
    def _provider_config_by_id(
        profiles: Sequence[ProviderConfig],
        profile_id: str,
    ) -> ProviderConfig | None:
        for profile in profiles:
            if profile.profile_id == profile_id:
                return profile
        return None

    @staticmethod
    def _merge_provider_config(
        high: ProviderConfig | None,
        low: ProviderConfig | None,
    ) -> ProviderConfig | None:
        if high is None:
            return low
        if low is None:
            return high
        return ProviderConfig(
            provider_kind=high.provider_kind or low.provider_kind,
            location=_first_non_blank(high.location, low.location) or low.location,
            profile_id=_first_non_blank(high.profile_id, low.profile_id),
            label=_first_non_blank(high.label, low.label),
            api_key=_first_non_blank(high.api_key, low.api_key),
            base_url=_first_non_blank(high.base_url, low.base_url),
            chat_model=_first_non_blank(high.chat_model, low.chat_model),
            embedding_model=_first_non_blank(high.embedding_model, low.embedding_model),
            rerank_model=_first_non_blank(high.rerank_model, low.rerank_model),
            embedding_model_path=_first_non_blank(high.embedding_model_path, low.embedding_model_path),
            rerank_model_path=_first_non_blank(high.rerank_model_path, low.rerank_model_path),
            enabled=high.enabled if high is not None else low.enabled,
        )

    @staticmethod
    def _merge_tokenizer_config(
        high: TokenizerConfig | None,
        low: TokenizerConfig | None,
    ) -> TokenizerConfig | None:
        if high is None:
            return low
        if low is None:
            return high
        return TokenizerConfig(
            embedding_model_name=_first_non_blank(high.embedding_model_name, low.embedding_model_name),
            tokenizer_model_name=_first_non_blank(high.tokenizer_model_name, low.tokenizer_model_name),
            chunking_tokenizer_model_name=_first_non_blank(
                high.chunking_tokenizer_model_name,
                low.chunking_tokenizer_model_name,
            ),
            tokenizer_backend=_first_non_blank(high.tokenizer_backend, low.tokenizer_backend),
            chunk_token_size=high.chunk_token_size if high.chunk_token_size is not None else low.chunk_token_size,
            chunk_overlap_tokens=(
                high.chunk_overlap_tokens
                if high.chunk_overlap_tokens is not None
                else low.chunk_overlap_tokens
            ),
            max_context_tokens=(
                high.max_context_tokens
                if high.max_context_tokens is not None
                else low.max_context_tokens
            ),
            prompt_reserved_tokens=(
                high.prompt_reserved_tokens
                if high.prompt_reserved_tokens is not None
                else low.prompt_reserved_tokens
            ),
            local_files_only=high.local_files_only if high.local_files_only is not None else low.local_files_only,
        )

    def _merge_assembly_config(
        self,
        high: AssemblyConfig | None,
        low: AssemblyConfig | None,
    ) -> AssemblyConfig | None:
        if high is None:
            return low
        if low is None:
            return high
        return AssemblyConfig(
            default_profile_id=_first_non_blank(high.default_profile_id, low.default_profile_id),
            profiles=tuple([*high.profiles, *low.profiles]),
            chat=self._merge_provider_config(high.chat, low.chat),
            embedding=self._merge_provider_config(high.embedding, low.embedding),
            rerank=self._merge_provider_config(high.rerank, low.rerank),
            tokenizer=self._merge_tokenizer_config(high.tokenizer, low.tokenizer),
        )

    def _merge_assembly_overrides(
        self,
        high: AssemblyOverrides | None,
        low: AssemblyOverrides | None,
    ) -> AssemblyOverrides | None:
        if high is None:
            return low
        if low is None:
            return high
        return AssemblyOverrides(
            chat=self._merge_provider_config(high.chat, low.chat),
            embedding=self._merge_provider_config(high.embedding, low.embedding),
            rerank=self._merge_provider_config(high.rerank, low.rerank),
            tokenizer=self._merge_tokenizer_config(high.tokenizer, low.tokenizer),
        )

    def _local_retrieval_provider(
        self,
        local_bge: ProviderConfig | None,
        ollama: ProviderConfig | None,
    ) -> ProviderConfig:
        if local_bge is not None:
            return replace(local_bge, profile_id=None, label=None)
        if ollama is not None and self._preview_supports(ollama, "embedding"):
            return replace(ollama, profile_id=None, label=None)
        return ProviderConfig(provider_kind="local-bge", location="local")

    def _rerank_provider(self, local_bge: ProviderConfig | None) -> ProviderConfig | None:
        if local_bge is None:
            return None
        return replace(local_bge, profile_id=None, label=None)

    def _embedding_provider(
        self,
        provider: ProviderConfig | None,
        *,
        fallback_kind: str,
        fallback_location: str,
        fallback_base_url: str | None = None,
    ) -> ProviderConfig:
        if provider is not None:
            return replace(provider, profile_id=None, label=None)
        return ProviderConfig(
            provider_kind=fallback_kind,
            location=fallback_location,
            base_url=fallback_base_url,
        )

    def _chat_provider(
        self,
        provider: ProviderConfig | None,
        *,
        fallback_kind: str,
        fallback_location: str,
        fallback_base_url: str | None = None,
    ) -> ProviderConfig:
        if provider is not None:
            return replace(provider, profile_id=None, label=None)
        return ProviderConfig(
            provider_kind=fallback_kind,
            location=fallback_location,
            base_url=fallback_base_url,
        )

    def _profile_from_provider_config(self, provider_config: ProviderConfig) -> CapabilityProfile:
        profile_id = provider_config.profile_id or provider_config.provider_kind
        model_label = (
            provider_config.chat_model
            or provider_config.embedding_model
            or provider_config.rerank_model
            or "default"
        )
        label = provider_config.label or (
            f"{provider_config.provider_kind} / "
            f"{model_label}"
        )

        def factory() -> object:
            return self._build_provider(provider_config)

        provider_preview = self._build_provider(provider_config)
        return CapabilityProfile(
            profile_id=profile_id,
            label=label,
            provider_kind=provider_config.provider_kind,
            location=provider_config.location,
            chat_model=_provider_model(provider_preview, "chat"),
            embedding_model=_provider_model(provider_preview, "embedding"),
            rerank_model=_provider_model(provider_preview, "rerank"),
            supports_chat=_supports_capability(provider_preview, "chat"),
            supports_embedding=_supports_capability(provider_preview, "embedding"),
            supports_rerank=_supports_capability(provider_preview, "rerank"),
            provider_config=provider_config,
            factory=factory,
        )

    def _resolve_embedding_binding(
        self,
        *,
        request: AssemblyRequest,
        profiles: Sequence[CapabilityProfile],
        explicit_profile: CapabilityProfile | None,
        config_profile: CapabilityProfile | None,
        compatibility_config: AssemblyConfig,
        compatibility_inputs: dict[str, str | int | bool | None],
        provider_cache: dict[str, object],
        issues: list[AssemblyIssue],
        decisions: list[AssemblyDecision],
    ) -> EmbeddingCapabilityBinding | None:
        candidates = self._capability_candidates(
            capability="embedding",
            request=request,
            profiles=profiles,
            explicit_profile=explicit_profile,
            config_profile=config_profile,
            compatibility_config=compatibility_config,
        )
        return cast(
            EmbeddingCapabilityBinding | None,
            self._choose_capability_binding(
                capability="embedding",
                candidates=candidates,
                provider_cache=provider_cache,
                issues=issues,
                decisions=decisions,
                required=request.requirements.require_embedding,
                allow_degraded=request.requirements.allow_degraded,
                compatibility_inputs=compatibility_inputs,
                default_space="default",
            ),
        )

    def _resolve_chat_binding(
        self,
        *,
        request: AssemblyRequest,
        profiles: Sequence[CapabilityProfile],
        explicit_profile: CapabilityProfile | None,
        config_profile: CapabilityProfile | None,
        compatibility_config: AssemblyConfig,
        provider_cache: dict[str, object],
        issues: list[AssemblyIssue],
        decisions: list[AssemblyDecision],
    ) -> ChatCapabilityBinding | None:
        candidates = self._capability_candidates(
            capability="chat",
            request=request,
            profiles=profiles,
            explicit_profile=explicit_profile,
            config_profile=config_profile,
            compatibility_config=compatibility_config,
        )
        return cast(
            ChatCapabilityBinding | None,
            self._choose_capability_binding(
                capability="chat",
                candidates=candidates,
                provider_cache=provider_cache,
                issues=issues,
                decisions=decisions,
                required=request.requirements.require_chat,
                allow_degraded=request.requirements.allow_degraded,
            ),
        )

    def _resolve_rerank_binding(
        self,
        *,
        request: AssemblyRequest,
        profiles: Sequence[CapabilityProfile],
        explicit_profile: CapabilityProfile | None,
        config_profile: CapabilityProfile | None,
        compatibility_config: AssemblyConfig,
        provider_cache: dict[str, object],
        issues: list[AssemblyIssue],
        decisions: list[AssemblyDecision],
    ) -> RerankCapabilityBinding | None:
        candidates = self._capability_candidates(
            capability="rerank",
            request=request,
            profiles=profiles,
            explicit_profile=explicit_profile,
            config_profile=config_profile,
            compatibility_config=compatibility_config,
        )
        return cast(
            RerankCapabilityBinding | None,
            self._choose_capability_binding(
                capability="rerank",
                candidates=candidates,
                provider_cache=provider_cache,
                issues=issues,
                decisions=decisions,
                required=request.requirements.require_rerank,
                allow_degraded=request.requirements.allow_degraded,
            ),
        )

    def _capability_candidates(
        self,
        *,
        capability: str,
        request: AssemblyRequest,
        profiles: Sequence[CapabilityProfile],
        explicit_profile: CapabilityProfile | None,
        config_profile: CapabilityProfile | None,
        compatibility_config: AssemblyConfig,
    ) -> list[_CandidateSource]:
        candidates: list[_CandidateSource] = []
        overrides = request.overrides or AssemblyOverrides()
        config = request.config or AssemblyConfig()

        explicit_spec = getattr(overrides, capability)
        if explicit_spec is not None and explicit_spec.enabled:
            candidates.append(
                _CandidateSource(
                    source="explicit",
                    provider_config=explicit_spec,
                    cache_key=f"explicit:{capability}",
                    reason=f"Using explicit {capability} override.",
                )
            )

        if explicit_profile is not None and self._profile_supports(explicit_profile, capability):
            candidates.append(
                _CandidateSource(
                    source="profile",
                    provider_config=explicit_profile.provider_config,
                    cache_key=f"profile:{explicit_profile.profile_id}",
                    reason=f"Using requested profile {explicit_profile.profile_id!r} for {capability}.",
                )
            )

        config_spec = getattr(config, capability)
        if config_spec is not None and config_spec.enabled:
            candidates.append(
                _CandidateSource(
                    source="config",
                    provider_config=config_spec,
                    cache_key=f"config:{capability}",
                    reason=f"Using structured config for {capability}.",
                )
            )

        if config_profile is not None and self._profile_supports(config_profile, capability):
            candidates.append(
                _CandidateSource(
                    source="config",
                    provider_config=config_profile.provider_config,
                    cache_key=f"profile-config:{config_profile.profile_id}",
                    reason=f"Using config default profile {config_profile.profile_id!r} for {capability}.",
                )
            )

        compat_profile = self._first_profile_supporting(compatibility_config.profiles, capability)
        if compat_profile is not None:
            candidates.append(
                _CandidateSource(
                    source="compat_env",
                    provider_config=compat_profile,
                    cache_key=f"compat:{capability}:{compat_profile.profile_id or compat_profile.provider_kind}",
                    reason=f"Using compatibility environment config for {capability}.",
                )
            )
        return candidates

    def _choose_capability_binding(
        self,
        *,
        capability: str,
        candidates: Sequence[_CandidateSource],
        provider_cache: dict[str, object],
        issues: list[AssemblyIssue],
        decisions: list[AssemblyDecision],
        required: bool,
        allow_degraded: bool,
        compatibility_inputs: dict[str, str | int | bool | None] | None = None,
        default_space: str = "default",
    ) -> CapabilityBinding | None:
        del compatibility_inputs
        for candidate in candidates:
            provider = self._provider_from_cache(candidate, provider_cache)
            if not _supports_capability(provider, capability):
                issues.append(
                    AssemblyIssue(
                        severity="warning",
                        code=f"{capability}_candidate_unusable",
                        message=(
                            f"{candidate.reason} Provider {candidate.provider_config.provider_kind!r} does not "
                            f"provide a usable {capability} capability."
                        ),
                    )
                )
                decisions.append(
                    AssemblyDecision(
                        capability=capability,
                        source=candidate.source,
                        provider_kind=candidate.provider_config.provider_kind,
                        provider_name=_provider_name(provider),
                        model_name=_provider_model(provider, capability),
                        location=candidate.provider_config.location,
                        reason=f"{candidate.reason} Candidate was rejected because the capability is unavailable.",
                        selected=False,
                    )
                )
                continue
            if capability == "embedding":
                embedding_binding = EmbeddingCapabilityBinding(
                    backend=provider,
                    space=default_space,
                    location=candidate.provider_config.location,
                )
                binding: CapabilityBinding = embedding_binding
            elif capability == "chat":
                chat_binding = ChatCapabilityBinding(
                    backend=provider,
                    location=candidate.provider_config.location,
                )
                binding = chat_binding
            else:
                rerank_binding = RerankCapabilityBinding(
                    backend=provider,
                    location=candidate.provider_config.location,
                )
                binding = rerank_binding
            decisions.append(
                AssemblyDecision(
                    capability=capability,
                    source=candidate.source,
                    provider_kind=candidate.provider_config.provider_kind,
                    provider_name=getattr(binding, "provider_name", None),
                    model_name=getattr(binding, "model_name", None),
                    location=candidate.provider_config.location,
                    reason=candidate.reason,
                )
            )
            return binding

        if required:
            issues.append(
                AssemblyIssue(
                    severity="warning" if allow_degraded else "error",
                    code=f"missing_required_{capability}",
                    message=f"No usable {capability} capability could be assembled.",
                )
            )
        return None

    def _build_token_contract(
        self,
        *,
        request: AssemblyRequest,
        embedding_binding: EmbeddingCapabilityBinding,
        compatibility_config: AssemblyConfig,
        compatibility_inputs: dict[str, str | int | bool | None],
        issues: list[AssemblyIssue],
        decisions: list[AssemblyDecision],
    ) -> TokenizerContract:
        explicit_tokenizer = request.overrides.tokenizer if request.overrides is not None else None
        config_tokenizer = request.config.tokenizer if request.config is not None else None
        compat_tokenizer = compatibility_config.tokenizer

        embedding_model = embedding_binding.model_name or DEFAULT_TOKENIZER_FALLBACK_MODEL
        explicit_embedding_override = (
            explicit_tokenizer.embedding_model_name
            if explicit_tokenizer is not None
            else None
        )
        config_embedding_override = config_tokenizer.embedding_model_name if config_tokenizer is not None else None
        compat_embedding_override = compat_tokenizer.embedding_model_name if compat_tokenizer is not None else None
        locked_embedding_model = _first_non_blank(
            explicit_embedding_override,
            config_embedding_override,
            compat_embedding_override,
        )
        if locked_embedding_model is not None and locked_embedding_model != embedding_model:
            issues.append(
                AssemblyIssue(
                    severity="error",
                    code="embedding_contract_mismatch",
                    message=(
                        "Configured embedding model does not match the selected embedding capability: "
                        f"{locked_embedding_model!r} != {embedding_model!r}."
                    ),
                )
            )

        tokenizer_model = _first_non_blank(
            explicit_tokenizer.tokenizer_model_name if explicit_tokenizer is not None else None,
            config_tokenizer.tokenizer_model_name if config_tokenizer is not None else None,
            compat_tokenizer.tokenizer_model_name if compat_tokenizer is not None else None,
            embedding_model,
            DEFAULT_TOKENIZER_FALLBACK_MODEL,
        )
        chunking_tokenizer_model = _first_non_blank(
            explicit_tokenizer.chunking_tokenizer_model_name if explicit_tokenizer is not None else None,
            config_tokenizer.chunking_tokenizer_model_name if config_tokenizer is not None else None,
            compat_tokenizer.chunking_tokenizer_model_name if compat_tokenizer is not None else None,
            tokenizer_model,
            DEFAULT_TOKENIZER_FALLBACK_MODEL,
        )
        tokenizer_backend = (
            _first_non_blank(
                explicit_tokenizer.tokenizer_backend if explicit_tokenizer is not None else None,
                config_tokenizer.tokenizer_backend if config_tokenizer is not None else None,
                compat_tokenizer.tokenizer_backend if compat_tokenizer is not None else None,
            )
            or "auto"
        )
        chunk_token_size = max(
            32,
            self._first_positive_int(
                explicit_tokenizer.chunk_token_size if explicit_tokenizer is not None else None,
                config_tokenizer.chunk_token_size if config_tokenizer is not None else None,
                compat_tokenizer.chunk_token_size if compat_tokenizer is not None else None,
                request.requirements.default_chunk_token_size,
            ),
        )
        chunk_overlap_tokens = max(
            0,
            self._first_non_negative_int(
                explicit_tokenizer.chunk_overlap_tokens if explicit_tokenizer is not None else None,
                config_tokenizer.chunk_overlap_tokens if config_tokenizer is not None else None,
                compat_tokenizer.chunk_overlap_tokens if compat_tokenizer is not None else None,
                request.requirements.default_chunk_overlap_tokens,
            ),
        )
        max_context_tokens = max(
            64,
            self._first_positive_int(
                explicit_tokenizer.max_context_tokens if explicit_tokenizer is not None else None,
                config_tokenizer.max_context_tokens if config_tokenizer is not None else None,
                compat_tokenizer.max_context_tokens if compat_tokenizer is not None else None,
                request.requirements.default_context_tokens,
            ),
        )
        prompt_reserved_tokens = max(
            32,
            self._first_positive_int(
                explicit_tokenizer.prompt_reserved_tokens if explicit_tokenizer is not None else None,
                config_tokenizer.prompt_reserved_tokens if config_tokenizer is not None else None,
                compat_tokenizer.prompt_reserved_tokens if compat_tokenizer is not None else None,
                request.requirements.default_prompt_reserved_tokens,
            ),
        )
        local_files_only = self._first_bool(
            explicit_tokenizer.local_files_only if explicit_tokenizer is not None else None,
            config_tokenizer.local_files_only if config_tokenizer is not None else None,
            compat_tokenizer.local_files_only if compat_tokenizer is not None else None,
            True,
        )
        source: DecisionSource = "default"
        if explicit_tokenizer is not None:
            source = "explicit"
        elif config_tokenizer is not None:
            source = "config"
        elif compat_tokenizer is not None:
            source = "compat_env"
        decisions.append(
            AssemblyDecision(
                capability="tokenizer_contract",
                source=source,
                provider_kind="tokenizer-contract",
                provider_name="tokenizer-contract",
                model_name=tokenizer_model,
                location="assembly",
                reason="Tokenizer contract was assembled through the unified contract governance chain.",
            )
        )
        compatibility_inputs["resolved_embedding_model_name"] = embedding_model
        return TokenizerContract(
            embedding_model_name=embedding_model,
            tokenizer_model_name=tokenizer_model or DEFAULT_TOKENIZER_FALLBACK_MODEL,
            chunking_tokenizer_model_name=(
                chunking_tokenizer_model
                or tokenizer_model
                or DEFAULT_TOKENIZER_FALLBACK_MODEL
            ),
            tokenizer_backend=tokenizer_backend,
            chunk_token_size=chunk_token_size,
            chunk_overlap_tokens=chunk_overlap_tokens,
            max_context_tokens=max_context_tokens,
            prompt_reserved_tokens=prompt_reserved_tokens,
            local_files_only=local_files_only,
        )

    @staticmethod
    def _runtime_contract_payload(
        token_contract: TokenizerContract,
        token_accounting: TokenAccountingService,
    ) -> dict[str, str | int | bool]:
        tokenizer_backend, _tokenizer_model = token_accounting.backend_descriptor()
        return {
            "embedding_model_name": token_contract.embedding_model_name,
            "tokenizer_model_name": token_contract.tokenizer_model_name,
            "chunking_tokenizer_model_name": token_contract.chunking_tokenizer_model_name,
            "tokenizer_backend": tokenizer_backend,
            "chunk_token_size": token_contract.chunk_token_size,
            "chunk_overlap_tokens": token_contract.normalized_chunk_overlap_tokens(),
        }

    @staticmethod
    def _diagnostics_status(issues: Sequence[AssemblyIssue]) -> AssemblyStatus:
        if any(issue.severity == "error" for issue in issues):
            return "invalid"
        if any(issue.severity == "warning" for issue in issues):
            return "degraded"
        return "valid"

    @staticmethod
    def _profile_supports(profile: CapabilityProfile, capability: str) -> bool:
        return {
            "chat": profile.supports_chat,
            "embedding": profile.supports_embedding,
            "rerank": profile.supports_rerank,
        }[capability]

    @staticmethod
    def _first_profile_supporting(profiles: Sequence[ProviderConfig], capability: str) -> ProviderConfig | None:
        for provider_config in profiles:
            if not provider_config.enabled:
                continue
            preview = CapabilityAssemblyService._preview_supports(provider_config, capability)
            if preview:
                return provider_config
        return None

    @staticmethod
    def _preview_supports(provider_config: ProviderConfig, capability: str) -> bool:
        if provider_config.provider_kind == "openai-compatible":
            if capability == "chat":
                return bool(provider_config.api_key and provider_config.chat_model)
            if capability == "embedding":
                return bool(provider_config.api_key and provider_config.embedding_model)
            return False
        if provider_config.provider_kind == "ollama":
            if capability == "chat":
                return bool(provider_config.base_url and provider_config.chat_model)
            if capability == "embedding":
                return bool(provider_config.base_url and provider_config.embedding_model)
            return False
        if provider_config.provider_kind == "local-bge":
            if capability == "embedding":
                return bool(provider_config.embedding_model or provider_config.embedding_model_path)
            if capability == "rerank":
                return bool(provider_config.rerank_model or provider_config.rerank_model_path)
            return False
        if provider_config.provider_kind in {"fallback", "default-embedding"}:
            return capability == "embedding"
        return False

    @staticmethod
    def _profile_by_id(profiles: Sequence[CapabilityProfile], profile_id: str | None) -> CapabilityProfile | None:
        if profile_id is None:
            return None
        for profile in profiles:
            if profile.profile_id == profile_id:
                return profile
        return None

    def _provider_from_cache(self, candidate: _CandidateSource, provider_cache: dict[str, object]) -> object:
        provider = provider_cache.get(candidate.cache_key)
        if provider is None:
            provider = self._build_provider(candidate.provider_config)
            provider_cache[candidate.cache_key] = provider
        return provider

    def _build_provider(self, provider_config: ProviderConfig) -> object:
        kind = provider_config.provider_kind
        if kind == "openai-compatible":
            return OpenAIProviderRepo(
                api_key=provider_config.api_key,
                base_url=_normalize_gemini_base_url(provider_config.base_url) or "https://api.openai.com/v1",
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
            )
        if kind in {"fallback", "default-embedding"}:
            return FallbackEmbeddingRepo()
        raise RuntimeError(f"Unsupported provider_kind: {kind}")

    @staticmethod
    def _first_positive_int(*values: int | None) -> int:
        for value in values:
            if isinstance(value, int) and value > 0:
                return value
        return 1

    @staticmethod
    def _first_non_negative_int(*values: int | None) -> int:
        for value in values:
            if isinstance(value, int) and value >= 0:
                return value
        return 0

    @staticmethod
    def _first_bool(*values: bool | None) -> bool:
        for value in values:
            if isinstance(value, bool):
                return value
        return False


__all__ = [
    "AssemblyConfig",
    "AssemblyContracts",
    "AssemblyDecision",
    "AssemblyDiagnostics",
    "AssemblyIssue",
    "AssemblyOverrides",
    "AssemblyProfileSpec",
    "AssemblyRequest",
    "CapabilityAssemblyService",
    "CapabilityBundle",
    "CapabilityCatalog",
    "CapabilityProfile",
    "CapabilityRequirements",
    "ChatCapabilityBinding",
    "EmbeddingCapabilityBinding",
    "ProviderConfig",
    "RerankCapabilityBinding",
    "RuntimeContractGovernance",
    "TokenizerConfig",
]
