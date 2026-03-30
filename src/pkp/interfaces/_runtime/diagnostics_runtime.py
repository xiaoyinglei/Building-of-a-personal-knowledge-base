from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from pkp.interfaces._runtime.provider_metadata import capability_configured, provider_model, provider_name
from pkp.schema._types import CapabilityHealth, HealthReport, IndexHealth, ProviderHealth


@dataclass(frozen=True)
class ProviderBinding:
    provider: object
    location: str


@dataclass(frozen=True)
class _ProbeCandidate:
    chunk_id: str
    text: str
    score: float
    section_path: tuple[str, ...] = ()


class DiagnosticsRuntime:
    def __init__(
        self,
        *,
        providers: Sequence[ProviderBinding] = (),
        metadata_repo: object | None = None,
        vector_repo: object | None = None,
    ) -> None:
        self._providers = tuple(providers)
        self._metadata_repo = metadata_repo
        self._vector_repo = vector_repo

    def report(self) -> HealthReport:
        providers = [self._probe_provider(binding) for binding in self._providers]
        indices = self._index_health()
        status = self._overall_status(providers, indices)
        return HealthReport(status=status, providers=providers, indices=indices)

    def _probe_provider(self, binding: ProviderBinding) -> ProviderHealth:
        capability_names = []
        if callable(getattr(binding.provider, "embed", None)):
            capability_names.append("embed")
        if callable(getattr(binding.provider, "chat", None)):
            capability_names.append("chat")
        if callable(getattr(binding.provider, "rerank", None)):
            capability_names.append("rerank")

        capabilities = {
            capability: self._probe_capability(binding.provider, capability)
            for capability in capability_names
            if capability_configured(binding.provider, capability)
        }
        return ProviderHealth(
            provider=provider_name(binding.provider),
            location=binding.location,
            capabilities=capabilities,
        )

    def _probe_capability(self, provider: object, capability: str) -> CapabilityHealth:
        model = provider_model(provider, capability)
        configured = capability_configured(provider, capability)
        if not configured:
            return CapabilityHealth(configured=False, available=False, model=model)

        try:
            if capability == "embed":
                embed = cast(object, getattr(provider, "embed", None))
                if not callable(embed):
                    return CapabilityHealth(configured=True, available=False, model=model)
                result = embed(["health probe"])
                available = bool(result)
            elif capability == "chat":
                chat = cast(object, getattr(provider, "chat", None))
                if not callable(chat):
                    return CapabilityHealth(configured=True, available=False, model=model)
                result = chat("Reply with OK.")
                available = bool(str(result).strip())
            else:
                rerank = cast(object, getattr(provider, "rerank", None))
                if not callable(rerank):
                    return CapabilityHealth(configured=True, available=False, model=model)
                result = rerank(
                    "health probe",
                    [
                        _ProbeCandidate(chunk_id="probe-a", text="candidate-a", score=1.0),
                        _ProbeCandidate(chunk_id="probe-b", text="candidate-b", score=0.5),
                    ],
                )
                available = isinstance(result, list) and len(result) == 2
            return CapabilityHealth(configured=True, available=available, model=model)
        except RuntimeError as exc:
            return CapabilityHealth(
                configured=True,
                available=False,
                model=model,
                error=str(exc),
            )

    def _index_health(self) -> IndexHealth:
        documents = 0
        chunks = 0
        vectors = 0
        missing_vectors = 0

        list_documents = getattr(self._metadata_repo, "list_documents", None)
        list_chunks = getattr(self._metadata_repo, "list_chunks", None)
        existing_item_ids = getattr(self._vector_repo, "existing_item_ids", None)
        vector_count = getattr(self._vector_repo, "count_vectors", None)

        if callable(list_documents):
            documents_list = list_documents(active_only=True)
            documents = len(documents_list)
            if callable(list_chunks):
                chunk_ids: list[str] = []
                for document in documents_list:
                    doc_chunks = list_chunks(document.doc_id)
                    chunks += len(doc_chunks)
                    chunk_ids.extend(chunk.chunk_id for chunk in doc_chunks)
                if callable(existing_item_ids):
                    vectors = len(existing_item_ids(chunk_ids))
                elif callable(vector_count):
                    vectors = int(vector_count(distinct_chunks=True))
                missing_vectors = max(0, chunks - vectors)

        return IndexHealth(
            documents=documents,
            chunks=chunks,
            vectors=vectors,
            missing_vectors=missing_vectors,
        )

    @staticmethod
    def _overall_status(providers: Sequence[ProviderHealth], indices: IndexHealth) -> str:
        any_unavailable_capability = any(
            capability.configured and not capability.available
            for provider in providers
            for capability in provider.capabilities.values()
        )
        if any_unavailable_capability or indices.missing_vectors > 0:
            return "degraded"
        return "ok"
