from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from rag.llm._rerank.cross_encoder import CrossEncoderConfig, ProviderBackedCrossEncoder
from rag.llm._rerank.pipeline import FormalRerankService, RerankPipelineConfig
from rag.llm.assembly import RerankCapabilityBinding
from rag.query.analysis import QueryUnderstandingService


class CandidateLike(Protocol):
    chunk_id: str
    text: str
    score: float
    section_path: Sequence[str]
    special_chunk_type: str | None
    chunk_role: object | None
    metadata: dict[str, str] | None


class ModelBackedRerankService:
    def __init__(
        self,
        *,
        binding: RerankCapabilityBinding | None = None,
        provider: object | None = None,
        config: RerankPipelineConfig | None = None,
        query_understanding_service: QueryUnderstandingService | None = None,
    ) -> None:
        resolved_config = config or RerankPipelineConfig()
        self._binding = binding
        resolved_provider = binding.backend if binding is not None else provider
        self._cross_encoder = ProviderBackedCrossEncoder(
            provider=resolved_provider,
            config=resolved_config.cross_encoder,
        )
        self._pipeline = FormalRerankService(
            cross_encoder=self._cross_encoder,
            config=resolved_config,
            query_understanding_service=query_understanding_service,
        )

    @property
    def last_response(self) -> object | None:
        return self._pipeline.last_response

    @property
    def provider_name(self) -> str:
        if self._binding is not None:
            return self._binding.provider_name
        response = self._pipeline.last_response
        if response is not None:
            backend_name = getattr(response, "backend_name", None)
            if isinstance(backend_name, str) and backend_name:
                return backend_name
        return "formal-rerank"

    @property
    def rerank_model_name(self) -> str:
        if self._binding is not None and self._binding.model_name:
            return self._binding.model_name
        response = self._pipeline.last_response
        if response is not None:
            model_name = getattr(response, "model_name", None)
            if isinstance(model_name, str) and model_name:
                return model_name
        configured_model_name = getattr(self._cross_encoder, "model_name", None)
        if isinstance(configured_model_name, str) and configured_model_name:
            return configured_model_name
        return "unconfigured-reranker"

    def rerank(self, query: str, candidates: Sequence[CandidateLike]) -> list[CandidateLike]:
        return cast(list[CandidateLike], self._pipeline.rerank(query, list(candidates)))


CrossEncoderReranker = ProviderBackedCrossEncoder
FormalRerankPipeline = FormalRerankService

__all__ = [
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "FormalRerankPipeline",
    "FormalRerankService",
    "ModelBackedRerankService",
    "ProviderBackedCrossEncoder",
    "RerankPipelineConfig",
]
