from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from pkp.rerank.cross_encoder import ProviderBackedCrossEncoder
from pkp.rerank.pipeline import FormalRerankService, RerankPipelineConfig
from pkp.service.query_understanding_service import QueryUnderstandingService


class CandidateLike(Protocol):
    chunk_id: str
    text: str
    score: float
    section_path: Sequence[str]
    special_chunk_type: str | None
    chunk_role: object | None
    metadata: dict[str, str] | None


class HeuristicRerankService:
    def __init__(
        self,
        *,
        query_understanding_service: QueryUnderstandingService | None = None,
        provider: object | None = None,
        config: RerankPipelineConfig | None = None,
    ) -> None:
        self._query_understanding_service = query_understanding_service or QueryUnderstandingService()
        resolved_config = config or RerankPipelineConfig()
        self._pipeline = FormalRerankService(
            cross_encoder=ProviderBackedCrossEncoder(provider=provider, config=resolved_config.cross_encoder),
            config=resolved_config,
        )

    @property
    def last_response(self) -> object | None:
        return self._pipeline.last_response

    def rerank(self, query: str, candidates: Sequence[CandidateLike]) -> list[CandidateLike]:
        return cast(list[CandidateLike], self._pipeline.rerank(query, list(candidates)))
