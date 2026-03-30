from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.llm._rerank.cross_encoder import CrossEncoderConfig
from rag.llm._rerank.models import FeatureRecord, RerankCandidate, RerankRequest, RerankResponse


class CrossEncoderProtocol(Protocol):
    backend_name: str
    model_name: str

    def score(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        config: CrossEncoderConfig,
    ) -> list[float]: ...


class ScoreCombinerProtocol(Protocol):
    def combine(
        self,
        *,
        candidate: RerankCandidate,
        feature_record: FeatureRecord,
    ) -> tuple[float, dict[str, float | int | bool | str]]: ...


class LearnedFusionRanker(Protocol):
    def score(
        self,
        *,
        request: RerankRequest,
        features: Sequence[FeatureRecord],
    ) -> list[float]: ...


class LLMRerankExtension(Protocol):
    def refine(
        self,
        *,
        request: RerankRequest,
        response: RerankResponse,
    ) -> RerankResponse: ...


class DistillationSink(Protocol):
    def record(self, *, request: RerankRequest, response: RerankResponse) -> None: ...


class FeedbackSink(Protocol):
    def record(self, *, request: RerankRequest, response: RerankResponse) -> None: ...


class ParentContextAssembler(Protocol):
    def build_context(self, candidate: RerankCandidate, query: str) -> str: ...
