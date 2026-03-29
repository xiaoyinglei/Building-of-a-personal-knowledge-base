from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from pkp.query.context import QueryUnderstandingService
from pkp.rerank.cross_encoder import CrossEncoderConfig, ProviderBackedCrossEncoder
from pkp.rerank.features import RerankFeatureExtractor
from pkp.rerank.fusion import FeatureBasedScoreCombiner
from pkp.rerank.interfaces import (
    CrossEncoderProtocol,
    DistillationSink,
    FeedbackSink,
    LearnedFusionRanker,
    LLMRerankExtension,
    ParentContextAssembler,
)
from pkp.rerank.models import (
    RerankCandidate,
    RerankRequest,
    RerankResponse,
    RerankResultItem,
)
from pkp.rerank.postprocess import CandidateDiversityController, PostprocessConfig
from pkp.types.query import QueryUnderstanding


class RerankPipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k: int = 20
    top_n: int = 8
    max_children_per_parent: int = 1
    preserve_special_slots: int = 1
    enable_feature_logging: bool = True
    candidate_truncation_strategy: str = "query_window"
    cross_encoder: CrossEncoderConfig = CrossEncoderConfig()


class FormalRerankService:
    def __init__(
        self,
        *,
        cross_encoder: CrossEncoderProtocol | None = None,
        feature_extractor: RerankFeatureExtractor | None = None,
        score_combiner: FeatureBasedScoreCombiner | None = None,
        diversity_controller: CandidateDiversityController | None = None,
        learned_ranker: LearnedFusionRanker | None = None,
        llm_extension: LLMRerankExtension | None = None,
        distillation_sink: DistillationSink | None = None,
        feedback_sink: FeedbackSink | None = None,
        parent_context_assembler: ParentContextAssembler | None = None,
        config: RerankPipelineConfig | None = None,
    ) -> None:
        self._config = config or RerankPipelineConfig()
        self._cross_encoder: CrossEncoderProtocol = (
            cross_encoder
            if cross_encoder is not None
            else ProviderBackedCrossEncoder(config=self._config.cross_encoder)
        )
        self._feature_extractor = feature_extractor or RerankFeatureExtractor()
        self._score_combiner = score_combiner or FeatureBasedScoreCombiner()
        self._diversity_controller = diversity_controller or CandidateDiversityController(
            PostprocessConfig(
                top_n=self._config.top_n,
                max_children_per_parent=self._config.max_children_per_parent,
                preserve_special_slots=self._config.preserve_special_slots,
            )
        )
        self._learned_ranker = learned_ranker
        self._llm_extension = llm_extension
        self._distillation_sink = distillation_sink
        self._feedback_sink = feedback_sink
        self._parent_context_assembler = parent_context_assembler
        self._query_understanding_service = QueryUnderstandingService()
        self.last_response: RerankResponse | None = None

    def rerank(self, query: str, candidates: Sequence[object]) -> list[object]:
        request = RerankRequest(
            query=query,
            query_analysis=self._query_understanding_service.analyze(query),
            candidate_list=[
                self._candidate_from_object(candidate, rank=index)
                for index, candidate in enumerate(candidates, start=1)
            ],
            top_k=min(len(candidates), self._config.top_k),
            top_n=min(len(candidates), self._config.top_n),
        )
        response = self.run(request)
        selected_ids = {item.chunk_id for item in response.items}
        ranked_ids = [item.chunk_id for item in response.items]
        candidate_by_id = {cast(Any, candidate).chunk_id: candidate for candidate in candidates}
        ordered = [candidate_by_id[chunk_id] for chunk_id in ranked_ids if chunk_id in candidate_by_id]
        if not ordered:
            return list(candidates)
        leftovers = [candidate for candidate in candidates if cast(Any, candidate).chunk_id not in selected_ids]
        return [*ordered, *leftovers]

    def run(self, request: RerankRequest) -> RerankResponse:
        limit = request.top_k or self._config.top_k
        candidates = list(request.candidate_list[:limit])
        enriched_candidates = [self._with_parent_context(request.query, candidate) for candidate in candidates]
        cross_encoder_scores = self._cross_encoder.score(
            request.query,
            enriched_candidates,
            config=self._config.cross_encoder.model_copy(
                update={
                    "top_k": request.top_k or self._config.top_k,
                    "top_n": request.top_n or self._config.top_n,
                    "candidate_truncation_strategy": self._config.candidate_truncation_strategy,
                }
            ),
        )
        feature_logs = self._feature_extractor.extract(
            request=request,
            candidates=enriched_candidates,
            cross_encoder_scores=cross_encoder_scores,
        )
        final_scores = (
            self._learned_ranker.score(request=request, features=feature_logs)
            if self._learned_ranker is not None
            else [
                self._score_combiner.combine(candidate=candidate, feature_record=feature_log)[0]
                for candidate, feature_log in zip(enriched_candidates, feature_logs, strict=True)
            ]
        )
        items: list[RerankResultItem] = []
        for index, (candidate, feature_log, rerank_score, final_score) in enumerate(
            zip(enriched_candidates, feature_logs, cross_encoder_scores, final_scores, strict=True),
            start=1,
        ):
            score_breakdown = (
                {}
                if self._learned_ranker is not None
                else self._score_combiner.combine(candidate=candidate, feature_record=feature_log)[1]
            )
            items.append(
                RerankResultItem(
                    chunk_id=candidate.chunk_id,
                    rerank_score=float(rerank_score),
                    final_score=float(final_score),
                    rank_before=candidate.unified_rank or index,
                    rank_after=index,
                    feature_summary={**feature_log.feature_dict, **score_breakdown},
                    channel_summary=list(candidate.retrieval_channels),
                    text=candidate.text,
                    doc_id=candidate.doc_id,
                    parent_id=candidate.parent_id,
                    chunk_type=candidate.chunk_type,
                    metadata=dict(candidate.metadata),
                )
            )
        ranked_items = sorted(items, key=lambda item: item.final_score, reverse=True)
        top_n = request.top_n or self._config.top_n
        kept_items, dropped_items = self._diversity_controller.postprocess(
            items=ranked_items[: max(top_n, len(ranked_items))],
            query_analysis=request.query_analysis,
        )
        response = RerankResponse(
            query=request.query,
            query_analysis=request.query_analysis,
            model_name=getattr(self._cross_encoder, "model_name", "unknown"),
            backend_name=getattr(self._cross_encoder, "backend_name", "unknown"),
            raw_candidates=enriched_candidates,
            feature_logs=feature_logs if self._config.enable_feature_logging else [],
            items=kept_items[:top_n],
            dropped_items=dropped_items,
        )
        if self._llm_extension is not None:
            response = self._llm_extension.refine(request=request, response=response)
        if self._distillation_sink is not None:
            self._distillation_sink.record(request=request, response=response)
        if self._feedback_sink is not None:
            self._feedback_sink.record(request=request, response=response)
        self.last_response = response
        return response

    def _with_parent_context(self, query: str, candidate: RerankCandidate) -> RerankCandidate:
        if self._parent_context_assembler is not None:
            return candidate.model_copy(
                update={"parent_text": self._parent_context_assembler.build_context(candidate, query)}
            )
        return candidate

    @staticmethod
    def _candidate_from_object(candidate: object, *, rank: int) -> RerankCandidate:
        candidate_any = cast(Any, candidate)
        metadata = getattr(candidate, "metadata", {})
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        section_path = list(getattr(candidate, "section_path", ()) or ())
        special_chunk_type = getattr(candidate, "special_chunk_type", None)
        chunk_role = getattr(candidate, "chunk_role", None)
        chunk_type = (
            special_chunk_type
            or getattr(chunk_role, "value", None)
            or "child"
        )
        retrieval_channels = list(getattr(candidate, "retrieval_channels", ()) or ())
        if not retrieval_channels:
            source_kind = getattr(candidate, "source_kind", "internal")
            retrieval_channels = [source_kind]
        page_value = metadata_dict.get("page_no")
        page_number = int(page_value) if isinstance(page_value, str) and page_value.isdigit() else None
        return RerankCandidate(
            chunk_id=candidate_any.chunk_id,
            doc_id=getattr(candidate_any, "doc_id", "unknown-doc"),
            parent_id=getattr(candidate_any, "parent_chunk_id", None),
            text=candidate_any.text,
            chunk_type=str(chunk_type),
            section_path=section_path,
            heading_text=section_path[-1] if section_path else None,
            page_start=page_number,
            page_end=page_number,
            retrieval_channels=retrieval_channels,
            dense_score=getattr(candidate, "dense_score", None),
            sparse_score=getattr(candidate, "sparse_score", None),
            special_score=getattr(candidate, "special_score", None),
            structure_score=getattr(candidate, "structure_score", None),
            metadata_score=getattr(candidate, "metadata_score", None),
            fusion_score=getattr(candidate, "fusion_score", getattr(candidate, "score", None)),
            rrf_score=getattr(candidate, "rrf_score", getattr(candidate, "score", None)),
            unified_rank=getattr(candidate, "unified_rank", getattr(candidate, "rank", rank)),
            metadata=metadata_dict,
            parent_text=getattr(candidate, "parent_text", None),
        )


def ensure_query_analysis(query: str, query_analysis: QueryUnderstanding | None) -> QueryUnderstanding:
    return query_analysis or QueryUnderstandingService().analyze(query)
