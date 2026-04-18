from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from rag.assembly import RerankCapabilityBinding
from rag.providers.adapters import (
    _infer_flagembedding_reranker_model_class,
    _load_flagembedding_module,
    suppress_backend_fast_tokenizer_padding_warning,
)
from rag.retrieval.analysis import QueryUnderstandingService
from rag.schema.query import QueryUnderstanding
from rag.utils.text import keyword_overlap, looks_command_like, search_terms, split_sentences, text_unit_count


class RerankCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    parent_id: str | None = None
    text: str
    chunk_type: str
    section_path: list[str] = Field(default_factory=list)
    heading_text: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    dense_score: float | None = None
    sparse_score: float | None = None
    special_score: float | None = None
    structure_score: float | None = None
    metadata_score: float | None = None
    fusion_score: float | None = None
    rrf_score: float | None = None
    unified_rank: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)
    parent_text: str | None = None


class RerankRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    query_analysis: QueryUnderstanding
    candidate_list: list[RerankCandidate]
    top_k: int | None = None
    top_n: int | None = None
    debug: bool = False


class FeatureRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    feature_dict: dict[str, float | int | bool | str] = Field(default_factory=dict)


class RerankResultItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    rerank_score: float
    final_score: float
    rank_before: int
    rank_after: int
    feature_summary: dict[str, float | int | bool | str] = Field(default_factory=dict)
    channel_summary: list[str] = Field(default_factory=list)
    text: str
    doc_id: str
    parent_id: str | None = None
    chunk_type: str
    metadata: dict[str, str] = Field(default_factory=dict)
    drop_reason: str | None = None


class RerankResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    query_analysis: QueryUnderstanding
    model_name: str
    backend_name: str
    raw_candidates: list[RerankCandidate]
    feature_logs: list[FeatureRecord] = Field(default_factory=list)
    items: list[RerankResultItem] = Field(default_factory=list)
    dropped_items: list[RerankResultItem] = Field(default_factory=list)


class CrossEncoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_path: str | None = None
    max_length: int = 512
    batch_size: int = 8
    top_k: int = 20
    top_n: int = 8
    candidate_truncation_strategy: str = "query_window"


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


class RerankFeatureExtractor:
    def extract(
        self,
        *,
        request: RerankRequest,
        candidates: Sequence[RerankCandidate],
        cross_encoder_scores: Sequence[float],
    ) -> list[FeatureRecord]:
        query_terms = search_terms(request.query)
        query_numbers = {term for term in query_terms if term.isdigit()}
        query_is_command_like = looks_command_like(request.query)
        requested_special_targets = set(request.query_analysis.special_targets)
        parent_counts = Counter(candidate.parent_id for candidate in candidates if candidate.parent_id)
        max_order_index = max((int(candidate.metadata.get("order_index", "0")) for candidate in candidates), default=0)

        records: list[FeatureRecord] = []
        for index, candidate in enumerate(candidates):
            heading_text = candidate.heading_text or ""
            section_text = " ".join(candidate.section_path)
            parent_text = candidate.parent_text or ""
            metadata = candidate.metadata
            text_numbers = {term for term in search_terms(candidate.text) if term.isdigit()}
            page_numbers = {
                value
                for value in (
                    str(candidate.page_start) if candidate.page_start is not None else "",
                    str(candidate.page_end) if candidate.page_end is not None else "",
                    metadata.get("page_no", ""),
                )
                if value
            }
            preferred_sections = set(request.query_analysis.preferred_section_terms)
            requested_pages = {str(page) for page in request.query_analysis.metadata_filters.page_numbers}
            feature_dict: dict[str, float | int | bool | str] = {
                "dense_score": round(candidate.dense_score or 0.0, 6),
                "sparse_score": round(candidate.sparse_score or 0.0, 6),
                "special_score": round(candidate.special_score or 0.0, 6),
                "structure_score": round(candidate.structure_score or 0.0, 6),
                "metadata_score": round(candidate.metadata_score or 0.0, 6),
                "fusion_score": round(candidate.fusion_score or 0.0, 6),
                "rrf_score": round(candidate.rrf_score or 0.0, 6),
                "retrieval_rank": candidate.unified_rank,
                "retrieval_channel_count": len(candidate.retrieval_channels),
                "title_hit": keyword_overlap(query_terms, heading_text),
                "section_path_hit": keyword_overlap(query_terms, section_text),
                "token_overlap": keyword_overlap(query_terms, candidate.text),
                "number_match": bool(query_numbers & (text_numbers | page_numbers)),
                "exact_phrase_hit": request.query.strip().lower() in candidate.text.lower(),
                "heading_level_match": 1 if preferred_sections & set(candidate.section_path) else 0,
                "candidate_is_command_like": looks_command_like(candidate.text),
                "query_is_command_like": query_is_command_like,
                "is_table": candidate.chunk_type == "table",
                "is_figure": candidate.chunk_type == "figure",
                "is_ocr_region": candidate.chunk_type == "ocr_region",
                "is_image_summary": candidate.chunk_type == "image_summary",
                "query_requires_special": bool(requested_special_targets),
                "special_target_match": bool(
                    requested_special_targets and candidate.chunk_type in requested_special_targets
                ),
                "parent_section_match": bool(preferred_sections & set(candidate.section_path)),
                "page_match": bool(requested_pages & page_numbers),
                "source_type": metadata.get("source_type", ""),
                "candidate_length": text_unit_count(candidate.text),
                "parent_length": text_unit_count(parent_text),
                "same_parent_candidate_count": 0
                if candidate.parent_id is None
                else parent_counts.get(candidate.parent_id, 0),
                "relative_position": (
                    0.0 if max_order_index <= 0 else round(int(metadata.get("order_index", "0")) / max_order_index, 6)
                ),
                "cross_encoder_score": round(float(cross_encoder_scores[index]), 6),
            }
            records.append(FeatureRecord(chunk_id=candidate.chunk_id, feature_dict=feature_dict))
        return records


class FeatureBasedScoreCombiner(ScoreCombinerProtocol):
    def combine(
        self,
        *,
        candidate: RerankCandidate,
        feature_record: FeatureRecord,
    ) -> tuple[float, dict[str, float | int | bool | str]]:
        del candidate
        features = feature_record.feature_dict
        retrieval_signal = max(
            float(features.get("dense_score", 0.0)),
            float(features.get("sparse_score", 0.0)),
            float(features.get("special_score", 0.0)),
            float(features.get("fusion_score", 0.0)),
            float(features.get("rrf_score", 0.0)),
        )
        cross_encoder_signal = float(features.get("cross_encoder_score", 0.0))
        special_target_match = bool(features.get("special_target_match", False))
        query_requires_special = bool(features.get("query_requires_special", False))
        constraint_match = bool(
            features.get("page_match", False)
            or features.get("parent_section_match", False)
            or features.get("exact_phrase_hit", False)
            or special_target_match
        )
        if special_target_match and query_requires_special:
            final_score = round(max(cross_encoder_signal, retrieval_signal), 6)
        else:
            final_score = round(cross_encoder_signal if cross_encoder_signal > 0.0 else retrieval_signal, 6)
        return final_score, {
            "cross_encoder_signal": round(cross_encoder_signal, 6),
            "retrieval_signal": round(retrieval_signal, 6),
            "constraint_match": constraint_match,
            "special_target_match": special_target_match,
        }


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_n: int = 8
    max_children_per_parent: int = 1
    preserve_special_slots: int = 1


class CandidateDiversityController:
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, config: PostprocessConfig | None = None) -> None:
        self._config = config or PostprocessConfig()

    def postprocess(
        self,
        *,
        items: list[RerankResultItem],
        query_analysis: QueryUnderstanding,
    ) -> tuple[list[RerankResultItem], list[RerankResultItem]]:
        kept: list[RerankResultItem] = []
        dropped: list[RerankResultItem] = []
        parent_counts: dict[str, int] = {}
        seen_texts: set[str] = set()

        for item in sorted(items, key=lambda candidate: candidate.final_score, reverse=True):
            normalized_text = self._normalize(item.text)
            if normalized_text in seen_texts:
                dropped.append(item.model_copy(update={"drop_reason": "near_duplicate"}))
                continue
            if item.chunk_type == "child" and item.parent_id is not None:
                current = parent_counts.get(item.parent_id, 0)
                if current >= self._config.max_children_per_parent:
                    dropped.append(item.model_copy(update={"drop_reason": "same_parent_redundant"}))
                    continue
                parent_counts[item.parent_id] = current + 1
            seen_texts.add(normalized_text)
            kept.append(item)

        kept = kept[: self._config.top_n]
        if query_analysis.needs_special and query_analysis.special_targets:
            kept, dropped = self._ensure_special_diversity(kept, dropped, query_analysis)

        reranked = [item.model_copy(update={"rank_after": index}) for index, item in enumerate(kept, start=1)]
        return reranked, dropped

    def _ensure_special_diversity(
        self,
        kept: list[RerankResultItem],
        dropped: list[RerankResultItem],
        query_analysis: QueryUnderstanding,
    ) -> tuple[list[RerankResultItem], list[RerankResultItem]]:
        target_types = set(query_analysis.special_targets)
        retained_specials = [item for item in kept if item.chunk_type in target_types]
        if retained_specials or self._config.preserve_special_slots <= 0:
            return kept, dropped

        special_pool = [item for item in dropped if item.chunk_type in target_types]
        if not special_pool:
            return kept, dropped

        selected = sorted(special_pool, key=lambda item: item.final_score, reverse=True)[0]
        replacement_index = next(
            (index for index in range(len(kept) - 1, -1, -1) if kept[index].chunk_type not in target_types),
            -1,
        )
        if replacement_index < 0:
            return kept, dropped
        replaced = kept[replacement_index]
        updated_dropped = [item for item in dropped if item.chunk_id != selected.chunk_id]
        updated_dropped.append(replaced.model_copy(update={"drop_reason": "special_diversity_replacement"}))
        kept[replacement_index] = selected
        return kept, updated_dropped

    @classmethod
    def _normalize(cls, text: str) -> str:
        return cls._WHITESPACE_RE.sub(" ", text.strip().lower())


def _expand_optional_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    if isinstance(raw, Path):
        return raw.expanduser()
    normalized = raw.strip()
    if not normalized:
        return None
    return Path(normalized).expanduser()


def _resolve_huggingface_snapshot_path(model_root: str | Path) -> Path:
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


def _resolve_local_model_reference(model_name: str, model_path: str | Path | None) -> str:
    expanded = _expand_optional_path(model_path)
    if expanded is None:
        return model_name
    return str(_resolve_huggingface_snapshot_path(expanded))


def _looks_like_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "tokenizer_config.json").exists()


class ProviderBackedCrossEncoder:
    def __init__(
        self,
        *,
        provider: object | None = None,
        config: CrossEncoderConfig | None = None,
    ) -> None:
        self._provider = provider
        self._config = config or CrossEncoderConfig()
        self._backend: object | None = None
        self.backend_name = "unconfigured"
        self.model_name = self._config.model_name
        self.device = "cpu"

    def load(self) -> None:
        if self._backend is not None:
            return
        backend = self._try_provider_backend()
        if backend is not None:
            self._backend = backend
            self.backend_name = "provider_rerank"
            return
        backend = self._try_flag_embedding_backend()
        if backend is not None:
            self._backend = backend
            self.backend_name = "bge_local"
            return
        raise RuntimeError(
            "No model-backed reranker is configured. "
            "Configure RAG_RERANK_MODEL/RAG_RERANK_MODEL_PATH or provide a provider with is_rerank_configured=true."
        )

    def release(self) -> None:
        self._backend = None

    def score(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        config: CrossEncoderConfig | None = None,
    ) -> list[float]:
        resolved = config or self._config
        self.load()
        prepared_candidates = [
            candidate.model_copy(update={"text": self._truncate_candidate(query, candidate.text, config=resolved)})
            for candidate in candidates
        ]
        if self.backend_name == "bge_local" and self._backend is not None:
            return self._score_with_local_backend(query, prepared_candidates, resolved)
        if self.backend_name == "provider_rerank" and callable(self._backend):
            return self._score_with_provider(query, prepared_candidates)
        raise RuntimeError("Cross encoder backend was not initialized")

    def _score_with_provider(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        rerank = cast(Any, self._backend)
        ranking = rerank(query, [candidate.text for candidate in candidates])
        size = max(len(candidates), 1)
        scores = [0.0] * len(candidates)
        for index, candidate_index in enumerate(ranking):
            if not isinstance(candidate_index, int) or candidate_index >= len(candidates):
                continue
            scores[candidate_index] = 1.0 - (index / size)
        return scores

    def _score_with_local_backend(
        self,
        query: str,
        candidates: list[RerankCandidate],
        config: CrossEncoderConfig,
    ) -> list[float]:
        backend = cast(Any, self._backend)
        pairs = [[query, candidate.text] for candidate in candidates]
        if hasattr(backend, "compute_score"):
            scores = backend.compute_score(
                pairs,
                batch_size=config.batch_size,
                max_length=config.max_length,
            )
            return [float(score) for score in scores]
        raise RuntimeError("Local rerank backend does not expose compute_score")

    def _try_provider_backend(self) -> object | None:
        rerank = getattr(self._provider, "rerank", None)
        if not callable(rerank):
            return None
        if not bool(getattr(self._provider, "is_rerank_configured", True)):
            return None
        model_name = getattr(self._provider, "rerank_model_name", None)
        if isinstance(model_name, str) and model_name:
            self.model_name = model_name
        return cast(object, rerank)

    def _try_flag_embedding_backend(self) -> object | None:
        if not self._config.model_path:
            return None
        try:
            module = _load_flagembedding_module()
        except ModuleNotFoundError:
            return None
        auto_reranker_cls = getattr(module, "FlagAutoReranker", None)
        if auto_reranker_cls is None:
            return None
        try:
            model_ref = _resolve_local_model_reference(self._config.model_name, self._config.model_path)
            backend = cast(
                object,
                auto_reranker_cls.from_finetuned(
                    model_ref,
                    model_class=_infer_flagembedding_reranker_model_class(model_ref),
                    use_fp16=False,
                    trust_remote_code=True,
                    batch_size=self._config.batch_size,
                    max_length=self._config.max_length,
                ),
            )
            return suppress_backend_fast_tokenizer_padding_warning(backend)
        except Exception:
            return None

    @staticmethod
    def _truncate_candidate(query: str, text: str, *, config: CrossEncoderConfig) -> str:
        if text_unit_count(text) <= config.max_length:
            return text
        if config.candidate_truncation_strategy == "head_tail":
            half = max(config.max_length // 2, 1)
            words = text.split()
            if len(words) <= config.max_length:
                return " ".join(words)
            return " ".join([*words[:half], *words[-half:]])

        query_terms = search_terms(query)
        sentences = split_sentences(text)
        scored_sentences = sorted(
            sentences,
            key=lambda sentence: (
                keyword_overlap(query_terms, sentence),
                -abs(len(sentence) - len(text) // max(len(sentences), 1)),
            ),
            reverse=True,
        )
        selected: list[str] = []
        budget = 0
        for sentence in scored_sentences:
            sentence_units = text_unit_count(sentence)
            if budget and budget + sentence_units > config.max_length:
                continue
            selected.append(sentence)
            budget += sentence_units
            if budget >= config.max_length:
                break
        if not selected:
            return text[: config.max_length]
        return " ".join(selected)


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
        query_understanding_service: QueryUnderstandingService | None = None,
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
        self._query_understanding_service = query_understanding_service or QueryUnderstandingService()
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
        ranked_items = sorted(
            items,
            key=lambda item: (
                item.final_score,
                bool(item.feature_summary.get("special_target_match", False)),
                bool(item.feature_summary.get("constraint_match", False)),
                -item.rank_before,
            ),
            reverse=True,
        )
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
        chunk_type = special_chunk_type or getattr(chunk_role, "value", None) or "child"
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
    "CandidateDiversityController",
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "FeatureRecord",
    "FormalRerankPipeline",
    "FormalRerankService",
    "ModelBackedRerankService",
    "PostprocessConfig",
    "ProviderBackedCrossEncoder",
    "RerankCandidate",
    "RerankPipelineConfig",
    "RerankRequest",
    "RerankResponse",
    "RerankResultItem",
]
