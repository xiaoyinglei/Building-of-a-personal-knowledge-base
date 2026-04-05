from __future__ import annotations

import importlib
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from rag.llm._integrations.huggingface import suppress_backend_fast_tokenizer_padding_warning
from rag.llm._rerank.models import RerankCandidate
from rag.llm.model_paths import resolve_local_model_reference
from rag.schema._types.text import (
    keyword_overlap,
    search_terms,
    split_sentences,
    text_unit_count,
)


class CrossEncoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_path: str | None = None
    max_length: int = 512
    batch_size: int = 8
    top_k: int = 20
    top_n: int = 8
    candidate_truncation_strategy: str = "query_window"


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
            module = importlib.import_module("FlagEmbedding")
        except ModuleNotFoundError:
            return None
        reranker_cls = getattr(module, "FlagReranker", None)
        if reranker_cls is None:
            return None
        try:
            model_ref = resolve_local_model_reference(self._config.model_name, self._config.model_path)
            backend = cast(object, reranker_cls(model_ref, use_fp16=False))
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
