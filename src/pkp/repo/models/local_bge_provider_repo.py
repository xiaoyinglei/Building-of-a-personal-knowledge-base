from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, cast

from pkp.config.model_paths import resolve_local_model_reference
from pkp.integrations.huggingface import suppress_backend_fast_tokenizer_padding_warning


class LocalBgeProviderRepo:
    def __init__(
        self,
        *,
        embedding_model: str = "BAAI/bge-m3",
        embedding_model_path: str | None = None,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        rerank_model_path: str | None = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        batch_size: int = 8,
        max_length: int = 1024,
        rerank_batch_size: int = 8,
        rerank_max_length: int = 512,
    ) -> None:
        self._embedding_model = embedding_model
        self._embedding_model_ref = resolve_local_model_reference(embedding_model, embedding_model_path)
        self._rerank_model = rerank_model
        self._rerank_model_ref = resolve_local_model_reference(rerank_model, rerank_model_path)
        self._normalize_embeddings = normalize_embeddings
        self._use_fp16 = use_fp16
        self._batch_size = batch_size
        self._max_length = max_length
        self._rerank_batch_size = rerank_batch_size
        self._rerank_max_length = rerank_max_length
        self._embedding_backend: object | None = None
        self._rerank_backend: object | None = None

    @property
    def provider_name(self) -> str:
        return "local-bge"

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model

    @property
    def rerank_model_name(self) -> str:
        return self._rerank_model

    @property
    def is_embed_configured(self) -> bool:
        return bool(self._embedding_model_ref)

    @property
    def is_rerank_configured(self) -> bool:
        return bool(self._rerank_model_ref)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        backend = self._embedding_backend or self._load_embedding_backend()
        self._embedding_backend = backend
        encoder = cast(Any, backend)
        payload = encoder.encode(
            list(texts),
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vectors = payload.get("dense_vecs") if isinstance(payload, dict) else payload
        if dense_vectors is None:
            raise RuntimeError("Local BGE embedding backend returned no dense vectors")
        return [list(vector) for vector in dense_vectors]

    def rerank(self, query: str, candidates: Sequence[object]) -> list[int]:
        if not candidates:
            return []
        backend = self._rerank_backend or self._load_rerank_backend()
        self._rerank_backend = backend
        reranker = cast(Any, backend)
        candidate_texts = [self._candidate_text(candidate) for candidate in candidates]
        pairs = [[query, candidate_text] for candidate_text in candidate_texts]
        scores = reranker.compute_score(
            pairs,
            batch_size=self._rerank_batch_size,
            max_length=self._rerank_max_length,
        )
        indexed = list(enumerate(float(score) for score in scores))
        indexed.sort(key=lambda item: item[1], reverse=True)
        return [index for index, _score in indexed]

    def _load_embedding_backend(self) -> object:
        try:
            module = importlib.import_module("FlagEmbedding")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency wiring
            raise RuntimeError("FlagEmbedding is required for local BGE embeddings") from exc
        encoder_cls = getattr(module, "BGEM3FlagModel", None)
        if encoder_cls is None:
            raise RuntimeError("FlagEmbedding BGEM3FlagModel is unavailable")
        try:
            backend = cast(
                object,
                encoder_cls(
                    self._embedding_model_ref,
                    normalize_embeddings=self._normalize_embeddings,
                    use_fp16=self._use_fp16,
                    batch_size=self._batch_size,
                    query_max_length=self._max_length,
                    passage_max_length=self._max_length,
                    return_sparse=False,
                    return_colbert_vecs=False,
                ),
            )
            return suppress_backend_fast_tokenizer_padding_warning(backend)
        except Exception as exc:  # pragma: no cover - backend-specific failure
            raise RuntimeError(f"Local BGE embedding load failed: {exc}") from exc

    def _load_rerank_backend(self) -> object:
        try:
            module = importlib.import_module("FlagEmbedding")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency wiring
            raise RuntimeError("FlagEmbedding is required for local BGE rerank") from exc
        reranker_cls = getattr(module, "FlagReranker", None)
        if reranker_cls is None:
            raise RuntimeError("FlagEmbedding FlagReranker is unavailable")
        try:
            backend = cast(
                object,
                reranker_cls(
                    self._rerank_model_ref,
                    use_fp16=self._use_fp16,
                    batch_size=self._rerank_batch_size,
                    max_length=self._rerank_max_length,
                    normalize=False,
                ),
            )
            return suppress_backend_fast_tokenizer_padding_warning(backend)
        except Exception as exc:  # pragma: no cover - backend-specific failure
            raise RuntimeError(f"Local BGE rerank load failed: {exc}") from exc

    @staticmethod
    def _candidate_text(candidate: object) -> str:
        if isinstance(candidate, str):
            return candidate
        text = getattr(candidate, "text", None)
        if isinstance(text, str):
            return text
        return str(candidate)
