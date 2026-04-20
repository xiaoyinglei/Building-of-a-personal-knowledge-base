from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import inspect
from typing import Any

from rag.retrieval.evidence import CandidateLike


@dataclass(frozen=True, slots=True)
class PreRerankDiagnostics:
    input_count: int
    deduplicated_count: int
    version_filtered_count: int
    noise_pruned_count: int
    output_count: int


@dataclass(frozen=True, slots=True)
class IndustrialRerankResult:
    ranked_candidates: list[CandidateLike]
    diagnostics: PreRerankDiagnostics
    top1_confidence: float | None
    exit_decision: str | None


class IndustrialRerankService:
    def __init__(
        self,
        *,
        min_keep: int = 3,
        max_model_candidates: int = 50,
        relative_score_floor: float = 0.2,
        absolute_score_floor: float = 0.01,
        empty_response_threshold: float = 0.15,
        asset_fallback_threshold: float = 0.35,
    ) -> None:
        self._min_keep = min_keep
        self._max_model_candidates = max_model_candidates
        self._relative_score_floor = relative_score_floor
        self._absolute_score_floor = absolute_score_floor
        self._empty_response_threshold = empty_response_threshold
        self._asset_fallback_threshold = asset_fallback_threshold

    def rank(
        self,
        *,
        query: str,
        fused_candidates: Sequence[CandidateLike],
        reranker: object | None,
        rerank_required: bool,
        rerank_pool_k: int | None,
        allow_asset_fallback: bool,
    ) -> IndustrialRerankResult:
        cleaned_candidates, diagnostics = self._pre_rerank_cleanup(fused_candidates)
        cleaned_candidates = cleaned_candidates[: self._max_model_candidates]
        diagnostics = PreRerankDiagnostics(
            input_count=diagnostics.input_count,
            deduplicated_count=diagnostics.deduplicated_count,
            version_filtered_count=diagnostics.version_filtered_count,
            noise_pruned_count=diagnostics.noise_pruned_count,
            output_count=len(cleaned_candidates),
        )
        ranked_candidates = (
            self._rerank_candidates(
                query=query,
                candidates=cleaned_candidates,
                reranker=reranker,
                rerank_pool_k=rerank_pool_k,
            )
            if rerank_required
            else list(cleaned_candidates)
        )
        top1_confidence = self._top1_confidence(ranked_candidates)
        exit_decision = self._exit_decision(
            top1_confidence=top1_confidence,
            has_candidates=bool(ranked_candidates),
            allow_asset_fallback=allow_asset_fallback,
        )
        return IndustrialRerankResult(
            ranked_candidates=ranked_candidates,
            diagnostics=diagnostics,
            top1_confidence=top1_confidence,
            exit_decision=exit_decision,
        )

    async def arank(
        self,
        *,
        query: str,
        fused_candidates: Sequence[CandidateLike],
        reranker: object | None,
        rerank_required: bool,
        rerank_pool_k: int | None,
        allow_asset_fallback: bool,
    ) -> IndustrialRerankResult:
        cleaned_candidates, diagnostics = self._pre_rerank_cleanup(fused_candidates)
        cleaned_candidates = cleaned_candidates[: self._max_model_candidates]
        diagnostics = PreRerankDiagnostics(
            input_count=diagnostics.input_count,
            deduplicated_count=diagnostics.deduplicated_count,
            version_filtered_count=diagnostics.version_filtered_count,
            noise_pruned_count=diagnostics.noise_pruned_count,
            output_count=len(cleaned_candidates),
        )
        ranked_candidates = (
            await self._arerank_candidates(
                query=query,
                candidates=cleaned_candidates,
                reranker=reranker,
                rerank_pool_k=rerank_pool_k,
            )
            if rerank_required
            else list(cleaned_candidates)
        )
        top1_confidence = self._top1_confidence(ranked_candidates)
        exit_decision = self._exit_decision(
            top1_confidence=top1_confidence,
            has_candidates=bool(ranked_candidates),
            allow_asset_fallback=allow_asset_fallback,
        )
        return IndustrialRerankResult(
            ranked_candidates=ranked_candidates,
            diagnostics=diagnostics,
            top1_confidence=top1_confidence,
            exit_decision=exit_decision,
        )

    def _pre_rerank_cleanup(
        self,
        candidates: Sequence[CandidateLike],
    ) -> tuple[list[CandidateLike], PreRerankDiagnostics]:
        deduplicated, deduplicated_count = self._deduplicate(candidates)
        active_candidates, version_filtered_count = self._filter_versions(deduplicated)
        pruned_candidates, noise_pruned_count = self._prune_noise(active_candidates)
        return pruned_candidates, PreRerankDiagnostics(
            input_count=len(candidates),
            deduplicated_count=deduplicated_count,
            version_filtered_count=version_filtered_count,
            noise_pruned_count=noise_pruned_count,
            output_count=len(pruned_candidates),
        )

    def _deduplicate(self, candidates: Sequence[CandidateLike]) -> tuple[list[CandidateLike], int]:
        kept: list[CandidateLike] = []
        seen: set[tuple[str, str]] = set()
        deduplicated_count = 0
        for candidate in candidates:
            key = self._dedupe_key(candidate)
            if key in seen:
                deduplicated_count += 1
                continue
            seen.add(key)
            kept.append(candidate)
        return kept, deduplicated_count

    def _filter_versions(self, candidates: Sequence[CandidateLike]) -> tuple[list[CandidateLike], int]:
        chosen: dict[tuple[str, str], tuple[int, float, int, CandidateLike]] = {}
        passthrough: list[tuple[int, CandidateLike]] = []
        dropped = 0
        for index, candidate in enumerate(candidates):
            metadata = _candidate_metadata(candidate)
            if _is_inactive_candidate(metadata):
                dropped += 1
                continue
            version_group_id = metadata.get("version_group_id")
            version_no = _safe_int(metadata.get("version_no"))
            version_key = self._version_key(candidate, version_group_id)
            if version_key is None or version_no is None:
                passthrough.append((index, candidate))
                continue
            score = float(getattr(candidate, "score", 0.0))
            current = chosen.get(version_key)
            if current is None or (version_no, score) > (current[0], current[1]):
                if current is not None:
                    dropped += 1
                chosen[version_key] = (version_no, score, index, candidate)
            else:
                dropped += 1
        kept = [candidate for _, candidate in passthrough]
        kept.extend(candidate for _, _, _, candidate in sorted(chosen.values(), key=lambda item: item[2]))
        kept.sort(key=lambda candidate: (-float(getattr(candidate, "score", 0.0)), getattr(candidate, "chunk_id", "")))
        return kept, dropped

    def _prune_noise(self, candidates: Sequence[CandidateLike]) -> tuple[list[CandidateLike], int]:
        if len(candidates) <= self._min_keep:
            return list(candidates), 0
        top_score = max(max(float(getattr(candidate, "score", 0.0)), 0.0) for candidate in candidates)
        floor = max(self._absolute_score_floor, top_score * self._relative_score_floor)
        kept: list[CandidateLike] = []
        dropped = 0
        for index, candidate in enumerate(candidates):
            score = max(float(getattr(candidate, "score", 0.0)), 0.0)
            if index < self._min_keep or score >= floor:
                kept.append(candidate)
                continue
            dropped += 1
        return kept, dropped

    @staticmethod
    def _rerank_candidates(
        *,
        query: str,
        candidates: Sequence[CandidateLike],
        reranker: object | None,
        rerank_pool_k: int | None,
    ) -> list[CandidateLike]:
        rerank = getattr(reranker, "rerank", None)
        if not callable(rerank):
            return list(candidates)
        if rerank_pool_k is None:
            return list(rerank(query, list(candidates)))
        normalized_limit = max(1, rerank_pool_k)
        head = list(candidates[:normalized_limit])
        tail = list(candidates[normalized_limit:])
        reranked_head = list(rerank(query, head))
        return [*reranked_head, *tail]

    @staticmethod
    async def _arerank_candidates(
        *,
        query: str,
        candidates: Sequence[CandidateLike],
        reranker: object | None,
        rerank_pool_k: int | None,
    ) -> list[CandidateLike]:
        rerank = getattr(reranker, "rerank", None)
        if not callable(rerank):
            return list(candidates)
        if rerank_pool_k is None:
            result = rerank(query, list(candidates))
            if inspect.isawaitable(result):
                return list(await result)
            return list(result)
        normalized_limit = max(1, rerank_pool_k)
        head = list(candidates[:normalized_limit])
        tail = list(candidates[normalized_limit:])
        reranked_head = rerank(query, head)
        if inspect.isawaitable(reranked_head):
            reranked_head = await reranked_head
        return [*list(reranked_head), *tail]

    @staticmethod
    def _top1_confidence(candidates: Sequence[CandidateLike]) -> float | None:
        if not candidates:
            return None
        top1 = abs(float(getattr(candidates[0], "score", 0.0)))
        if len(candidates) == 1:
            return round(top1 / (top1 + 1.0), 6)
        top2 = abs(float(getattr(candidates[1], "score", 0.0)))
        denominator = max(top1 + top2, 1e-6)
        confidence = max(0.0, min(1.0, 0.5 * (top1 / denominator) + 0.5 * ((top1 - top2) / denominator)))
        return round(confidence, 6)

    def _exit_decision(
        self,
        *,
        top1_confidence: float | None,
        has_candidates: bool,
        allow_asset_fallback: bool,
    ) -> str | None:
        if not has_candidates or top1_confidence is None:
            return "empty_response"
        if top1_confidence < self._empty_response_threshold:
            return "empty_response"
        if allow_asset_fallback and top1_confidence < self._asset_fallback_threshold:
            return "asset_fallback"
        return "answer"

    @staticmethod
    def _dedupe_key(candidate: CandidateLike) -> tuple[str, str]:
        metadata = _candidate_metadata(candidate)
        section_id = metadata.get("section_id")
        if section_id is not None:
            return ("section", str(section_id))
        return ("chunk", getattr(candidate, "chunk_id", ""))

    @staticmethod
    def _version_key(candidate: CandidateLike, version_group_id: object | None) -> tuple[str, str] | None:
        if version_group_id is None:
            return None
        metadata = _candidate_metadata(candidate)
        section_id = metadata.get("section_id")
        if section_id is not None:
            return (str(version_group_id), str(section_id))
        return (str(version_group_id), getattr(candidate, "chunk_id", ""))


def _candidate_metadata(candidate: CandidateLike) -> dict[str, Any]:
    metadata = getattr(candidate, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _is_inactive_candidate(metadata: dict[str, Any]) -> bool:
    is_active = metadata.get("is_active")
    if isinstance(is_active, bool):
        return not is_active
    if isinstance(is_active, str) and is_active.lower() in {"0", "false", "no"}:
        return True
    doc_status = str(metadata.get("doc_status", "")).strip().lower()
    return doc_status in {"retired", "expired", "deleted"}


def _safe_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


__all__ = [
    "IndustrialRerankResult",
    "IndustrialRerankService",
    "PreRerankDiagnostics",
]
