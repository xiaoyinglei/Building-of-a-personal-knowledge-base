from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from pkp.query.query import ContextEvidence
from pkp.service.answer_generation_service import AnswerGenerationService
from pkp.service.evidence_service import CandidateLike, EvidenceBundle, EvidenceService, SelfCheckResult
from pkp.service.query_understanding_service import QueryUnderstandingService
from pkp.service.routing_service import RoutingDecision, RoutingService
from pkp.types.envelope import EvidenceItem
from pkp.types.retrieval import RetrievalResult
from pkp.types.text import text_unit_count

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass(slots=True)
class ContextEvidenceMerger:
    def merge(self, retrieval: RetrievalResult) -> list[EvidenceItem]:
        internal_by_id = {item.chunk_id: item for item in retrieval.evidence.internal}
        ordered_internal = [
            internal_by_id[chunk_id]
            for chunk_id in retrieval.reranked_chunk_ids
            if chunk_id in internal_by_id
        ]
        seen_internal = {item.chunk_id for item in ordered_internal}
        ordered_internal.extend(item for item in retrieval.evidence.internal if item.chunk_id not in seen_internal)

        merged: list[EvidenceItem] = []
        merged_by_chunk_id: dict[str, EvidenceItem] = {}
        ordered_chunk_ids: list[str] = []

        for item in [*ordered_internal, *retrieval.evidence.graph]:
            existing = merged_by_chunk_id.get(item.chunk_id)
            if existing is None:
                merged_by_chunk_id[item.chunk_id] = item
                ordered_chunk_ids.append(item.chunk_id)
                continue
            merged_by_chunk_id[item.chunk_id] = self._merge_duplicate_item(existing, item)

        merged.extend(merged_by_chunk_id[chunk_id] for chunk_id in ordered_chunk_ids)

        seen_external: set[str] = set()
        for item in retrieval.evidence.external:
            if item.chunk_id in seen_external:
                continue
            seen_external.add(item.chunk_id)
            merged.append(item)
        return merged

    @staticmethod
    def _merge_duplicate_item(existing: EvidenceItem, incoming: EvidenceItem) -> EvidenceItem:
        preferred = existing
        secondary = incoming
        if existing.evidence_kind != "internal" and incoming.evidence_kind == "internal":
            preferred = incoming
            secondary = existing

        merged_kind = "internal" if "internal" in {existing.evidence_kind, incoming.evidence_kind} else preferred.evidence_kind
        merged_text = preferred.text if len(preferred.text) >= len(secondary.text) else secondary.text

        return preferred.model_copy(
            update={
                "evidence_kind": merged_kind,
                "score": max(float(existing.score), float(incoming.score)),
                "text": merged_text,
                "section_path": preferred.section_path or secondary.section_path,
                "file_name": preferred.file_name or secondary.file_name,
                "source_id": preferred.source_id or secondary.source_id,
                "chunk_type": preferred.chunk_type or secondary.chunk_type,
                "source_type": preferred.source_type or secondary.source_type,
                "special_chunk_type": preferred.special_chunk_type or secondary.special_chunk_type,
                "parent_chunk_id": preferred.parent_chunk_id or secondary.parent_chunk_id,
                "page_start": preferred.page_start if preferred.page_start is not None else secondary.page_start,
                "page_end": preferred.page_end if preferred.page_end is not None else secondary.page_end,
            }
        )


@dataclass(frozen=True, slots=True)
class ContextPromptBuildResult:
    grounded_candidate: str
    prompt: str
    token_count: int


@dataclass(slots=True)
class ContextPromptBuilder:
    answer_generation_service: AnswerGenerationService

    def build(
        self,
        *,
        query: str,
        grounded_candidate: str,
        evidence: list[ContextEvidence],
        runtime_mode,
        token_count: int,
    ) -> ContextPromptBuildResult:
        prompt = self.answer_generation_service.build_prompt(
            query=query,
            evidence_pack=[item.as_evidence_item() for item in evidence],
            grounded_candidate=grounded_candidate,
            runtime_mode=runtime_mode,
        )
        return ContextPromptBuildResult(
            grounded_candidate=grounded_candidate,
            prompt=prompt,
            token_count=token_count,
        )


@dataclass(frozen=True, slots=True)
class ContextTruncationResult:
    evidence: list[ContextEvidence]
    token_budget: int
    token_count: int
    truncated_count: int


@dataclass(slots=True)
class EvidenceTruncator:
    def truncate(
        self,
        evidence: list[EvidenceItem],
        *,
        token_budget: int,
        max_evidence_chunks: int,
    ) -> ContextTruncationResult:
        normalized_budget = max(token_budget, 1)
        normalized_max_chunks = min(max(max_evidence_chunks, 1), normalized_budget)
        prioritized_items = self._prioritize_evidence(evidence, normalized_max_chunks)
        assigned_budgets = self._allocate_token_budgets(prioritized_items, normalized_budget)

        selected: list[ContextEvidence] = []
        consumed = 0
        clipped_count = 0

        for item, item_budget in zip(prioritized_items, assigned_budgets, strict=False):
            original_token_count = text_unit_count(item.text)
            effective_budget = max(item_budget, 1)
            selected_text = item.text
            selected_token_count = original_token_count
            was_truncated = False

            if original_token_count > effective_budget:
                clipped = self._clip_text(item.text, effective_budget)
                clipped_token_count = text_unit_count(clipped)
                if not clipped.strip():
                    continue
                selected_text = clipped
                selected_token_count = min(clipped_token_count, effective_budget)
                was_truncated = clipped_token_count < original_token_count or clipped.endswith(" ...")

            selected.append(
                ContextEvidence(
                    evidence_id=f"E{len(selected) + 1}",
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    source_id=item.source_id,
                    citation_anchor=item.citation_anchor,
                    text=selected_text,
                    score=item.score,
                    evidence_kind=item.evidence_kind,
                    chunk_role=item.chunk_role,
                    special_chunk_type=item.special_chunk_type,
                    parent_chunk_id=item.parent_chunk_id,
                    section_path=list(item.section_path),
                    file_name=item.file_name,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    chunk_type=item.chunk_type,
                    source_type=item.source_type,
                    token_count=original_token_count,
                    selected_token_count=selected_token_count,
                    truncated=was_truncated,
                )
            )
            consumed += selected_token_count
            if was_truncated:
                clipped_count += 1

        skipped_count = max(0, len(evidence) - len(prioritized_items))
        truncated_count = skipped_count + clipped_count
        return ContextTruncationResult(
            evidence=selected,
            token_budget=normalized_budget,
            token_count=consumed,
            truncated_count=truncated_count,
        )

    def _prioritize_evidence(self, evidence: list[EvidenceItem], max_evidence_chunks: int) -> list[EvidenceItem]:
        if len(evidence) <= max_evidence_chunks:
            return list(evidence)

        indexed_items = list(enumerate(evidence))
        selected_indices: list[int] = []
        selected_docs: set[str] = set()
        selected_groups: set[str] = set()

        def pick_best(predicate: Callable[[EvidenceItem], bool] | None = None) -> None:
            remaining = [
                (index, item)
                for index, item in indexed_items
                if index not in selected_indices and (predicate(item) if predicate is not None else True)
            ]
            if not remaining or len(selected_indices) >= max_evidence_chunks:
                return
            best_index, best_item = max(
                remaining,
                key=lambda pair: self._selection_score(
                    pair[1],
                    original_index=pair[0],
                    selected_docs=selected_docs,
                    selected_groups=selected_groups,
                ),
            )
            selected_indices.append(best_index)
            if best_item.doc_id:
                selected_docs.add(best_item.doc_id)
            selected_groups.add(self._group_key(best_item))

        pick_best(lambda item: item.evidence_kind == "internal" and item.special_chunk_type is not None)
        pick_best(lambda item: item.evidence_kind == "internal")
        pick_best(lambda item: item.evidence_kind == "graph")
        while len(selected_indices) < max_evidence_chunks:
            previous_count = len(selected_indices)
            pick_best()
            if len(selected_indices) == previous_count:
                break

        return [evidence[index] for index in sorted(selected_indices)]

    def _allocate_token_budgets(self, evidence: list[EvidenceItem], token_budget: int) -> list[int]:
        if not evidence:
            return []

        weights = [self._budget_weight(item) for item in evidence]
        total_weight = sum(weights) or 1.0
        raw_budgets = [max(1, int(token_budget * (weight / total_weight))) for weight in weights]
        assigned = raw_budgets[:]
        remainder = token_budget - sum(assigned)

        if remainder > 0:
            for index in range(remainder):
                assigned[index % len(assigned)] += 1
        elif remainder < 0:
            for _ in range(-remainder):
                candidates = [index for index, value in enumerate(assigned) if value > 1]
                if not candidates:
                    break
                largest = max(candidates, key=lambda index: assigned[index])
                assigned[largest] -= 1
        return assigned

    def _budget_weight(self, item: EvidenceItem) -> float:
        weight = max(float(item.score), 0.0) + 0.1
        if item.evidence_kind == "internal":
            weight += 0.3
        if item.special_chunk_type:
            weight += 0.25
        if item.chunk_role and getattr(item.chunk_role, "value", "") == "special":
            weight += 0.15
        if item.page_start is not None:
            weight += 0.05
        return weight

    def _selection_score(
        self,
        item: EvidenceItem,
        *,
        original_index: int,
        selected_docs: set[str],
        selected_groups: set[str],
    ) -> tuple[float, float, float, int]:
        score = max(float(item.score), 0.0)
        novelty_bonus = 0.0 if item.doc_id in selected_docs else 0.15
        group_bonus = 0.0 if self._group_key(item) in selected_groups else 0.1
        kind_bonus = 0.2 if item.evidence_kind == "internal" else 0.0
        if item.special_chunk_type:
            kind_bonus += 0.15
        return (score + novelty_bonus + group_bonus + kind_bonus, score, kind_bonus, -original_index)

    @staticmethod
    def _group_key(item: EvidenceItem) -> str:
        if item.parent_chunk_id:
            return f"parent:{item.doc_id}:{item.parent_chunk_id}"
        if item.special_chunk_type:
            return f"special:{item.doc_id}:{item.chunk_id}"
        return f"chunk:{item.doc_id}:{item.chunk_id}"

    @classmethod
    def _clip_text(cls, text: str, budget: int) -> str:
        normalized_budget = max(budget, 1)
        tokens = cls._token_units(text)
        if len(tokens) <= normalized_budget:
            return text
        clipped = "".join(tokens[:normalized_budget]).strip()
        if not clipped:
            return ""
        return f"{clipped} ..."

    @classmethod
    def _token_units(cls, text: str) -> list[str]:
        return cls._findall(text)

    @staticmethod
    def _findall(text: str) -> list[str]:
        return _TOKEN_RE.findall(text)


__all__ = [
    "CandidateLike",
    "ContextEvidenceMerger",
    "ContextPromptBuildResult",
    "ContextPromptBuilder",
    "ContextTruncationResult",
    "EvidenceBundle",
    "EvidenceService",
    "EvidenceTruncator",
    "QueryUnderstandingService",
    "RoutingDecision",
    "RoutingService",
    "SelfCheckResult",
]
