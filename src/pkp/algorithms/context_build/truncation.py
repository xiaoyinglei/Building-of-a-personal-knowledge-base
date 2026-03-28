from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable
from typing import Final

from pkp.core.results import ContextEvidence
from pkp.types.envelope import EvidenceItem
from pkp.types.text import text_unit_count

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")


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
            selected_docs.add(best_item.doc_id)
            selected_groups.add(self._evidence_group(best_item))

        pick_best()
        if max_evidence_chunks >= 2:
            pick_best(lambda item: self._evidence_group(item) == "special")
        if max_evidence_chunks >= 3:
            pick_best(lambda item: self._evidence_group(item) == "graph")
        if max_evidence_chunks >= 4:
            pick_best(lambda item: self._evidence_group(item) == "external")
        while len(selected_indices) < max_evidence_chunks:
            before = len(selected_indices)
            pick_best()
            if len(selected_indices) == before:
                break

        selected_set = set(selected_indices)
        prioritized = [item for index, item in indexed_items if index in selected_set]
        prioritized.sort(
            key=lambda item: (
                self._group_priority(self._evidence_group(item)),
                -float(item.score),
                item.doc_id,
                item.chunk_id,
            )
        )
        return prioritized

    def _allocate_token_budgets(self, evidence: list[EvidenceItem], token_budget: int) -> list[int]:
        if not evidence:
            return []

        count = len(evidence)
        base_share = max(1, token_budget // count)
        budgets = [1 for _item in evidence]
        remaining_budget = max(0, token_budget - count)
        distribution_order = sorted(
            range(count),
            key=lambda index: (
                self._group_priority(self._evidence_group(evidence[index])),
                -float(evidence[index].score),
                index,
            ),
        )

        floor_targets = [
            min(text_unit_count(item.text), self._floor_target(item, base_share))
            for item in evidence
        ]
        remaining_budget = self._distribute_budget(budgets, floor_targets, remaining_budget, distribution_order)

        cap_targets = [
            min(text_unit_count(item.text), self._cap_target(item, base_share))
            for item in evidence
        ]
        remaining_budget = self._distribute_budget(budgets, cap_targets, remaining_budget, distribution_order)

        original_targets = [text_unit_count(item.text) for item in evidence]
        self._distribute_budget(budgets, original_targets, remaining_budget, distribution_order)
        return budgets

    @staticmethod
    def _distribute_budget(
        budgets: list[int],
        targets: list[int],
        remaining_budget: int,
        order: list[int],
    ) -> int:
        while remaining_budget > 0:
            changed = False
            for index in order:
                if budgets[index] >= targets[index]:
                    continue
                budgets[index] += 1
                remaining_budget -= 1
                changed = True
                if remaining_budget <= 0:
                    break
            if not changed:
                break
        return remaining_budget

    def _floor_target(self, item: EvidenceItem, base_share: int) -> int:
        group = self._evidence_group(item)
        if group == "special":
            return max(24, int(base_share * 0.55))
        if group == "graph":
            return max(20, int(base_share * 0.50))
        if group == "external":
            return max(16, int(base_share * 0.35))
        return max(18, int(base_share * 0.45))

    def _cap_target(self, item: EvidenceItem, base_share: int) -> int:
        group = self._evidence_group(item)
        if group == "special":
            multiplier = 1.45 if (item.special_chunk_type or "") in {"table", "figure", "caption", "ocr_region", "image_summary"} else 1.25
            return max(self._floor_target(item, base_share), int(base_share * multiplier))
        if group == "graph":
            return max(self._floor_target(item, base_share), int(base_share * 1.20))
        if group == "external":
            return max(self._floor_target(item, base_share), int(base_share * 0.90))
        return max(self._floor_target(item, base_share), int(base_share * 1.10))

    def _selection_score(
        self,
        item: EvidenceItem,
        *,
        original_index: int,
        selected_docs: set[str],
        selected_groups: set[str],
    ) -> float:
        group = self._evidence_group(item)
        score = float(item.score)
        if group == "special":
            score += 0.35
        elif group == "graph":
            score += 0.22
        elif group == "internal":
            score += 0.08
        else:
            score -= 0.02
        if item.doc_id not in selected_docs:
            score += 0.06
        if group not in selected_groups:
            score += 0.04
        if (item.special_chunk_type or "") in {"table", "figure", "caption", "ocr_region", "image_summary"}:
            score += 0.08
        return score - original_index * 1e-6

    @staticmethod
    def _evidence_group(item: EvidenceItem) -> str:
        if item.special_chunk_type:
            return "special"
        if item.evidence_kind == "graph":
            return "graph"
        if item.evidence_kind == "external":
            return "external"
        return "internal"

    @staticmethod
    def _group_priority(group: str) -> int:
        priorities = {
            "special": 0,
            "graph": 1,
            "internal": 2,
            "external": 3,
        }
        return priorities.get(group, 9)

    @staticmethod
    def _clip_text(text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        end = 0
        units = 0
        for match in _TOKEN_RE.finditer(text):
            units += 1
            if units > budget:
                break
            end = match.end()
        if end <= 0:
            stripped = text.strip()
            return stripped[:budget].strip()
        clipped = text[:end].strip()
        return clipped if end >= len(text) else f"{clipped} ..."
