from __future__ import annotations

import re
from dataclasses import dataclass
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
        normalized_max_chunks = max(max_evidence_chunks, 1)
        selected: list[ContextEvidence] = []
        consumed = 0
        truncated_count = 0
        skipped_count = 0

        for item in evidence:
            if len(selected) >= normalized_max_chunks:
                skipped_count += 1
                continue
            remaining = normalized_budget - consumed
            if remaining <= 0:
                skipped_count += 1
                continue

            original_token_count = text_unit_count(item.text)
            selected_text = item.text
            selected_token_count = original_token_count
            was_truncated = False

            if original_token_count > remaining:
                clipped = self._clip_text(item.text, remaining)
                clipped_token_count = text_unit_count(clipped)
                if clipped.strip():
                    selected_text = clipped
                    selected_token_count = min(clipped_token_count, remaining)
                    was_truncated = clipped_token_count < original_token_count or clipped.endswith(" ...")
                else:
                    skipped_count += 1
                    continue

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
                truncated_count += 1

        truncated_count += skipped_count
        return ContextTruncationResult(
            evidence=selected,
            token_budget=normalized_budget,
            token_count=consumed,
            truncated_count=truncated_count,
        )

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
