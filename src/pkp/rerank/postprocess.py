from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from pkp.rerank.models import RerankResultItem
from pkp.types.query import QueryUnderstanding


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

        reranked = [
            item.model_copy(update={"rank_after": index})
            for index, item in enumerate(kept, start=1)
        ]
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
            (
                index
                for index in range(len(kept) - 1, -1, -1)
                if kept[index].chunk_type not in target_types
            ),
            -1,
        )
        if replacement_index < 0:
            return kept, dropped
        replaced = kept[replacement_index]
        updated_dropped = [
            item
            for item in dropped
            if item.chunk_id != selected.chunk_id
        ]
        updated_dropped.append(replaced.model_copy(update={"drop_reason": "special_diversity_replacement"}))
        kept[replacement_index] = selected
        return kept, updated_dropped

    @classmethod
    def _normalize(cls, text: str) -> str:
        return cls._WHITESPACE_RE.sub(" ", text.strip().lower())
