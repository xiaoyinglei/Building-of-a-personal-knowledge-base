from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rag.assembly import TokenAccountingService, TokenizerContract
from rag.providers.generation import AnswerGenerationService
from rag.retrieval.evidence import classify_retrieval_family
from rag.schema.query import EvidenceItem
from rag.schema.runtime import RuntimeMode
from rag.utils.text import DEFAULT_TOKENIZER_FALLBACK_MODEL

if TYPE_CHECKING:
    from rag.retrieval.models import ContextEvidence


def _default_token_accounting() -> TokenAccountingService:
    return TokenAccountingService(
        TokenizerContract(
            embedding_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            chunking_tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
        )
    )


@dataclass(frozen=True, slots=True)
class ContextPromptBuildResult:
    grounded_candidate: str
    prompt: str
    token_count: int


@dataclass(slots=True)
class ContextPromptBuilder:
    answer_generation_service: AnswerGenerationService
    token_accounting: TokenAccountingService = field(default_factory=_default_token_accounting)

    def build(
        self,
        *,
        query: str,
        grounded_candidate: str,
        evidence: list[ContextEvidence],
        runtime_mode: RuntimeMode,
        response_type: str,
        user_prompt: str | None,
        conversation_history: Sequence[tuple[str, str]],
        prompt_style: Literal["full", "compact", "minimal"] = "full",
    ) -> ContextPromptBuildResult:
        prompt = self.answer_generation_service.build_prompt(
            query=query,
            evidence_pack=[item.as_evidence_item() for item in evidence],
            grounded_candidate=grounded_candidate,
            runtime_mode=runtime_mode,
            response_type=response_type,
            user_prompt=user_prompt,
            conversation_history=conversation_history,
            prompt_style=prompt_style,
        )
        return ContextPromptBuildResult(
            grounded_candidate=grounded_candidate,
            prompt=prompt,
            token_count=self.token_accounting.count(prompt),
        )


@dataclass(frozen=True, slots=True)
class ContextTruncationResult:
    evidence: list[ContextEvidence]
    token_budget: int
    token_count: int
    truncated_count: int


@dataclass(slots=True)
class EvidenceTruncator:
    token_accounting: TokenAccountingService = field(default_factory=_default_token_accounting)

    def truncate(
        self,
        evidence: list[EvidenceItem],
        *,
        token_budget: int,
        max_evidence_chunks: int,
        mode: str = "mix",
    ) -> ContextTruncationResult:
        from rag.retrieval.models import ContextEvidence

        normalized_budget = max(token_budget, 1)
        normalized_max_chunks = min(max(max_evidence_chunks, 1), normalized_budget)
        family_order = self._family_order(mode)
        coverage_order = self._family_coverage_order(mode)
        prioritized_items = self._prioritize_evidence(
            evidence,
            normalized_max_chunks,
            coverage_order=coverage_order,
            family_order=family_order,
        )
        assigned_budgets = self._allocate_token_budgets(prioritized_items, token_budget=normalized_budget)

        selected: list[ContextEvidence] = []
        consumed = 0
        clipped_count = 0

        for item, item_budget in zip(prioritized_items, assigned_budgets, strict=False):
            original_token_count = self.token_accounting.count(item.text)
            effective_budget = max(item_budget, 1)
            selected_text = item.text
            selected_token_count = original_token_count
            was_truncated = False

            if original_token_count > effective_budget:
                clipped = self._clip_text(item.text, effective_budget)
                clipped_token_count = self.token_accounting.count(clipped)
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
                    benchmark_doc_id=item.benchmark_doc_id,
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
                    retrieval_channels=list(item.retrieval_channels),
                    retrieval_family=self._evidence_family(item),
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

    def _prioritize_evidence(
        self,
        evidence: list[EvidenceItem],
        max_evidence_chunks: int,
        *,
        coverage_order: tuple[str, ...],
        family_order: tuple[str, ...],
    ) -> list[EvidenceItem]:
        if len(evidence) <= max_evidence_chunks:
            return list(evidence)

        indexed_items = list(enumerate(evidence))
        selected_indices: list[int] = []
        selected_docs: set[str] = set()
        selected_groups: set[str] = set()
        family_priority = {
            family: len(family_order) - position
            for position, family in enumerate(family_order)
        }

        def select(index: int, item: EvidenceItem) -> None:
            selected_indices.append(index)
            if item.doc_id:
                selected_docs.add(item.doc_id)
            selected_groups.add(self._group_key(item))

        for family in coverage_order:
            if len(selected_indices) >= max_evidence_chunks:
                break
            family_candidates = [
                (index, item)
                for index, item in indexed_items
                if index not in selected_indices and self._evidence_family(item) == family
            ]
            if not family_candidates:
                continue
            best_index, best_item = max(
                family_candidates,
                key=lambda pair: self._selection_key(
                    pair[1],
                    original_index=pair[0],
                    family_priority=family_priority,
                    selected_docs=selected_docs,
                    selected_groups=selected_groups,
                ),
            )
            select(best_index, best_item)

        remaining = sorted(
            [
                (index, item)
                for index, item in indexed_items
                if index not in selected_indices
            ],
            key=lambda pair: self._selection_key(
                pair[1],
                original_index=pair[0],
                family_priority=family_priority,
                selected_docs=selected_docs,
                selected_groups=selected_groups,
            ),
            reverse=True,
        )
        for index, item in remaining:
            if len(selected_indices) >= max_evidence_chunks:
                break
            select(index, item)

        return [evidence[index] for index in sorted(selected_indices)]

    def _allocate_token_budgets(self, evidence: list[EvidenceItem], token_budget: int) -> list[int]:
        if not evidence:
            return []
        assigned = [1] * len(evidence)
        remaining = max(token_budget - len(evidence), 0)
        desired_counts = [max(self.token_accounting.count(item.text), 1) for item in evidence]
        ranked_indices = sorted(
            range(len(evidence)),
            key=lambda index: self._budget_priority(evidence[index], original_index=index),
            reverse=True,
        )
        while remaining > 0:
            progress = False
            for index in ranked_indices:
                if assigned[index] >= desired_counts[index]:
                    continue
                assigned[index] += 1
                remaining -= 1
                progress = True
                if remaining <= 0:
                    break
            if not progress:
                break
        return assigned

    @staticmethod
    def _family_order(mode: str) -> tuple[str, ...]:
        from rag.retrieval.models import QueryMode, normalize_query_mode

        resolved_mode = normalize_query_mode(mode)
        family_order_by_mode: dict[QueryMode, tuple[str, ...]] = {
            QueryMode.BYPASS: ("vector", "kg", "multimodal", "external"),
            QueryMode.NAIVE: ("vector", "multimodal", "kg", "external"),
            QueryMode.LOCAL: ("kg", "multimodal", "vector", "external"),
            QueryMode.GLOBAL: ("kg", "multimodal", "vector", "external"),
            QueryMode.HYBRID: ("kg", "multimodal", "vector", "external"),
            QueryMode.MIX: ("kg", "vector", "multimodal", "external"),
        }
        return family_order_by_mode[resolved_mode]

    @staticmethod
    def _family_coverage_order(mode: str) -> tuple[str, ...]:
        from rag.retrieval.models import QueryMode, normalize_query_mode

        resolved_mode = normalize_query_mode(mode)
        coverage_order_by_mode: dict[QueryMode, tuple[str, ...]] = {
            QueryMode.BYPASS: ("vector",),
            QueryMode.NAIVE: ("vector",),
            QueryMode.LOCAL: ("kg", "multimodal"),
            QueryMode.GLOBAL: ("kg", "multimodal"),
            QueryMode.HYBRID: ("kg", "multimodal"),
            QueryMode.MIX: ("kg", "vector", "multimodal"),
        }
        return coverage_order_by_mode[resolved_mode]

    def _budget_priority(
        self,
        item: EvidenceItem,
        *,
        original_index: int,
    ) -> tuple[float, int, int, int, int]:
        return (
            max(float(item.score), 0.0),
            int(item.evidence_kind == "internal"),
            int(bool(item.special_chunk_type)),
            int(item.page_start is not None),
            -original_index,
        )

    def _selection_key(
        self,
        item: EvidenceItem,
        *,
        original_index: int,
        family_priority: dict[str, int],
        selected_docs: set[str],
        selected_groups: set[str],
    ) -> tuple[int, float, int, int, int, int]:
        return (
            family_priority.get(self._evidence_family(item), 0),
            max(float(item.score), 0.0),
            int(self._group_key(item) not in selected_groups),
            int(bool(item.doc_id) and item.doc_id not in selected_docs),
            int(item.evidence_kind == "internal"),
            -original_index,
        )

    @staticmethod
    def _group_key(item: EvidenceItem) -> str:
        if item.parent_chunk_id:
            return f"parent:{item.doc_id}:{item.parent_chunk_id}"
        if item.special_chunk_type:
            return f"special:{item.doc_id}:{item.chunk_id}"
        return f"chunk:{item.doc_id}:{item.chunk_id}"

    def _clip_text(self, text: str, budget: int) -> str:
        return self.token_accounting.clip(text, budget, add_ellipsis=True)

    @staticmethod
    def _evidence_family(item: EvidenceItem) -> str:
        return item.retrieval_family or classify_retrieval_family(
            evidence_kind=item.evidence_kind,
            special_chunk_type=item.special_chunk_type,
            retrieval_channels=item.retrieval_channels,
        )


__all__ = [
    "ContextPromptBuildResult",
    "ContextPromptBuilder",
    "ContextTruncationResult",
    "EvidenceTruncator",
]
