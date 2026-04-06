from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.llm.generation import AnswerGenerationService
from rag.query.analysis import section_family_aliases
from rag.schema._types.access import AccessPolicy, RuntimeMode
from rag.schema._types.content import ChunkRole
from rag.schema._types.envelope import EvidenceItem
from rag.schema._types.query import ComplexityLevel, QueryUnderstanding, TaskType
from rag.schema._types.text import (
    DEFAULT_TOKENIZER_FALLBACK_MODEL,
    TokenAccountingService,
    TokenizerContract,
)

if TYPE_CHECKING:
    from rag.query.query import ContextEvidence
    from rag.schema._types.retrieval import RetrievalResult

_RETRIEVAL_FAMILIES: Final[tuple[str, ...]] = ("kg", "vector", "multimodal", "external")


def _default_token_accounting() -> TokenAccountingService:
    return TokenAccountingService(
        TokenizerContract(
            embedding_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            chunking_tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
        )
    )


class CandidateLike(Protocol):
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str
    source_id: str | None
    section_path: Sequence[str]
    chunk_role: ChunkRole | None
    special_chunk_type: str | None
    parent_chunk_id: str | None


def classify_retrieval_family(
    *,
    evidence_kind: str,
    special_chunk_type: str | None,
    retrieval_channels: Sequence[str] = (),
) -> str:
    channels = {channel for channel in retrieval_channels if channel}
    if evidence_kind == "external":
        return "external"
    if special_chunk_type is not None or channels & {"special", "section", "metadata"}:
        return "multimodal"
    if evidence_kind == "graph" or channels & {"local", "global"}:
        return "kg"
    if channels & {"vector", "full_text"}:
        return "vector"
    return "kg" if evidence_kind == "graph" else "vector"


class EvidenceThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    fast_min_evidence_chunks: int = 2
    fast_min_sections: int = 1
    deep_min_evidence_chunks: int = 4
    deep_min_supporting_units: int = 2


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    internal: list[EvidenceItem] = Field(default_factory=list)
    external: list[EvidenceItem] = Field(default_factory=list)
    graph: list[EvidenceItem] = Field(default_factory=list)

    @property
    def all(self) -> list[EvidenceItem]:
        return [*self.internal, *self.external, *self.graph]


class SelfCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieve_more: bool
    evidence_sufficient: bool
    claim_supported: bool


class EvidenceService:
    def __init__(self, thresholds: EvidenceThresholds | None = None) -> None:
        self._thresholds = thresholds or EvidenceThresholds()

    @staticmethod
    def _candidate_source_scope(candidate: CandidateLike) -> set[str]:
        scope = {candidate.doc_id}
        if candidate.source_id:
            scope.add(candidate.source_id)
        return scope

    @staticmethod
    def _candidate_access_policy(candidate: object) -> AccessPolicy | None:
        policy = getattr(candidate, "effective_access_policy", None)
        if isinstance(policy, AccessPolicy):
            return policy
        return None

    @staticmethod
    def _is_candidate_external(candidate: object) -> bool:
        return getattr(candidate, "source_kind", "internal") == "external"

    @staticmethod
    def _is_candidate_graph(candidate: object) -> bool:
        return getattr(candidate, "source_kind", "internal") == "graph"

    def filter_candidates(
        self,
        candidates: Sequence[CandidateLike],
        *,
        source_scope: Sequence[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding | None = None,
    ) -> list[CandidateLike]:
        allowed_scope = set(source_scope)
        filtered: list[CandidateLike] = []
        for candidate in candidates:
            if allowed_scope and not self._candidate_source_scope(candidate) & allowed_scope:
                continue
            if self._is_candidate_external(candidate) and access_policy.external_retrieval.value != "allow":
                continue
            candidate_policy = self._candidate_access_policy(candidate)
            if candidate_policy is not None:
                if runtime_mode not in candidate_policy.allowed_runtimes:
                    continue
                if not (candidate_policy.allowed_locations & access_policy.allowed_locations):
                    continue
            if query_understanding is not None and not self._matches_explicit_constraints(
                candidate, query_understanding
            ):
                continue
            filtered.append(candidate)
        return filtered

    @staticmethod
    def _matches_explicit_constraints(candidate: CandidateLike, query_understanding: QueryUnderstanding) -> bool:
        if getattr(candidate, "source_kind", "internal") == "external":
            return True
        if not query_understanding.has_explicit_constraints():
            return True
        if not EvidenceService._matches_metadata_constraints(candidate, query_understanding):
            return False
        if not EvidenceService._matches_structure_constraints(candidate, query_understanding):
            return False
        return True

    @staticmethod
    def _matches_metadata_constraints(candidate: CandidateLike, query_understanding: QueryUnderstanding) -> bool:
        metadata_filters = query_understanding.metadata_filters
        candidate_metadata = getattr(candidate, "metadata", {}) or {}
        candidate_source_type = getattr(candidate, "source_type", None) or candidate_metadata.get("source_type")
        if metadata_filters.source_types and candidate_source_type is not None:
            if candidate_source_type not in metadata_filters.source_types:
                return False

        candidate_file_name = getattr(candidate, "file_name", None)
        if metadata_filters.file_names and candidate_file_name is not None:
            if candidate_file_name not in metadata_filters.file_names:
                return False
        if metadata_filters.document_titles and candidate_file_name is not None:
            if candidate_file_name not in metadata_filters.document_titles:
                return False

        if not metadata_filters.page_numbers and not metadata_filters.page_ranges:
            return True
        candidate_pages = EvidenceService._candidate_pages(candidate)
        if not candidate_pages:
            return True
        if metadata_filters.page_numbers and not candidate_pages & set(metadata_filters.page_numbers):
            if not metadata_filters.page_ranges:
                return False
        if metadata_filters.page_ranges and not any(
            any(page_range.start <= page <= page_range.end for page in candidate_pages)
            for page_range in metadata_filters.page_ranges
        ):
            if not metadata_filters.page_numbers:
                return False
            if not candidate_pages & set(metadata_filters.page_numbers):
                return False
        return True

    @staticmethod
    def _matches_structure_constraints(candidate: CandidateLike, query_understanding: QueryUnderstanding) -> bool:
        constraints = query_understanding.structure_constraints
        if not constraints.requires_structure_match:
            return True
        section_path = tuple(getattr(candidate, "section_path", ()) or ())
        if not section_path:
            return True
        section_text = " ".join(section_path).lower()
        preferred_terms = {term.lower() for term in query_understanding.preferred_section_terms}
        heading_hints = {hint.lower() for hint in constraints.heading_hints}
        title_hints = {hint.lower() for hint in constraints.title_hints}
        semantic_aliases = {
            alias.lower()
            for family in constraints.semantic_section_families
            for alias in section_family_aliases(family)
        }
        match_terms = preferred_terms | heading_hints | title_hints | semantic_aliases
        if not match_terms:
            return True
        matched = any(term in section_text for term in match_terms)
        if constraints.prefer_heading_match:
            return matched
        return matched

    @staticmethod
    def _candidate_pages(candidate: CandidateLike) -> set[int]:
        metadata = getattr(candidate, "metadata", {}) or {}
        pages: set[int] = set()
        page_no = metadata.get("page_no")
        if isinstance(page_no, str) and page_no.isdigit():
            pages.add(int(page_no))
        page_start = getattr(candidate, "page_start", None)
        page_end = getattr(candidate, "page_end", None)
        if isinstance(page_start, int) and isinstance(page_end, int):
            pages.update(range(page_start, page_end + 1))
        elif isinstance(page_start, int):
            pages.add(page_start)
        elif isinstance(page_end, int):
            pages.add(page_end)
        return pages

    @staticmethod
    def _to_evidence_item(candidate: CandidateLike) -> EvidenceItem:
        evidence_kind = getattr(candidate, "source_kind", "internal")
        if evidence_kind not in {"internal", "external", "graph"}:
            evidence_kind = "internal"
        retrieval_channels = list(getattr(candidate, "retrieval_channels", []) or [])
        retrieval_family = classify_retrieval_family(
            evidence_kind=evidence_kind,
            special_chunk_type=getattr(candidate, "special_chunk_type", None),
            retrieval_channels=retrieval_channels,
        )
        return EvidenceItem(
            chunk_id=candidate.chunk_id,
            doc_id=candidate.doc_id,
            source_id=getattr(candidate, "source_id", None),
            citation_anchor=candidate.citation_anchor,
            text=candidate.text,
            score=float(candidate.score),
            evidence_kind=evidence_kind,
            chunk_role=getattr(candidate, "chunk_role", None),
            special_chunk_type=getattr(candidate, "special_chunk_type", None),
            parent_chunk_id=getattr(candidate, "parent_chunk_id", None),
            file_name=getattr(candidate, "file_name", None),
            section_path=list(getattr(candidate, "section_path", ()) or ()),
            page_start=getattr(candidate, "page_start", None),
            page_end=getattr(candidate, "page_end", None),
            chunk_type=getattr(candidate, "chunk_type", None),
            source_type=getattr(candidate, "source_type", None),
            retrieval_channels=retrieval_channels,
            retrieval_family=retrieval_family,
        )

    def assemble_bundle(self, candidates: Sequence[CandidateLike]) -> EvidenceBundle:
        internal: list[EvidenceItem] = []
        external: list[EvidenceItem] = []
        graph: list[EvidenceItem] = []
        for candidate in candidates:
            item = self._to_evidence_item(candidate)
            if item.evidence_kind == "external":
                external.append(item)
            elif item.evidence_kind == "graph":
                graph.append(item)
            else:
                internal.append(item)
        return EvidenceBundle(internal=internal, external=external, graph=graph)

    def evaluate_self_check(
        self,
        *,
        bundle: EvidenceBundle,
        task_type: TaskType,
        complexity_level: ComplexityLevel,
    ) -> SelfCheckResult:
        internal = bundle.internal
        section_keys = {item.citation_anchor if item.citation_anchor else item.chunk_id for item in internal}
        doc_ids = {item.doc_id for item in internal}

        if task_type in {TaskType.LOOKUP, TaskType.SINGLE_DOC_QA} or complexity_level in {
            ComplexityLevel.L1_DIRECT,
            ComplexityLevel.L2_SCOPED,
        }:
            evidence_sufficient = (
                len(internal) >= self._thresholds.fast_min_evidence_chunks
                and len(section_keys) >= self._thresholds.fast_min_sections
            )
        else:
            evidence_sufficient = len(internal) >= self._thresholds.deep_min_evidence_chunks and (
                len(doc_ids) >= self._thresholds.deep_min_supporting_units
                or len(section_keys) >= self._thresholds.deep_min_supporting_units
            )

        claim_supported = evidence_sufficient and bool(internal)
        retrieve_more = not evidence_sufficient
        return SelfCheckResult(
            retrieve_more=retrieve_more,
            evidence_sufficient=evidence_sufficient,
            claim_supported=claim_supported,
        )

    @staticmethod
    def evidence_counts(bundle: EvidenceBundle) -> Counter[str]:
        return Counter(item.evidence_kind for item in bundle.all)


@dataclass(slots=True)
class ContextEvidenceMerger:
    def merge(self, retrieval: RetrievalResult) -> list[EvidenceItem]:
        internal_by_id = {item.chunk_id: item for item in retrieval.evidence.internal}
        ordered_internal = [
            internal_by_id[chunk_id] for chunk_id in retrieval.reranked_chunk_ids if chunk_id in internal_by_id
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

        merged_kind = (
            "internal" if "internal" in {existing.evidence_kind, incoming.evidence_kind} else preferred.evidence_kind
        )
        merged_text = preferred.text if len(preferred.text) >= len(secondary.text) else secondary.text
        merged_channels = list(
            dict.fromkeys([*existing.retrieval_channels, *incoming.retrieval_channels])
        )
        merged_family = classify_retrieval_family(
            evidence_kind=merged_kind,
            special_chunk_type=preferred.special_chunk_type or secondary.special_chunk_type,
            retrieval_channels=merged_channels,
        )

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
                "retrieval_channels": merged_channels,
                "retrieval_family": merged_family,
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
        from rag.query.query import ContextEvidence

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
        from rag.query.query import QueryMode, normalize_query_mode

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
        from rag.query.query import QueryMode, normalize_query_mode

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
    "CandidateLike",
    "ContextEvidenceMerger",
    "ContextPromptBuildResult",
    "ContextPromptBuilder",
    "ContextTruncationResult",
    "EvidenceBundle",
    "EvidenceService",
    "EvidenceTruncator",
    "SelfCheckResult",
]
