from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.llm.generation import AnswerGenerationService
from rag.query.policies import RoutingThresholds
from rag.schema._types.access import AccessPolicy, RuntimeMode
from rag.schema._types.content import ChunkRole
from rag.schema._types.envelope import EvidenceItem
from rag.schema._types.query import ComplexityLevel, TaskType
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


@dataclass(frozen=True, slots=True)
class ContextAssemblyPolicy:
    family_order: tuple[str, ...]
    family_chunk_targets: dict[str, int]
    family_token_shares: dict[str, float]


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
    def __init__(self, thresholds: RoutingThresholds | None = None) -> None:
        self._thresholds = thresholds or RoutingThresholds()

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
            filtered.append(candidate)
        return filtered

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
        policy = self._build_context_policy(evidence=evidence, mode=mode, max_evidence_chunks=normalized_max_chunks)
        prioritized_items = self._prioritize_evidence(evidence, normalized_max_chunks, policy=policy)
        assigned_budgets = self._allocate_family_token_budgets(
            prioritized_items,
            token_budget=normalized_budget,
            policy=policy,
        )

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
        policy: ContextAssemblyPolicy,
    ) -> list[EvidenceItem]:
        if len(evidence) <= max_evidence_chunks:
            return list(evidence)

        indexed_items = list(enumerate(evidence))
        selected_indices: list[int] = []
        selected_docs: set[str] = set()
        selected_groups: set[str] = set()
        family_counts: Counter[str] = Counter()

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
            family_counts[self._evidence_family(best_item)] += 1

        for family in policy.family_order:
            target = policy.family_chunk_targets.get(family, 0)
            while family_counts[family] < target and len(selected_indices) < max_evidence_chunks:
                previous_count = len(selected_indices)
                def belongs_to_family(item: EvidenceItem, *, family: str = family) -> bool:
                    return self._evidence_family(item) == family

                pick_best(belongs_to_family)
                if len(selected_indices) == previous_count:
                    break
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

    def _allocate_family_token_budgets(
        self,
        evidence: list[EvidenceItem],
        *,
        token_budget: int,
        policy: ContextAssemblyPolicy,
    ) -> list[int]:
        if not evidence:
            return []

        family_groups: dict[str, list[tuple[int, EvidenceItem]]] = defaultdict(list)
        for index, item in enumerate(evidence):
            family_groups[self._evidence_family(item)].append((index, item))

        family_budgets = self._family_budget_pools(
            family_groups=family_groups,
            token_budget=token_budget,
            policy=policy,
        )
        assigned = [1] * len(evidence)
        for family, indexed_items in family_groups.items():
            indices = [index for index, _item in indexed_items]
            items = [item for _index, item in indexed_items]
            budgets = self._allocate_token_budgets(items, family_budgets[family])
            for index, budget in zip(indices, budgets, strict=False):
                assigned[index] = budget
        return assigned

    def _family_budget_pools(
        self,
        *,
        family_groups: dict[str, list[tuple[int, EvidenceItem]]],
        token_budget: int,
        policy: ContextAssemblyPolicy,
    ) -> dict[str, int]:
        present_families = [family for family in policy.family_order if family in family_groups]
        remaining_families = [family for family in family_groups if family not in present_families]
        ordered_families = [*present_families, *sorted(remaining_families)]
        base_budgets = {family: len(items) for family, items in family_groups.items()}
        remaining_budget = token_budget - sum(base_budgets.values())
        if remaining_budget <= 0:
            return base_budgets

        weights = {
            family: max(policy.family_token_shares.get(family, 0.0), 0.0)
            for family in ordered_families
        }
        if not any(weight > 0.0 for weight in weights.values()):
            weights = {family: 1.0 for family in ordered_families}
        total_weight = sum(weights.values()) or 1.0

        allocated = dict(base_budgets)
        fractional: list[tuple[float, str]] = []
        for family in ordered_families:
            raw_extra = remaining_budget * (weights[family] / total_weight)
            extra = int(raw_extra)
            allocated[family] += extra
            fractional.append((raw_extra - extra, family))

        remainder = token_budget - sum(allocated.values())
        for _fraction, family in sorted(fractional, key=lambda item: (-item[0], ordered_families.index(item[1]))):
            if remainder <= 0:
                break
            allocated[family] += 1
            remainder -= 1
        return allocated

    def _build_context_policy(
        self,
        *,
        evidence: list[EvidenceItem],
        mode: str,
        max_evidence_chunks: int,
    ) -> ContextAssemblyPolicy:
        from rag.query.query import QueryMode, normalize_query_mode

        resolved_mode = normalize_query_mode(mode)
        family_shares_by_mode: dict[QueryMode, dict[str, float]] = {
            QueryMode.BYPASS: {"kg": 0.0, "vector": 0.0, "multimodal": 0.0, "external": 0.0},
            QueryMode.NAIVE: {"kg": 0.1, "vector": 0.8, "multimodal": 0.1, "external": 0.0},
            QueryMode.LOCAL: {"kg": 0.78, "vector": 0.0, "multimodal": 0.22, "external": 0.0},
            QueryMode.GLOBAL: {"kg": 0.74, "vector": 0.0, "multimodal": 0.26, "external": 0.0},
            QueryMode.HYBRID: {"kg": 0.66, "vector": 0.0, "multimodal": 0.28, "external": 0.06},
            QueryMode.MIX: {"kg": 0.45, "vector": 0.35, "multimodal": 0.15, "external": 0.05},
        }
        family_order_by_mode: dict[QueryMode, tuple[str, ...]] = {
            QueryMode.BYPASS: ("vector", "kg", "multimodal", "external"),
            QueryMode.NAIVE: ("vector", "multimodal", "kg", "external"),
            QueryMode.LOCAL: ("kg", "multimodal", "vector", "external"),
            QueryMode.GLOBAL: ("kg", "multimodal", "vector", "external"),
            QueryMode.HYBRID: ("kg", "multimodal", "vector", "external"),
            QueryMode.MIX: ("kg", "vector", "multimodal", "external"),
        }
        family_order = family_order_by_mode[resolved_mode]
        shares = family_shares_by_mode[resolved_mode]
        available_families = {self._evidence_family(item) for item in evidence}
        family_chunk_targets = self._allocate_family_chunk_targets(
            available_families=available_families,
            family_order=family_order,
            family_shares=shares,
            max_evidence_chunks=max_evidence_chunks,
        )
        return ContextAssemblyPolicy(
            family_order=family_order,
            family_chunk_targets=family_chunk_targets,
            family_token_shares=shares,
        )

    @staticmethod
    def _allocate_family_chunk_targets(
        *,
        available_families: set[str],
        family_order: tuple[str, ...],
        family_shares: dict[str, float],
        max_evidence_chunks: int,
    ) -> dict[str, int]:
        active_families = [family for family in family_order if family in available_families]
        if not active_families or max_evidence_chunks <= 0:
            return {}
        targets = {family: 0 for family in active_families}
        remaining = max_evidence_chunks
        for family in active_families:
            if remaining <= 0:
                break
            if family_shares.get(family, 0.0) > 0.0:
                targets[family] += 1
                remaining -= 1
        if remaining <= 0:
            return targets

        raw_allocations: dict[str, float] = {
            family: remaining * family_shares.get(family, 0.0) for family in active_families
        }
        for family, raw in raw_allocations.items():
            extra = int(raw)
            targets[family] += extra
            remaining -= extra
        if remaining <= 0:
            return targets
        for family in sorted(
            active_families,
            key=lambda family: (-(raw_allocations.get(family, 0.0) % 1), family_order.index(family)),
        ):
            if remaining <= 0:
                break
            targets[family] += 1
            remaining -= 1
        return targets

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
        retrieval_family = self._evidence_family(item)
        kind_bonus = 0.2 if item.evidence_kind == "internal" else 0.0
        if retrieval_family == "kg":
            kind_bonus += 0.12
        if retrieval_family == "multimodal":
            kind_bonus += 0.15
        if item.special_chunk_type:
            kind_bonus += 0.08
        return (score + novelty_bonus + group_bonus + kind_bonus, score, kind_bonus, -original_index)

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
