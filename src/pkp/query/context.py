from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from pkp.config.policies import RoutingThresholds
from pkp.llm.generation import AnswerGenerationService
from pkp.types.access import AccessPolicy, RuntimeMode
from pkp.types.content import ChunkRole
from pkp.types.envelope import EvidenceItem
from pkp.types.query import ComplexityLevel, QueryUnderstanding, TaskType
from pkp.types.text import focus_terms, text_unit_count

if TYPE_CHECKING:
    from pkp.query.query import ContextEvidence
    from pkp.types.retrieval import RetrievalResult

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")
_COMPARISON_TOKENS = ("compare", "versus", " vs ", "difference", "contrast")
_TIMELINE_TOKENS = ("timeline", "trend", "over time", "chronology")
_RESEARCH_TOKENS = ("research", "synthesize", "summary", "summarize", "why", "how ")
_SCOPED_TOKENS = ("this document", "this source", "in this doc")


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


class QueryUnderstandingService:
    _PAGE_PATTERN = re.compile(r"(?:第\s*(\d+)\s*页|page\s*(\d+))", re.IGNORECASE)
    _SPECIAL_SIGNAL_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("table", ("表格", "table", "指标", "数值", "统计表")),
        ("figure", ("图片", "图", "figure", "截图", "流程图")),
        ("ocr_region", ("ocr", "识别", "截图文字", "图中文字", "区域文字")),
        ("image_summary", ("图片总结", "图像摘要", "画面内容", "visual summary", "image summary")),
        ("caption", ("图注", "图题", "caption")),
        ("formula", ("公式", "equation", "formula", "latex", "数学表达式")),
    )
    _STRUCTURE_TERMS: tuple[str, ...] = (
        "架构",
        "结构",
        "模块",
        "章节",
        "标题",
        "目录",
        "分层",
        "第几层",
        "哪几层",
        "section",
        "heading",
        "module",
        "architecture",
    )
    _SOURCE_TYPE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("pdf", ("pdf", "pdf文档", "扫描件")),
        ("markdown", ("markdown", "md", "markdown文档")),
        ("docx", ("docx", "word", "word文档", "文档")),
        ("image", ("图片", "图像", "image", "截图")),
    )

    def analyze(self, query: str) -> QueryUnderstanding:
        normalized = query.strip()
        lowered = normalized.lower()
        special_targets = [
            target
            for target, keywords in self._SPECIAL_SIGNAL_MAP
            if any(keyword in lowered for keyword in keywords)
        ]
        structure_hit = any(term in normalized or term in lowered for term in self._STRUCTURE_TERMS)
        preferred_sections = self._preferred_section_terms(normalized)
        page_numbers = self._page_numbers(normalized)
        source_types = [
            source_type
            for source_type, keywords in self._SOURCE_TYPE_TERMS
            if any(keyword in lowered for keyword in keywords)
        ]
        metadata_filters: dict[str, list[str] | str | bool] = {}
        if page_numbers:
            metadata_filters["page_numbers"] = page_numbers
        if source_types:
            metadata_filters["source_types"] = source_types
        if special_targets:
            metadata_filters["special_targets"] = special_targets
        if preferred_sections:
            metadata_filters["preferred_section_terms"] = preferred_sections
        needs_metadata = bool(page_numbers or source_types)

        confidence = 0.35
        if special_targets:
            confidence += 0.25
        if structure_hit or preferred_sections:
            confidence += 0.2
        if needs_metadata:
            confidence += 0.15
        if len(normalized) >= 8:
            confidence += 0.1
        confidence = min(confidence, 0.95)

        if needs_metadata:
            intent = "localized_lookup"
            query_type = "page_lookup" if page_numbers else "metadata_lookup"
        elif special_targets:
            intent = "special_lookup"
            query_type = special_targets[0]
        elif structure_hit or preferred_sections:
            intent = "structure_lookup"
            query_type = "structure"
        else:
            intent = "semantic_lookup"
            query_type = "general"

        return QueryUnderstanding(
            intent=intent,
            query_type=query_type,
            needs_dense=True,
            needs_sparse=True,
            needs_special=bool(special_targets),
            needs_structure=structure_hit or bool(preferred_sections),
            needs_metadata=needs_metadata,
            structure_constraints={
                "preferred_section_terms": preferred_sections,
                "prefer_heading_match": structure_hit or bool(preferred_sections),
            },
            metadata_filters=metadata_filters,
            special_targets=special_targets,
            confidence=confidence,
        )

    @staticmethod
    def _preferred_section_terms(query: str) -> list[str]:
        exact_phrases = [
            phrase
            for phrase in ("系统架构", "技术架构", "组织架构", "系统设计", "工作总结", "项目计划")
            if phrase in query
        ]
        candidates = [
            term
            for term in focus_terms(query)
            if len(term) >= 2 and term not in {"什么", "哪些", "多少", "如何", "为什么"}
        ]
        section_terms: list[str] = list(exact_phrases)
        for candidate in candidates:
            if any(candidate != phrase and candidate in phrase for phrase in exact_phrases):
                continue
            if candidate.endswith(("架构", "结构", "模块", "章节", "流程", "总结", "工作")):
                section_terms.append(candidate)
        seen: set[str] = set()
        ordered: list[str] = []
        for term in section_terms:
            if term in seen:
                continue
            seen.add(term)
            ordered.append(term)
        return ordered[:3]

    @classmethod
    def _page_numbers(cls, query: str) -> list[str]:
        numbers: list[str] = []
        for direct, english in cls._PAGE_PATTERN.findall(query):
            value = direct or english
            if value and value not in numbers:
                numbers.append(value)
        return numbers


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType
    complexity_level: ComplexityLevel
    runtime_mode: RuntimeMode
    source_scope: list[str] = Field(default_factory=list)
    web_search_allowed: bool = False
    graph_expansion_allowed: bool = False
    rerank_required: bool = True


_DEEP_TASK_TYPES = {
    TaskType.COMPARISON,
    TaskType.SYNTHESIS,
    TaskType.TIMELINE,
    TaskType.RESEARCH,
}


class RoutingService:
    def __init__(self, thresholds: RoutingThresholds | None = None) -> None:
        self._thresholds = thresholds or RoutingThresholds()

    @staticmethod
    def _normalized_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def _classify_task_type(self, query: str, source_scope: Sequence[str]) -> TaskType:
        normalized = self._normalized_query(query)
        if any(token in normalized for token in _COMPARISON_TOKENS):
            return TaskType.COMPARISON
        if any(token in normalized for token in _TIMELINE_TOKENS):
            return TaskType.TIMELINE
        if any(token in normalized for token in _RESEARCH_TOKENS) or len(source_scope) > 1:
            return TaskType.RESEARCH if len(source_scope) > 1 or "research" in normalized else TaskType.SYNTHESIS
        if len(source_scope) == 1 or any(token in normalized for token in _SCOPED_TOKENS):
            return TaskType.SINGLE_DOC_QA
        return TaskType.LOOKUP

    def _classify_complexity(
        self,
        task_type: TaskType,
        source_scope: Sequence[str],
    ) -> ComplexityLevel:
        if task_type is TaskType.COMPARISON:
            return ComplexityLevel.L3_COMPARATIVE
        if task_type in {TaskType.TIMELINE, TaskType.RESEARCH, TaskType.SYNTHESIS}:
            return ComplexityLevel.L4_RESEARCH
        if len(source_scope) == 1:
            return ComplexityLevel.L2_SCOPED
        return ComplexityLevel.L1_DIRECT

    @staticmethod
    def _choose_runtime(task_type: TaskType, complexity_level: ComplexityLevel) -> RuntimeMode:
        if task_type in _DEEP_TASK_TYPES:
            return RuntimeMode.DEEP
        if complexity_level in {
            ComplexityLevel.L3_COMPARATIVE,
            ComplexityLevel.L4_RESEARCH,
        }:
            return RuntimeMode.DEEP
        return RuntimeMode.FAST

    def route(
        self,
        query: str,
        *,
        source_scope: Sequence[str] = (),
        access_policy: AccessPolicy | None = None,
    ) -> RoutingDecision:
        del access_policy
        task_type = self._classify_task_type(query, source_scope)
        complexity_level = self._classify_complexity(task_type, source_scope)
        runtime_mode = self._choose_runtime(task_type, complexity_level)
        web_search_allowed = task_type in {
            TaskType.COMPARISON,
            TaskType.SYNTHESIS,
            TaskType.TIMELINE,
            TaskType.RESEARCH,
        }
        graph_expansion_allowed = runtime_mode is RuntimeMode.DEEP
        return RoutingDecision(
            task_type=task_type,
            complexity_level=complexity_level,
            runtime_mode=runtime_mode,
            source_scope=list(source_scope),
            web_search_allowed=web_search_allowed,
            graph_expansion_allowed=graph_expansion_allowed,
        )


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
        from pkp.query.query import ContextEvidence

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
