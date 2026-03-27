from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from pkp.config.policies import RoutingThresholds
from pkp.types.access import AccessPolicy, RuntimeMode
from pkp.types.content import ChunkRole
from pkp.types.envelope import EvidenceItem
from pkp.types.query import ComplexityLevel, TaskType


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
