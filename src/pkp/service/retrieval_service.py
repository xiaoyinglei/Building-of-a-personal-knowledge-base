from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from pkp.config.policies import RoutingThresholds
from pkp.service.artifact_service import ArtifactService
from pkp.service.evidence_service import (
    CandidateLike,
    EvidenceBundle,
    EvidenceService,
    SelfCheckResult,
)
from pkp.service.graph_expansion_service import GraphExpansionService
from pkp.service.query_understanding_service import QueryUnderstandingService
from pkp.service.routing_service import RoutingDecision, RoutingService
from pkp.service.telemetry_service import TelemetryService
from pkp.types.access import AccessPolicy, ExecutionLocationPreference
from pkp.types.content import ChunkRole
from pkp.types.diagnostics import RetrievalDiagnostics
from pkp.types.envelope import PreservationSuggestion
from pkp.types.text import (
    looks_command_like,
    looks_definition_query,
    looks_definition_text,
    looks_operation_query,
    looks_structure_query,
    looks_structure_text,
)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: RoutingDecision
    evidence: EvidenceBundle
    self_check: SelfCheckResult
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    graph_expanded: bool = False
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    preservation_suggestion: PreservationSuggestion = Field(
        default_factory=lambda: PreservationSuggestion(suggested=False)
    )


class Reranker(Protocol):
    def __call__(self, query: str, candidates: list[CandidateLike]) -> Sequence[CandidateLike]: ...


@dataclass(frozen=True)
class _FusedCandidate:
    candidate: CandidateLike
    fused_score: float
    rank: int
    supporting_branches: int
    branch_scores: dict[str, float]


@dataclass
class _FusedCandidateView:
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str
    source_id: str | None
    section_path: Sequence[str]
    effective_access_policy: AccessPolicy | None = None
    chunk_role: ChunkRole | None = None
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    parent_text: str | None = None
    metadata: dict[str, str] | None = None
    retrieval_channels: list[str] | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    special_score: float | None = None
    structure_score: float | None = None
    metadata_score: float | None = None
    fusion_score: float | None = None
    rrf_score: float | None = None
    unified_rank: int | None = None


class RetrievalService:
    def __init__(
        self,
        *,
        full_text_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        vector_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        section_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        special_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        metadata_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        graph_expander: Callable[[str, list[str], list[CandidateLike]], Sequence[CandidateLike]] | None = None,
        web_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        reranker: Reranker | None = None,
        routing_service: RoutingService | None = None,
        query_understanding_service: QueryUnderstandingService | None = None,
        evidence_service: EvidenceService | None = None,
        graph_expansion_service: GraphExpansionService | None = None,
        artifact_service: ArtifactService | None = None,
        telemetry_service: TelemetryService | None = None,
        thresholds: RoutingThresholds | None = None,
    ) -> None:
        self._full_text_retriever = full_text_retriever or (lambda _query, _scope: [])
        self._vector_retriever = vector_retriever or (lambda _query, _scope: [])
        self._section_retriever = section_retriever or (lambda _query, _scope: [])
        self._special_retriever = special_retriever or (lambda _query, _scope: [])
        self._metadata_retriever = metadata_retriever or (lambda _query, _scope: [])
        self._graph_expander = graph_expander or (lambda _query, _scope, _evidence: [])
        self._web_retriever = web_retriever or (lambda _query, _scope: [])
        self._reranker = reranker
        self._routing_service = routing_service or RoutingService(thresholds)
        self._query_understanding_service = query_understanding_service or QueryUnderstandingService()
        self._evidence_service = evidence_service or EvidenceService(thresholds)
        self._graph_expansion_service = graph_expansion_service or GraphExpansionService()
        self._artifact_service = artifact_service or ArtifactService()
        self._telemetry_service = telemetry_service
        self._thresholds = thresholds or RoutingThresholds()

    @staticmethod
    def _branch_key(candidate: CandidateLike) -> str:
        return candidate.chunk_id

    @staticmethod
    def _normalized_branch_scores(branch: Sequence[CandidateLike]) -> dict[str, float]:
        positive_scores = [max(float(candidate.score), 0.0) for candidate in branch]
        max_score = max(positive_scores, default=0.0)
        if max_score > 0.0:
            return {
                candidate.chunk_id: max(float(candidate.score), 0.0) / max_score
                for candidate in branch
            }

        size = len(branch)
        if size <= 1:
            return {candidate.chunk_id: 1.0 for candidate in branch}
        return {
            candidate.chunk_id: 1.0 - ((index - 1) / size)
            for index, candidate in enumerate(branch, start=1)
        }

    @staticmethod
    def _branch_weight(branch_name: str, query: str) -> float:
        if looks_structure_query(query):
            return {
                "full_text": 1.0,
                "vector": 0.95,
                "section": 1.3,
                "metadata": 1.2,
                "web": 0.6,
            }.get(branch_name, 1.0)
        if looks_definition_query(query):
            return {
                "full_text": 0.9,
                "vector": 1.35,
                "section": 0.8,
                "metadata": 0.9,
                "web": 0.6,
            }.get(branch_name, 1.0)
        if looks_operation_query(query):
            return {
                "full_text": 1.1,
                "vector": 1.0,
                "section": 0.9,
                "metadata": 0.9,
                "web": 0.6,
            }.get(branch_name, 1.0)
        return {
            "full_text": 1.0,
            "vector": 1.1,
            "section": 0.9,
            "metadata": 1.0,
            "web": 0.6,
        }.get(branch_name, 1.0)

    @staticmethod
    def _candidate_quality_prior(query: str, candidate: CandidateLike) -> float:
        text = candidate.text
        section_text = " ".join(candidate.section_path)
        query_is_command_like = looks_command_like(query)
        query_is_definition_like = looks_definition_query(query)
        query_is_structure_like = looks_structure_query(query)

        prior = 0.0
        if not query_is_command_like and looks_command_like(text):
            prior -= 0.35
            if query_is_definition_like and not query_is_structure_like:
                prior -= 0.2
        if (
            query_is_definition_like
            and not query_is_structure_like
            and not looks_command_like(text)
        ):
            if looks_definition_text(text):
                prior += 0.22
            if looks_definition_text(section_text):
                prior += 0.08
        if query_is_structure_like:
            if looks_structure_text(text):
                prior += 0.12
            if looks_structure_text(section_text):
                prior += 0.12
        return prior

    def _rrf_fuse(
        self,
        query: str,
        branches: Sequence[tuple[str, Sequence[CandidateLike]]],
    ) -> list[CandidateLike]:
        fused: dict[str, _FusedCandidate] = {}
        k = 60
        for branch_name, branch in branches:
            branch_weight = self._branch_weight(branch_name, query)
            normalized_scores = self._normalized_branch_scores(branch)
            for index, candidate in enumerate(branch, start=1):
                normalized_score = normalized_scores.get(candidate.chunk_id, 0.0)
                score = (
                    branch_weight / (k + index)
                    + (branch_weight * normalized_score * 0.3)
                    + self._candidate_quality_prior(query, candidate)
                )
                key = self._branch_key(candidate)
                existing = fused.get(key)
                if existing is None:
                    fused[key] = _FusedCandidate(
                        candidate=candidate,
                        fused_score=score,
                        rank=index,
                        supporting_branches=1,
                        branch_scores={branch_name: max(float(candidate.score), 0.0)},
                    )
                    continue
                branch_scores = dict(existing.branch_scores)
                branch_scores[branch_name] = max(float(candidate.score), 0.0)
                fused[key] = _FusedCandidate(
                    candidate=existing.candidate,
                    fused_score=existing.fused_score + score,
                    rank=min(existing.rank, index),
                    supporting_branches=existing.supporting_branches + 1,
                    branch_scores=branch_scores,
                )

        ordered = sorted(
            fused.values(),
            key=lambda fused_candidate: (
                -(fused_candidate.fused_score + max(0, fused_candidate.supporting_branches - 1) * 0.05),
                -fused_candidate.supporting_branches,
                fused_candidate.rank,
            ),
        )
        return [
            _FusedCandidateView(
                chunk_id=item.candidate.chunk_id,
                doc_id=item.candidate.doc_id,
                text=item.candidate.text,
                citation_anchor=item.candidate.citation_anchor,
                score=item.fused_score + max(0, item.supporting_branches - 1) * 0.05,
                rank=item.rank,
                source_kind=item.candidate.source_kind,
                source_id=item.candidate.source_id,
                section_path=tuple(item.candidate.section_path),
                effective_access_policy=getattr(item.candidate, "effective_access_policy", None),
                chunk_role=getattr(item.candidate, "chunk_role", None),
                special_chunk_type=getattr(item.candidate, "special_chunk_type", None),
                parent_chunk_id=getattr(item.candidate, "parent_chunk_id", None),
                parent_text=getattr(item.candidate, "parent_text", None),
                metadata=getattr(item.candidate, "metadata", None),
                retrieval_channels=sorted(item.branch_scores),
                dense_score=item.branch_scores.get("vector"),
                sparse_score=item.branch_scores.get("full_text"),
                special_score=item.branch_scores.get("special"),
                structure_score=item.branch_scores.get("section"),
                metadata_score=item.branch_scores.get("metadata"),
                fusion_score=item.fused_score + max(0, item.supporting_branches - 1) * 0.05,
                rrf_score=item.fused_score + max(0, item.supporting_branches - 1) * 0.05,
                unified_rank=index,
            )
            for index, item in enumerate(ordered, start=1)
        ]

    @staticmethod
    def _apply_parent_backfill(candidates: Sequence[CandidateLike]) -> tuple[list[CandidateLike], int]:
        enriched: list[CandidateLike] = []
        backfilled = 0
        for candidate in candidates:
            parent_text = getattr(candidate, "parent_text", None)
            parent_chunk_id = getattr(candidate, "parent_chunk_id", None)
            if parent_chunk_id and parent_text:
                enriched.append(
                    _FusedCandidateView(
                        chunk_id=candidate.chunk_id,
                        doc_id=candidate.doc_id,
                        text=parent_text,
                        citation_anchor=candidate.citation_anchor,
                        score=float(candidate.score),
                        rank=int(candidate.rank),
                        source_kind=candidate.source_kind,
                        source_id=candidate.source_id,
                        section_path=tuple(candidate.section_path),
                        effective_access_policy=getattr(candidate, "effective_access_policy", None),
                        chunk_role=getattr(candidate, "chunk_role", None),
                        special_chunk_type=getattr(candidate, "special_chunk_type", None),
                        parent_chunk_id=parent_chunk_id,
                        parent_text=parent_text,
                        metadata=getattr(candidate, "metadata", None),
                        retrieval_channels=list(getattr(candidate, "retrieval_channels", []) or []),
                        dense_score=getattr(candidate, "dense_score", None),
                        sparse_score=getattr(candidate, "sparse_score", None),
                        special_score=getattr(candidate, "special_score", None),
                        structure_score=getattr(candidate, "structure_score", None),
                        metadata_score=getattr(candidate, "metadata_score", None),
                        fusion_score=getattr(candidate, "fusion_score", None),
                        rrf_score=getattr(candidate, "rrf_score", None),
                        unified_rank=getattr(candidate, "unified_rank", None),
                    )
                )
                backfilled += 1
                continue
            enriched.append(candidate)
        return enriched, backfilled

    @staticmethod
    def _collapse_redundant_candidates(candidates: Sequence[CandidateLike]) -> tuple[list[CandidateLike], int]:
        collapsed = 0
        ordered: list[CandidateLike] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            if candidate.source_kind != "internal":
                ordered.append(candidate)
                continue
            if getattr(candidate, "special_chunk_type", None):
                key = ("chunk", candidate.doc_id, candidate.chunk_id)
            elif parent_chunk_id := getattr(candidate, "parent_chunk_id", None):
                key = ("parent", candidate.doc_id, parent_chunk_id)
            else:
                key = ("chunk", candidate.doc_id, candidate.chunk_id)
            if key in seen_keys:
                collapsed += 1
                continue
            seen_keys.add(key)
            ordered.append(candidate)
        return ordered, collapsed

    def retrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
    ) -> RetrievalResult:
        scope = list(source_scope)
        prepare_for_policy = getattr(self._vector_retriever, "prepare_for_policy", None)
        if callable(prepare_for_policy):
            prepare_for_policy(
                access_policy=access_policy,
                execution_location_preference=execution_location_preference,
            )
        decision = self._routing_service.route(
            query,
            source_scope=scope,
            access_policy=access_policy,
        )
        query_understanding = self._query_understanding_service.analyze(query)

        internal_branches: list[tuple[str, list[CandidateLike]]] = []
        full_text_candidates: list[CandidateLike] = []
        if query_understanding.needs_sparse:
            full_text_candidates = self._evidence_service.filter_candidates(
                self._full_text_retriever(query, scope),
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
        self._record_branch_usage("full_text", full_text_candidates, decision.runtime_mode.value)
        vector_candidates: list[CandidateLike] = []
        if query_understanding.needs_dense:
            vector_candidates = self._evidence_service.filter_candidates(
                self._vector_retriever(query, scope),
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
        self._record_branch_usage("vector", vector_candidates, decision.runtime_mode.value)
        section_candidates: list[CandidateLike] = []
        if query_understanding.needs_structure:
            section_candidates = self._evidence_service.filter_candidates(
                self._section_retriever(query, scope),
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
        self._record_branch_usage("section", section_candidates, decision.runtime_mode.value)
        special_candidates: list[CandidateLike] = []
        if query_understanding.needs_special:
            special_candidates = self._evidence_service.filter_candidates(
                self._special_retriever(query, scope),
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
            self._record_branch_usage("special", special_candidates, decision.runtime_mode.value)
        metadata_candidates: list[CandidateLike] = []
        if query_understanding.needs_metadata:
            metadata_candidates = self._evidence_service.filter_candidates(
                self._metadata_retriever(query, scope),
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
            self._record_branch_usage("metadata", metadata_candidates, decision.runtime_mode.value)
        if full_text_candidates:
            internal_branches.append(("full_text", full_text_candidates))
        if vector_candidates:
            internal_branches.append(("vector", vector_candidates))
        if section_candidates:
            internal_branches.append(("section", section_candidates))
        if special_candidates:
            internal_branches.append(("special", special_candidates))
        if metadata_candidates:
            internal_branches.append(("metadata", metadata_candidates))

        fused_candidates = self._rrf_fuse(query, internal_branches)
        candidate_count = sum(len(branch) for _, branch in internal_branches)
        if self._telemetry_service is not None:
            self._telemetry_service.record_rrf_fusion(
                branch_count=len(internal_branches),
                candidate_count=candidate_count,
                fused_count=len(fused_candidates),
                duplicate_count=max(0, candidate_count - len(fused_candidates)),
            )
        if self._reranker is None:
            raise ValueError("reranker is required for retrieval")
        reranked_candidates = list(self._reranker(query, fused_candidates))
        reranked_candidates, parent_backfilled_count = self._apply_parent_backfill(reranked_candidates)
        reranked_candidates, collapsed_candidate_count = self._collapse_redundant_candidates(reranked_candidates)
        if self._telemetry_service is not None:
            fused_ids = [candidate.chunk_id for candidate in fused_candidates]
            reranked_ids = [candidate.chunk_id for candidate in reranked_candidates]
            self._telemetry_service.record_rerank_effectiveness(
                input_count=len(fused_candidates),
                output_count=len(reranked_candidates),
                reordered=fused_ids != reranked_ids,
                top1_changed=(fused_ids[:1] != reranked_ids[:1]),
            )

        internal_evidence = self._evidence_service.assemble_bundle(reranked_candidates)
        internal_self_check = self._evidence_service.evaluate_self_check(
            bundle=internal_evidence,
            task_type=decision.task_type,
            complexity_level=decision.complexity_level,
        )
        web_candidates: list[CandidateLike] = []
        if (
            decision.web_search_allowed
            and access_policy.external_retrieval.value == "allow"
            and internal_self_check.retrieve_more
        ):
            retrieved_web_candidates = self._evidence_service.filter_candidates(
                self._web_retriever(query, scope),
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
            self._record_branch_usage("web", retrieved_web_candidates, decision.runtime_mode.value)
            if retrieved_web_candidates:
                web_candidates = retrieved_web_candidates
                combined_branches = [*internal_branches, ("web", web_candidates)]
                combined_candidate_count = sum(len(branch) for _, branch in combined_branches)
                fused_candidates = self._rrf_fuse(query, combined_branches)
                if self._telemetry_service is not None:
                    self._telemetry_service.record_rrf_fusion(
                        branch_count=len(combined_branches),
                        candidate_count=combined_candidate_count,
                        fused_count=len(fused_candidates),
                        duplicate_count=max(0, combined_candidate_count - len(fused_candidates)),
                    )
                reranked_candidates = list(self._reranker(query, fused_candidates))
                reranked_candidates, parent_backfilled_count = self._apply_parent_backfill(reranked_candidates)
                reranked_candidates, collapsed_candidate_count = self._collapse_redundant_candidates(
                    reranked_candidates
                )
                if self._telemetry_service is not None:
                    fused_ids = [candidate.chunk_id for candidate in fused_candidates]
                    reranked_ids = [candidate.chunk_id for candidate in reranked_candidates]
                    self._telemetry_service.record_rerank_effectiveness(
                        input_count=len(fused_candidates),
                        output_count=len(reranked_candidates),
                        reordered=fused_ids != reranked_ids,
                        top1_changed=(fused_ids[:1] != reranked_ids[:1]),
                    )

        evidence = self._evidence_service.assemble_bundle(reranked_candidates)
        graph_expanded = False
        if decision.graph_expansion_allowed:
            internal_candidates = [
                candidate for candidate in reranked_candidates if candidate.source_kind == "internal"
            ]
            graph_candidates = self._graph_expansion_service.expand(
                query=query,
                source_scope=scope,
                evidence=evidence,
                graph_candidates=self._graph_expander(query, scope, internal_candidates),
                access_policy=access_policy,
            )
            if graph_candidates:
                if self._telemetry_service is not None:
                    self._telemetry_service.record_graph_expansion(
                        seed_count=len(internal_candidates),
                        added_count=len(graph_candidates),
                    )
                graph_expanded = True
                graph_items = self._evidence_service.assemble_bundle(graph_candidates).graph
                evidence = EvidenceBundle(
                    internal=evidence.internal,
                    external=evidence.external,
                    graph=[*evidence.graph, *graph_items],
                )

        self_check = self._evidence_service.evaluate_self_check(
            bundle=evidence,
            task_type=decision.task_type,
            complexity_level=decision.complexity_level,
        )
        preservation_suggestion = self._artifact_service.suggest_preservation(
            query=query,
            runtime_mode=decision.runtime_mode,
            evidence=evidence.all,
            differences_or_conflicts=[],
        )
        if self._telemetry_service is not None and preservation_suggestion.suggested:
            self._telemetry_service.record_preservation_suggestion(
                artifact_type=preservation_suggestion.artifact_type or "unknown",
                runtime_mode=decision.runtime_mode.value,
                evidence_count=len(evidence.all),
                conflict_count=0,
            )

        return RetrievalResult(
            decision=decision,
            evidence=evidence,
            self_check=self_check,
            reranked_chunk_ids=[candidate.chunk_id for candidate in reranked_candidates],
            graph_expanded=graph_expanded,
            diagnostics=RetrievalDiagnostics(
                branch_hits={
                    "full_text": len(full_text_candidates),
                    "vector": len(vector_candidates),
                    "section": len(section_candidates),
                    **({"special": len(special_candidates)} if special_candidates else {}),
                    **({"metadata": len(metadata_candidates)} if metadata_candidates else {}),
                    **({"web": len(web_candidates)} if web_candidates else {}),
                },
                reranked_chunk_ids=[candidate.chunk_id for candidate in reranked_candidates],
                embedding_provider=getattr(self._vector_retriever, "last_provider", None),
                rerank_provider=getattr(self._reranker, "last_provider", None),
                attempts=[
                    *list(getattr(self._vector_retriever, "last_attempts", [])),
                    *list(getattr(self._reranker, "last_attempts", [])),
                ],
                fusion_input_count=candidate_count + len(web_candidates),
                fused_count=len(reranked_candidates),
                graph_expanded=graph_expanded,
                query_understanding=query_understanding,
                parent_backfilled_count=parent_backfilled_count,
                collapsed_candidate_count=collapsed_candidate_count,
            ),
            preservation_suggestion=preservation_suggestion,
        )

    def _record_branch_usage(
        self,
        branch: str,
        candidates: Sequence[CandidateLike],
        runtime_mode: str,
    ) -> None:
        if self._telemetry_service is None:
            return
        self._telemetry_service.record_branch_usage(
            branch=branch,
            hit_count=len(candidates),
            runtime_mode=runtime_mode,
        )
