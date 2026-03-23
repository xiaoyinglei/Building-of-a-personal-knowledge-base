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
from pkp.service.routing_service import RoutingDecision, RoutingService
from pkp.service.telemetry_service import TelemetryService
from pkp.types.access import AccessPolicy
from pkp.types.envelope import PreservationSuggestion


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: RoutingDecision
    evidence: EvidenceBundle
    self_check: SelfCheckResult
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    graph_expanded: bool = False
    preservation_suggestion: PreservationSuggestion = Field(
        default_factory=lambda: PreservationSuggestion(suggested=False)
    )


class Reranker(Protocol):
    def __call__(self, query: str, candidates: list[CandidateLike]) -> Sequence[CandidateLike]: ...


@dataclass(frozen=True)
class _FusedCandidate:
    candidate: CandidateLike
    rrf_score: float
    rank: int


class RetrievalService:
    def __init__(
        self,
        *,
        full_text_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        vector_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        section_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        graph_expander: Callable[[str, list[str], list[CandidateLike]], Sequence[CandidateLike]] | None = None,
        web_retriever: Callable[[str, list[str]], Sequence[CandidateLike]] | None = None,
        reranker: Reranker | None = None,
        routing_service: RoutingService | None = None,
        evidence_service: EvidenceService | None = None,
        graph_expansion_service: GraphExpansionService | None = None,
        artifact_service: ArtifactService | None = None,
        telemetry_service: TelemetryService | None = None,
        thresholds: RoutingThresholds | None = None,
    ) -> None:
        self._full_text_retriever = full_text_retriever or (lambda _query, _scope: [])
        self._vector_retriever = vector_retriever or (lambda _query, _scope: [])
        self._section_retriever = section_retriever or (lambda _query, _scope: [])
        self._graph_expander = graph_expander or (lambda _query, _scope, _evidence: [])
        self._web_retriever = web_retriever or (lambda _query, _scope: [])
        self._reranker = reranker
        self._routing_service = routing_service or RoutingService(thresholds)
        self._evidence_service = evidence_service or EvidenceService(thresholds)
        self._graph_expansion_service = graph_expansion_service or GraphExpansionService()
        self._artifact_service = artifact_service or ArtifactService()
        self._telemetry_service = telemetry_service
        self._thresholds = thresholds or RoutingThresholds()

    @staticmethod
    def _branch_key(candidate: CandidateLike) -> str:
        return candidate.chunk_id

    def _rrf_fuse(self, branches: Sequence[Sequence[CandidateLike]]) -> list[CandidateLike]:
        fused: dict[str, _FusedCandidate] = {}
        k = 60
        for branch in branches:
            for index, candidate in enumerate(branch, start=1):
                score = 1.0 / (k + index)
                key = self._branch_key(candidate)
                existing = fused.get(key)
                if existing is None:
                    fused[key] = _FusedCandidate(candidate=candidate, rrf_score=score, rank=index)
                    continue
                fused[key] = _FusedCandidate(
                    candidate=existing.candidate,
                    rrf_score=existing.rrf_score + score,
                    rank=min(existing.rank, index),
                )

        ordered = sorted(
            fused.values(),
            key=lambda fused_candidate: (-fused_candidate.rrf_score, fused_candidate.rank),
        )
        return [item.candidate for item in ordered]

    def retrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
    ) -> RetrievalResult:
        scope = list(source_scope)
        decision = self._routing_service.route(
            query,
            source_scope=scope,
            access_policy=access_policy,
        )

        branch_candidates: list[list[CandidateLike]] = []
        full_text_candidates = self._evidence_service.filter_candidates(
            self._full_text_retriever(query, scope),
            source_scope=scope,
            access_policy=access_policy,
            runtime_mode=decision.runtime_mode,
        )
        self._record_branch_usage("full_text", full_text_candidates, decision.runtime_mode.value)
        vector_candidates = self._evidence_service.filter_candidates(
            self._vector_retriever(query, scope),
            source_scope=scope,
            access_policy=access_policy,
            runtime_mode=decision.runtime_mode,
        )
        self._record_branch_usage("vector", vector_candidates, decision.runtime_mode.value)
        section_candidates = self._evidence_service.filter_candidates(
            self._section_retriever(query, scope),
            source_scope=scope,
            access_policy=access_policy,
            runtime_mode=decision.runtime_mode,
        )
        self._record_branch_usage("section", section_candidates, decision.runtime_mode.value)
        branch_candidates.extend([full_text_candidates, vector_candidates, section_candidates])

        if decision.web_search_allowed and access_policy.external_retrieval.value == "allow":
            web_candidates = self._evidence_service.filter_candidates(
                self._web_retriever(query, scope),
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
            self._record_branch_usage("web", web_candidates, decision.runtime_mode.value)
            branch_candidates.append(web_candidates)

        candidate_count = sum(len(branch) for branch in branch_candidates)
        fused_candidates = self._rrf_fuse(branch_candidates)
        if self._telemetry_service is not None:
            self._telemetry_service.record_rrf_fusion(
                branch_count=len(branch_candidates),
                candidate_count=candidate_count,
                fused_count=len(fused_candidates),
                duplicate_count=max(0, candidate_count - len(fused_candidates)),
            )
        if self._reranker is None:
            raise ValueError("reranker is required for retrieval")
        reranked_candidates = list(self._reranker(query, fused_candidates))
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
