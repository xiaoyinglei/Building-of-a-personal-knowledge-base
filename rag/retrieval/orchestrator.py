from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.retrieval.analysis import (
    QueryUnderstandingService,
    RoutingDecision,
    RoutingService,
    narrow_access_policy_for_query,
)
from rag.retrieval.evidence import (
    ArtifactService,
    CandidateLike,
    EvidenceBundle,
    EvidenceService,
    EvidenceThresholds,
    SelfCheckResult,
)
from rag.retrieval.graph import GraphExpansionService
from rag.retrieval.models import QueryMode, QueryOptions, RetrievalResult, normalize_query_mode
from rag.schema.core import ChunkRole
from rag.schema.query import QueryUnderstanding
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference, ProviderAttempt, RetrievalDiagnostics
from rag.utils.telemetry import TelemetryService


class RetrievalExecutor(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_mode: QueryMode | str | None = None,
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult: ...


class RetrieverFn(Protocol):
    def __call__(
        self,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
    ) -> Sequence[CandidateLike]: ...


class GraphExpander(Protocol):
    def __call__(
        self,
        query: str,
        source_scope: list[str],
        evidence: list[CandidateLike],
    ) -> Sequence[CandidateLike]: ...


class Reranker(Protocol):
    def __call__(self, query: str, candidates: list[CandidateLike]) -> Sequence[CandidateLike]: ...


@dataclass(slots=True)
class BranchRetrieverRegistry:
    full_text_retriever: RetrieverFn
    vector_retriever: RetrieverFn
    section_retriever: RetrieverFn
    special_retriever: RetrieverFn
    metadata_retriever: RetrieverFn
    local_retriever: RetrieverFn
    global_retriever: RetrieverFn
    web_retriever: RetrieverFn

    def collect_web(
        self,
        *,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
    ) -> list[CandidateLike]:
        return list(self.web_retriever(query, source_scope, query_understanding))

    def get(self, branch: str) -> RetrieverFn:
        return {
            "full_text": self.full_text_retriever,
            "vector": self.vector_retriever,
            "section": self.section_retriever,
            "special": self.special_retriever,
            "metadata": self.metadata_retriever,
            "local": self.local_retriever,
            "global": self.global_retriever,
        }.get(branch, self.vector_retriever)


@dataclass(slots=True)
class UnifiedReranker:
    reranker: Reranker | None = None

    def rerank(self, query: str, candidates: list[CandidateLike]) -> list[CandidateLike]:
        if self.reranker is None:
            return list(candidates)
        return list(self.reranker(query, candidates))


@dataclass(frozen=True)
class FusedCandidate:
    candidate: CandidateLike
    fused_score: float
    rank: int
    supporting_branches: int
    branch_scores: dict[str, float]


@dataclass
class FusedCandidateView(CandidateLike):
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str
    source_id: str | None
    section_path: Sequence[str]
    benchmark_doc_id: str | None = None
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

    @property
    def item_id(self) -> str:
        return self.chunk_id


@dataclass(slots=True)
class ReciprocalRankFusion:
    rank_constant: int = 60

    def fuse(
        self,
        *,
        query: str,
        mode: QueryMode,
        branches: Sequence[tuple[str, Sequence[CandidateLike]]],
    ) -> list[CandidateLike]:
        del query, mode
        fused: dict[str, FusedCandidate] = {}
        for branch_name, branch in branches:
            for index, candidate in enumerate(branch, start=1):
                score = 1.0 / (self.rank_constant + index)
                existing = fused.get(candidate.chunk_id)
                branch_scores = {branch_name: max(float(candidate.score), 0.0)}
                if existing is None:
                    fused[candidate.chunk_id] = FusedCandidate(
                        candidate=candidate,
                        fused_score=score,
                        rank=index,
                        supporting_branches=1,
                        branch_scores=branch_scores,
                    )
                    continue
                merged_scores = dict(existing.branch_scores)
                merged_scores.update(branch_scores)
                fused[candidate.chunk_id] = FusedCandidate(
                    candidate=existing.candidate,
                    fused_score=existing.fused_score + score,
                    rank=min(existing.rank, index),
                    supporting_branches=existing.supporting_branches + 1,
                    branch_scores=merged_scores,
                )

        ordered = sorted(
            fused.values(),
            key=lambda item: (-item.fused_score, -item.supporting_branches, item.rank, item.candidate.chunk_id),
        )
        return [self._to_view(item, index) for index, item in enumerate(ordered, start=1)]

    @staticmethod
    def _to_view(item: FusedCandidate, unified_rank: int) -> FusedCandidateView:
        final_score = item.fused_score
        return FusedCandidateView(
            chunk_id=item.candidate.chunk_id,
            doc_id=item.candidate.doc_id,
            benchmark_doc_id=getattr(item.candidate, "benchmark_doc_id", None),
            text=item.candidate.text,
            citation_anchor=item.candidate.citation_anchor,
            score=final_score,
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
            fusion_score=final_score,
            rrf_score=final_score,
            unified_rank=unified_rank,
        )


@dataclass(frozen=True, slots=True)
class BranchExecutionSpec:
    branch: str
    limit: int


@dataclass(frozen=True, slots=True)
class ModeExecutionSpec:
    mode: QueryMode
    executor_name: str
    internal_branches: tuple[BranchExecutionSpec, ...]
    allow_web: bool = True
    allow_graph_expansion: bool = True
    web_limit: int = 0
    graph_limit: int = 0

class RetrievalService:
    def __init__(
        self,
        *,
        full_text_retriever: RetrieverFn | None = None,
        vector_retriever: RetrieverFn | None = None,
        local_retriever: RetrieverFn | None = None,
        global_retriever: RetrieverFn | None = None,
        section_retriever: RetrieverFn | None = None,
        special_retriever: RetrieverFn | None = None,
        metadata_retriever: RetrieverFn | None = None,
        graph_expander: GraphExpander | None = None,
        web_retriever: RetrieverFn | None = None,
        reranker: Reranker | None = None,
        routing_service: RoutingService | None = None,
        query_understanding_service: QueryUnderstandingService | None = None,
        evidence_service: EvidenceService | None = None,
        graph_expansion_service: GraphExpansionService | None = None,
        artifact_service: ArtifactService | None = None,
        telemetry_service: TelemetryService | None = None,
        evidence_thresholds: EvidenceThresholds | None = None,
    ) -> None:
        self._full_text_retriever: RetrieverFn = full_text_retriever or (lambda _query, _scope, _understanding: [])
        self._vector_retriever: RetrieverFn = vector_retriever or (lambda _query, _scope, _understanding: [])
        self._local_retriever: RetrieverFn = local_retriever or (lambda _query, _scope, _understanding: [])
        self._global_retriever: RetrieverFn = global_retriever or (lambda _query, _scope, _understanding: [])
        self._section_retriever: RetrieverFn = section_retriever or (lambda _query, _scope, _understanding: [])
        self._special_retriever: RetrieverFn = special_retriever or (lambda _query, _scope, _understanding: [])
        self._metadata_retriever: RetrieverFn = metadata_retriever or (lambda _query, _scope, _understanding: [])
        self._graph_expander: GraphExpander = graph_expander or (lambda _query, _scope, _evidence: [])
        self._web_retriever: RetrieverFn = web_retriever or (lambda _query, _scope, _understanding: [])
        self._reranker = reranker
        self._routing_service = routing_service or RoutingService()
        self._query_understanding_service = query_understanding_service or QueryUnderstandingService()
        self._evidence_service = evidence_service or EvidenceService(evidence_thresholds)
        self._graph_expansion_service = graph_expansion_service or GraphExpansionService()
        self._artifact_service = artifact_service or ArtifactService()
        self._telemetry_service = telemetry_service
        self._fusion = ReciprocalRankFusion()
        self._unified_reranker = UnifiedReranker(reranker=self._reranker)
        self.last_result: RetrievalResult | None = None
        self._branch_registry = BranchRetrieverRegistry(
            full_text_retriever=self._full_text_retriever,
            vector_retriever=self._vector_retriever,
            local_retriever=self._local_retriever,
            global_retriever=self._global_retriever,
            section_retriever=self._section_retriever,
            special_retriever=self._special_retriever,
            metadata_retriever=self._metadata_retriever,
            web_retriever=self._web_retriever,
        )
        self.branch_registry = self._branch_registry
        self.routing_service = self._routing_service
        self.query_understanding_service = self._query_understanding_service
        self.evidence_service = self._evidence_service
        self.graph_expansion_service = self._graph_expansion_service
        self.artifact_service = self._artifact_service
        self.fusion = self._fusion
        self.reranker = self._unified_reranker
        self.graph_expander = self._graph_expander
        self.telemetry_service = self._telemetry_service

    @staticmethod
    def _benchmark_doc_ids(candidates: Sequence[CandidateLike]) -> list[str]:
        ranked_doc_ids: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            benchmark_doc_id = getattr(candidate, "benchmark_doc_id", None)
            if not isinstance(benchmark_doc_id, str):
                continue
            normalized = benchmark_doc_id.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ranked_doc_ids.append(normalized)
        return ranked_doc_ids

    def run(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_mode: QueryMode | str | None = None,
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        scope = list(source_scope)
        resolved_mode = normalize_query_mode(query_options.mode if query_options is not None else query_mode)
        query_understanding = self.query_understanding_service.analyze(
            query,
            access_policy=access_policy,
            execution_location_preference=(
                execution_location_preference or ExecutionLocationPreference.LOCAL_FIRST
            ),
        )
        effective_access_policy = narrow_access_policy_for_query(access_policy, query_understanding)
        decision = self.routing_service.route(
            query,
            query_understanding=query_understanding,
            source_scope=scope,
            access_policy=effective_access_policy,
        )
        if resolved_mode is QueryMode.BYPASS:
            return self._run_bypass_mode(
                query=query,
                decision=decision,
                query_understanding=query_understanding,
            )
        return self._execute_mode(
            query=query,
            source_scope=scope,
            access_policy=effective_access_policy,
            decision=decision,
            query_understanding=query_understanding,
            query_options=query_options,
            spec=self._mode_spec(
                resolved_mode=resolved_mode,
                query_understanding=query_understanding,
                query_options=query_options,
            ),
        )

    def _mode_spec(
        self,
        resolved_mode: QueryMode,
        query_understanding: QueryUnderstanding,
        query_options: QueryOptions | None,
    ) -> ModeExecutionSpec:
        retrieval_limit = (
            max(query_options.retrieval_pool_k or query_options.chunk_top_k or query_options.top_k, 1)
            if query_options is not None
            else 8
        )
        final_limit = max(query_options.chunk_top_k or query_options.top_k, 1) if query_options is not None else 8
        aux_branches = tuple(
            BranchExecutionSpec(branch, max(1, retrieval_limit))
            for branch, enabled in (
                ("section", query_understanding.needs_structure),
                ("special", query_understanding.needs_special),
                ("metadata", query_understanding.needs_metadata),
            )
            if enabled
        )
        web_limit = max(1, retrieval_limit // 2)
        graph_limit = max(2, final_limit)
        if resolved_mode is QueryMode.NAIVE:
            return ModeExecutionSpec(
                mode=QueryMode.NAIVE,
                executor_name="naive",
                internal_branches=(BranchExecutionSpec("vector", retrieval_limit * 2),),
                allow_web=False,
                allow_graph_expansion=False,
            )
        if resolved_mode is QueryMode.LOCAL:
            return ModeExecutionSpec(
                mode=QueryMode.LOCAL,
                executor_name="local",
                internal_branches=(BranchExecutionSpec("local", retrieval_limit * 2), *aux_branches),
                web_limit=web_limit,
                graph_limit=graph_limit,
            )
        if resolved_mode is QueryMode.GLOBAL:
            return ModeExecutionSpec(
                mode=QueryMode.GLOBAL,
                executor_name="global",
                internal_branches=(BranchExecutionSpec("global", retrieval_limit * 2), *aux_branches),
                web_limit=web_limit,
                graph_limit=graph_limit,
            )
        if resolved_mode is QueryMode.HYBRID:
            return ModeExecutionSpec(
                mode=QueryMode.HYBRID,
                executor_name="hybrid",
                internal_branches=(
                    BranchExecutionSpec("local", retrieval_limit),
                    BranchExecutionSpec("global", retrieval_limit),
                    *aux_branches,
                ),
                web_limit=web_limit,
                graph_limit=graph_limit,
            )
        kg_limit = max(2, retrieval_limit - 1)
        return ModeExecutionSpec(
            mode=QueryMode.MIX,
            executor_name="mix",
            internal_branches=(
                BranchExecutionSpec("local", kg_limit),
                BranchExecutionSpec("global", kg_limit),
                BranchExecutionSpec("vector", retrieval_limit),
                BranchExecutionSpec("full_text", retrieval_limit),
                *aux_branches,
            ),
            web_limit=web_limit,
            graph_limit=graph_limit,
        )

    def _execute_mode(
        self,
        *,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
        query_options: QueryOptions | None,
        spec: ModeExecutionSpec,
    ) -> RetrievalResult:
        internal_branches, branch_hits, branch_limits = self._collect_internal_branches(
            spec=spec,
            query=query,
            source_scope=source_scope,
            access_policy=access_policy,
            decision=decision,
            query_understanding=query_understanding,
        )
        reranked_candidates, candidate_count, parent_backfilled_count, collapsed_candidate_count = self._rank_branches(
            query=query,
            mode=spec.mode,
            branches=internal_branches,
            query_options=query_options,
            rerank_required=decision.rerank_required,
        )
        evidence = self.evidence_service.assemble_bundle(reranked_candidates)
        self_check = self.evidence_service.evaluate_self_check(
            bundle=evidence,
            task_type=decision.task_type,
            complexity_level=decision.complexity_level,
        )

        web_candidates: list[CandidateLike] = []
        if (
            spec.allow_web
            and spec.web_limit > 0
            and decision.web_search_allowed
            and access_policy.external_retrieval.value == "allow"
            and self_check.retrieve_more
        ):
            web_candidates = self._collect_web_candidates(
                query=query,
                source_scope=source_scope,
                access_policy=access_policy,
                decision=decision,
                limit=spec.web_limit,
                query_understanding=query_understanding,
            )
            branch_hits["web"] = len(web_candidates)
            branch_limits["web"] = spec.web_limit
            if web_candidates:
                reranked_candidates, candidate_count, parent_backfilled_count, collapsed_candidate_count = (
                    self._rank_branches(
                        query=query,
                        mode=spec.mode,
                        branches=[*internal_branches, ("web", web_candidates)],
                        query_options=query_options,
                        rerank_required=decision.rerank_required,
                    )
                )
                evidence = self.evidence_service.assemble_bundle(reranked_candidates)

        graph_expanded = False
        if spec.allow_graph_expansion and decision.graph_expansion_allowed:
            internal_candidates = [
                candidate for candidate in reranked_candidates if candidate.source_kind == "internal"
            ]
            graph_candidates = self.graph_expansion_service.expand(
                query=query,
                source_scope=source_scope,
                evidence=evidence,
                graph_candidates=self.graph_expander(query, source_scope, internal_candidates),
                access_policy=access_policy,
            )
            if spec.graph_limit > 0:
                graph_candidates = graph_candidates[: spec.graph_limit]
            if graph_candidates:
                graph_expanded = True
                if self.telemetry_service is not None:
                    self.telemetry_service.record_graph_expansion(
                        seed_count=len(internal_candidates),
                        added_count=len(graph_candidates),
                    )
                graph_items = self.evidence_service.assemble_bundle(graph_candidates).graph
                evidence = EvidenceBundle(
                    internal=evidence.internal,
                    external=evidence.external,
                    graph=[*evidence.graph, *graph_items],
                )

        self_check = self.evidence_service.evaluate_self_check(
            bundle=evidence,
            task_type=decision.task_type,
            complexity_level=decision.complexity_level,
        )
        preservation_suggestion = self.artifact_service.suggest_preservation(
            query=query,
            runtime_mode=decision.runtime_mode,
            evidence=evidence.all,
            differences_or_conflicts=[],
        )
        reranked_benchmark_doc_ids = self._benchmark_doc_ids(reranked_candidates)
        if self.telemetry_service is not None and preservation_suggestion.suggested:
            self.telemetry_service.record_preservation_suggestion(
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
            reranked_benchmark_doc_ids=reranked_benchmark_doc_ids,
            graph_expanded=graph_expanded,
            diagnostics=RetrievalDiagnostics(
                mode_executor=spec.executor_name,
                branch_hits=branch_hits,
                branch_limits=branch_limits,
                reranked_chunk_ids=[candidate.chunk_id for candidate in reranked_candidates],
                reranked_benchmark_doc_ids=reranked_benchmark_doc_ids,
                embedding_provider=self._embedding_provider(),
                rerank_provider=getattr(self.reranker.reranker, "last_provider", None),
                attempts=self._provider_attempts(),
                fusion_input_count=candidate_count + len(web_candidates),
                fused_count=len(reranked_candidates),
                graph_expanded=graph_expanded,
                query_understanding=query_understanding,
                query_understanding_debug=self.query_understanding_service.diagnostics_payload(),
                parent_backfilled_count=parent_backfilled_count,
                collapsed_candidate_count=collapsed_candidate_count,
            ),
            preservation_suggestion=preservation_suggestion,
        )

    @staticmethod
    def _run_bypass_mode(
        *,
        query: str,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
    ) -> RetrievalResult:
        del query
        return RetrievalResult(
            decision=decision.model_copy(
                update={
                    "runtime_mode": decision.runtime_mode,
                    "web_search_allowed": False,
                    "graph_expansion_allowed": False,
                }
            ),
            evidence=EvidenceBundle(),
            self_check=SelfCheckResult(
                retrieve_more=False,
                evidence_sufficient=False,
                claim_supported=False,
            ),
            reranked_chunk_ids=[],
            reranked_benchmark_doc_ids=[],
            graph_expanded=False,
            diagnostics=RetrievalDiagnostics(
                mode_executor="bypass",
                branch_hits={},
                branch_limits={},
                reranked_chunk_ids=[],
                reranked_benchmark_doc_ids=[],
                fusion_input_count=0,
                fused_count=0,
                graph_expanded=False,
                query_understanding=query_understanding,
                query_understanding_debug={},
                parent_backfilled_count=0,
                collapsed_candidate_count=0,
            ),
        )

    def _collect_internal_branches(
        self,
        *,
        spec: ModeExecutionSpec,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
    ) -> tuple[list[tuple[str, list[CandidateLike]]], dict[str, int], dict[str, int]]:
        internal_branches: list[tuple[str, list[CandidateLike]]] = []
        branch_hits: dict[str, int] = {}
        branch_limits = {branch_spec.branch: branch_spec.limit for branch_spec in spec.internal_branches}
        for branch_spec in spec.internal_branches:
            candidates = list(
                self._call_branch(
                    branch_spec.branch,
                    query=query,
                    source_scope=source_scope,
                    query_understanding=query_understanding,
                )
            )
            filtered = self.evidence_service.filter_candidates(
                candidates,
                source_scope=source_scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
                query_understanding=query_understanding,
            )
            limited = filtered[: branch_spec.limit]
            branch_hits[branch_spec.branch] = len(limited)
            if self.telemetry_service is not None:
                self.telemetry_service.record_branch_usage(
                    branch=branch_spec.branch,
                    hit_count=len(limited),
                    runtime_mode=decision.runtime_mode.value,
                )
            if limited:
                internal_branches.append((branch_spec.branch, limited))
        return internal_branches, branch_hits, branch_limits

    def _collect_web_candidates(
        self,
        *,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        limit: int,
        query_understanding: QueryUnderstanding,
    ) -> list[CandidateLike]:
        filtered = self.evidence_service.filter_candidates(
            self.branch_registry.collect_web(
                query=query,
                source_scope=source_scope,
                query_understanding=query_understanding,
            ),
            source_scope=source_scope,
            access_policy=access_policy,
            runtime_mode=decision.runtime_mode,
            query_understanding=query_understanding,
        )
        limited = filtered[:limit]
        if self.telemetry_service is not None:
            self.telemetry_service.record_branch_usage(
                branch="web",
                hit_count=len(limited),
                runtime_mode=decision.runtime_mode.value,
            )
        return limited

    def _rank_branches(
        self,
        *,
        query: str,
        mode: QueryMode,
        branches: list[tuple[str, list[CandidateLike]]],
        query_options: QueryOptions | None,
        rerank_required: bool,
    ) -> tuple[list[CandidateLike], int, int, int]:
        candidate_count = sum(len(branch) for _, branch in branches)
        fused_candidates = self.fusion.fuse(query=query, mode=mode, branches=branches)
        if self.telemetry_service is not None:
            self.telemetry_service.record_rrf_fusion(
                branch_count=len(branches),
                candidate_count=candidate_count,
                fused_count=len(fused_candidates),
                duplicate_count=max(0, candidate_count - len(fused_candidates)),
            )

        reranked_candidates = (
            self._rerank_candidates(
                query=query,
                fused_candidates=fused_candidates,
                rerank_pool_k=(query_options.rerank_pool_k if query_options is not None else None),
            )
            if rerank_required and (query_options is None or query_options.enable_rerank)
            else list(fused_candidates)
        )
        parent_backfilled_count = 0
        collapsed_candidate_count = 0
        collapsed_candidates: list[CandidateLike] = []
        seen_keys: set[tuple[str, str, str]] = set()
        enable_parent_backfill = query_options.enable_parent_backfill if query_options is not None else True
        for candidate in reranked_candidates:
            parent_text = getattr(candidate, "parent_text", None)
            parent_chunk_id = getattr(candidate, "parent_chunk_id", None)
            if enable_parent_backfill and parent_chunk_id and parent_text:
                candidate = FusedCandidateView(
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
                parent_backfilled_count += 1
            if candidate.source_kind != "internal":
                collapsed_candidates.append(candidate)
                continue
            special_chunk_type = getattr(candidate, "special_chunk_type", None)
            parent_chunk_id = getattr(candidate, "parent_chunk_id", None)
            dedupe_key = (
                ("chunk", candidate.doc_id, candidate.chunk_id)
                if special_chunk_type or not parent_chunk_id
                else ("parent", candidate.doc_id, parent_chunk_id)
            )
            if dedupe_key in seen_keys:
                collapsed_candidate_count += 1
                continue
            seen_keys.add(dedupe_key)
            collapsed_candidates.append(candidate)
        if query_options is None:
            reranked_candidates = collapsed_candidates
        else:
            limit = query_options.chunk_top_k or query_options.top_k
            reranked_candidates = [] if limit <= 0 else collapsed_candidates[:limit]
        if self.telemetry_service is not None:
            fused_ids = [candidate.chunk_id for candidate in fused_candidates]
            reranked_ids = [candidate.chunk_id for candidate in reranked_candidates]
            self.telemetry_service.record_rerank_effectiveness(
                input_count=len(fused_candidates),
                output_count=len(reranked_candidates),
                reordered=fused_ids != reranked_ids,
                top1_changed=(fused_ids[:1] != reranked_ids[:1]),
            )
        return reranked_candidates, candidate_count, parent_backfilled_count, collapsed_candidate_count

    def _rerank_candidates(
        self,
        *,
        query: str,
        fused_candidates: list[CandidateLike],
        rerank_pool_k: int | None,
    ) -> list[CandidateLike]:
        if not fused_candidates:
            return []
        if rerank_pool_k is None:
            return self.reranker.rerank(query, list(fused_candidates))
        normalized_limit = max(1, rerank_pool_k)
        head = list(fused_candidates[:normalized_limit])
        tail = list(fused_candidates[normalized_limit:])
        reranked_head = self.reranker.rerank(query, head)
        return [*reranked_head, *tail]

    def _call_branch(
        self,
        branch_name: str,
        *,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
    ) -> Sequence[CandidateLike]:
        retriever = self.branch_registry.get(branch_name)
        return retriever(query, source_scope, query_understanding)

    def _embedding_provider(self) -> str | None:
        for retriever in (
            self.branch_registry.local_retriever,
            self.branch_registry.global_retriever,
            self.branch_registry.vector_retriever,
            self.branch_registry.special_retriever,
        ):
            provider = getattr(retriever, "last_provider", None)
            if isinstance(provider, str) and provider:
                return provider
        return None

    def _provider_attempts(self) -> list[ProviderAttempt]:
        attempts: list[ProviderAttempt] = []
        for retriever in (
            self.branch_registry.local_retriever,
            self.branch_registry.global_retriever,
            self.branch_registry.vector_retriever,
            self.branch_registry.special_retriever,
        ):
            attempts.extend(
                attempt
                for attempt in getattr(retriever, "last_attempts", [])
                if isinstance(attempt, ProviderAttempt)
            )
        attempts.extend(
            attempt
            for attempt in getattr(self.reranker.reranker, "last_attempts", [])
            if isinstance(attempt, ProviderAttempt)
        )
        return attempts

    def retrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_mode: QueryMode | str | None = None,
        query_options: QueryOptions | None = None,
    ) -> RetrievalResult:
        scope = list(source_scope)
        self._prepare_retriever_policies(
            access_policy=access_policy,
            execution_location_preference=execution_location_preference,
        )
        result = self.run(
            query,
            access_policy=access_policy,
            source_scope=scope,
            execution_location_preference=execution_location_preference,
            query_mode=query_mode,
            query_options=query_options,
        )
        self.last_result = result
        return result

    def _prepare_retriever_policies(
        self,
        *,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference | None,
    ) -> None:
        for retriever in (
            self._vector_retriever,
            self._local_retriever,
            self._global_retriever,
            self._special_retriever,
        ):
            prepare_for_policy = getattr(retriever, "prepare_for_policy", None)
            if callable(prepare_for_policy):
                prepare_for_policy(
                    access_policy=access_policy,
                    execution_location_preference=execution_location_preference,
                )


__all__ = [
    "BranchRetrieverRegistry",
    "GraphExpander",
    "ReciprocalRankFusion",
    "RetrievalService",
    "Reranker",
    "RetrieverFn",
    "UnifiedReranker",
]
