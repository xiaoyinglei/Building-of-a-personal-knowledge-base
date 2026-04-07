from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from rag.llm._generation.answer_generator import AnswerGenerator
from rag.query.analysis import (
    QueryUnderstandingService,
    RoutingDecision,
    RoutingService,
    narrow_access_policy_for_query,
)
from rag.query.artifact import ArtifactService
from rag.query.branches import BranchRetrieverRegistry, GraphExpander, Reranker, RetrieverFn, UnifiedReranker
from rag.query.context import (
    CandidateLike,
    ContextEvidenceMerger,
    ContextPromptBuilder,
    ContextPromptBuildResult,
    ContextTruncationResult,
    EvidenceBundle,
    EvidenceService,
    EvidenceThresholds,
    EvidenceTruncator,
    SelfCheckResult,
)
from rag.query.fusion import FusedCandidateView, ReciprocalRankFusion
from rag.query.graph import GraphExpansionService
from rag.query.query import BuiltContext, QueryMode, QueryOptions, RAGQueryResult, normalize_query_mode
from rag.schema._types.diagnostics import ProviderAttempt, RetrievalDiagnostics
from rag.schema._types.envelope import EvidenceItem
from rag.schema._types.query import QueryUnderstanding
from rag.schema.document import AccessPolicy, ExecutionLocationPreference
from rag.schema.query import RetrievalResult
from rag.utils._telemetry import TelemetryService


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


@dataclass(slots=True)
class QueryPipeline:
    branch_registry: BranchRetrieverRegistry
    routing_service: RoutingService
    query_understanding_service: QueryUnderstandingService
    evidence_service: EvidenceService
    graph_expansion_service: GraphExpansionService
    artifact_service: ArtifactService
    fusion: ReciprocalRankFusion
    reranker: UnifiedReranker
    graph_expander: GraphExpander
    telemetry_service: TelemetryService | None = None

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
        if resolved_mode is QueryMode.NAIVE:
            return self._run_naive_mode(
                query=query,
                source_scope=scope,
                access_policy=effective_access_policy,
                decision=decision,
                query_understanding=query_understanding,
                query_options=query_options,
            )
        if resolved_mode is QueryMode.LOCAL:
            return self._run_local_mode(
                query=query,
                source_scope=scope,
                access_policy=effective_access_policy,
                decision=decision,
                query_understanding=query_understanding,
                query_options=query_options,
            )
        if resolved_mode is QueryMode.GLOBAL:
            return self._run_global_mode(
                query=query,
                source_scope=scope,
                access_policy=effective_access_policy,
                decision=decision,
                query_understanding=query_understanding,
                query_options=query_options,
            )
        if resolved_mode is QueryMode.HYBRID:
            return self._run_hybrid_mode(
                query=query,
                source_scope=scope,
                access_policy=effective_access_policy,
                decision=decision,
                query_understanding=query_understanding,
                query_options=query_options,
            )
        return self._run_mix_mode(
            query=query,
            source_scope=scope,
            access_policy=effective_access_policy,
            decision=decision,
            query_understanding=query_understanding,
            query_options=query_options,
        )

    def _run_naive_mode(
        self,
        *,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
        query_options: QueryOptions | None,
    ) -> RetrievalResult:
        retrieval_limit = self._requested_retrieval_limit(query_options)
        return self._execute_mode(
            query=query,
            source_scope=source_scope,
            access_policy=access_policy,
            decision=decision,
            query_understanding=query_understanding,
            query_options=query_options,
            spec=ModeExecutionSpec(
                mode=QueryMode.NAIVE,
                executor_name="naive",
                internal_branches=(BranchExecutionSpec("vector", retrieval_limit * 2),),
                allow_web=False,
                allow_graph_expansion=False,
            ),
        )

    def _run_local_mode(
        self,
        *,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
        query_options: QueryOptions | None,
    ) -> RetrievalResult:
        retrieval_limit = self._requested_retrieval_limit(query_options)
        aux_branches = self._auxiliary_branch_specs(query_understanding, retrieval_limit)
        return self._execute_mode(
            query=query,
            source_scope=source_scope,
            access_policy=access_policy,
            decision=decision,
            query_understanding=query_understanding,
            query_options=query_options,
            spec=ModeExecutionSpec(
                mode=QueryMode.LOCAL,
                executor_name="local",
                internal_branches=(BranchExecutionSpec("local", retrieval_limit * 2), *aux_branches),
                allow_web=True,
                allow_graph_expansion=True,
                web_limit=max(1, retrieval_limit // 2),
                graph_limit=max(2, self._requested_final_limit(query_options)),
            ),
        )

    def _run_global_mode(
        self,
        *,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
        query_options: QueryOptions | None,
    ) -> RetrievalResult:
        retrieval_limit = self._requested_retrieval_limit(query_options)
        aux_branches = self._auxiliary_branch_specs(query_understanding, retrieval_limit)
        return self._execute_mode(
            query=query,
            source_scope=source_scope,
            access_policy=access_policy,
            decision=decision,
            query_understanding=query_understanding,
            query_options=query_options,
            spec=ModeExecutionSpec(
                mode=QueryMode.GLOBAL,
                executor_name="global",
                internal_branches=(BranchExecutionSpec("global", retrieval_limit * 2), *aux_branches),
                allow_web=True,
                allow_graph_expansion=True,
                web_limit=max(1, retrieval_limit // 2),
                graph_limit=max(2, self._requested_final_limit(query_options)),
            ),
        )

    def _run_hybrid_mode(
        self,
        *,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
        query_options: QueryOptions | None,
    ) -> RetrievalResult:
        retrieval_limit = self._requested_retrieval_limit(query_options)
        aux_branches = self._auxiliary_branch_specs(query_understanding, retrieval_limit)
        return self._execute_mode(
            query=query,
            source_scope=source_scope,
            access_policy=access_policy,
            decision=decision,
            query_understanding=query_understanding,
            query_options=query_options,
            spec=ModeExecutionSpec(
                mode=QueryMode.HYBRID,
                executor_name="hybrid",
                internal_branches=(
                    BranchExecutionSpec("local", retrieval_limit),
                    BranchExecutionSpec("global", retrieval_limit),
                    *aux_branches,
                ),
                allow_web=True,
                allow_graph_expansion=True,
                web_limit=max(1, retrieval_limit // 2),
                graph_limit=max(2, self._requested_final_limit(query_options)),
            ),
        )

    def _run_mix_mode(
        self,
        *,
        query: str,
        source_scope: list[str],
        access_policy: AccessPolicy,
        decision: RoutingDecision,
        query_understanding: QueryUnderstanding,
        query_options: QueryOptions | None,
    ) -> RetrievalResult:
        retrieval_limit = self._requested_retrieval_limit(query_options)
        aux_branches = self._auxiliary_branch_specs(query_understanding, retrieval_limit)
        text_branches = self._mix_text_branches(query_understanding, retrieval_limit)
        kg_limit = max(2, retrieval_limit - 1)
        return self._execute_mode(
            query=query,
            source_scope=source_scope,
            access_policy=access_policy,
            decision=decision,
            query_understanding=query_understanding,
            query_options=query_options,
            spec=ModeExecutionSpec(
                mode=QueryMode.MIX,
                executor_name="mix",
                internal_branches=(
                    BranchExecutionSpec("local", kg_limit),
                    BranchExecutionSpec("global", kg_limit),
                    *text_branches,
                    *aux_branches,
                ),
                allow_web=True,
                allow_graph_expansion=True,
                web_limit=max(1, retrieval_limit // 2),
                graph_limit=max(2, self._requested_final_limit(query_options)),
            ),
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
            graph_expanded=graph_expanded,
            diagnostics=RetrievalDiagnostics(
                mode_executor=spec.executor_name,
                branch_hits=branch_hits,
                branch_limits=branch_limits,
                reranked_chunk_ids=[candidate.chunk_id for candidate in reranked_candidates],
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
            graph_expanded=False,
            diagnostics=RetrievalDiagnostics(
                mode_executor="bypass",
                branch_hits={},
                branch_limits={},
                reranked_chunk_ids=[],
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

        reranked_candidates = self._rerank_candidates(
            query,
            fused_candidates,
            query_options=query_options,
            rerank_required=rerank_required,
        )
        reranked_candidates, parent_backfilled_count = self._apply_parent_backfill(reranked_candidates)
        reranked_candidates, collapsed_candidate_count = self._collapse_redundant_candidates(reranked_candidates)
        reranked_candidates = self._limit_candidates(reranked_candidates, query_options=query_options)
        self._record_rerank_effectiveness(fused_candidates=fused_candidates, reranked_candidates=reranked_candidates)
        return reranked_candidates, candidate_count, parent_backfilled_count, collapsed_candidate_count

    def _auxiliary_branch_specs(
        self,
        query_understanding: QueryUnderstanding,
        retrieval_limit: int,
    ) -> tuple[BranchExecutionSpec, ...]:
        branches: list[str] = []
        if query_understanding.needs_structure:
            branches.append("section")
        if query_understanding.needs_special:
            branches.append("special")
        if query_understanding.needs_metadata:
            branches.append("metadata")
        if not branches:
            return ()
        branch_limit = max(1, retrieval_limit)
        return tuple(BranchExecutionSpec(branch, branch_limit) for branch in branches)

    @staticmethod
    def _mix_text_branches(
        query_understanding: QueryUnderstanding,
        retrieval_limit: int,
    ) -> tuple[BranchExecutionSpec, ...]:
        del query_understanding
        return (
            BranchExecutionSpec("vector", retrieval_limit),
            BranchExecutionSpec("full_text", retrieval_limit),
        )

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

    @staticmethod
    def _requested_retrieval_limit(query_options: QueryOptions | None) -> int:
        if query_options is None:
            return 8
        return max(query_options.top_k, 1)

    @staticmethod
    def _requested_final_limit(query_options: QueryOptions | None) -> int:
        if query_options is None:
            return 8
        return max(query_options.chunk_top_k or query_options.top_k, 1)

    def _rerank_candidates(
        self,
        query: str,
        candidates: Sequence[CandidateLike],
        *,
        query_options: QueryOptions | None,
        rerank_required: bool,
    ) -> list[CandidateLike]:
        if not rerank_required:
            return list(candidates)
        if query_options is not None and not query_options.enable_rerank:
            return list(candidates)
        return self.reranker.rerank(query, list(candidates))

    @staticmethod
    def _limit_candidates(
        candidates: Sequence[CandidateLike],
        *,
        query_options: QueryOptions | None,
    ) -> list[CandidateLike]:
        if query_options is None:
            return list(candidates)
        limit = query_options.chunk_top_k or query_options.top_k
        if limit <= 0:
            return []
        return list(candidates[:limit])

    @staticmethod
    def _apply_parent_backfill(candidates: Sequence[CandidateLike]) -> tuple[list[CandidateLike], int]:
        enriched: list[CandidateLike] = []
        backfilled = 0
        for candidate in candidates:
            parent_text = getattr(candidate, "parent_text", None)
            parent_chunk_id = getattr(candidate, "parent_chunk_id", None)
            if parent_chunk_id and parent_text:
                enriched.append(
                    FusedCandidateView(
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

    def _record_rerank_effectiveness(
        self,
        *,
        fused_candidates: Sequence[CandidateLike],
        reranked_candidates: Sequence[CandidateLike],
    ) -> None:
        if self.telemetry_service is None:
            return
        fused_ids = [candidate.chunk_id for candidate in fused_candidates]
        reranked_ids = [candidate.chunk_id for candidate in reranked_candidates]
        self.telemetry_service.record_rerank_effectiveness(
            input_count=len(fused_candidates),
            output_count=len(reranked_candidates),
            reordered=fused_ids != reranked_ids,
            top1_changed=(fused_ids[:1] != reranked_ids[:1]),
        )

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
                attempt for attempt in getattr(retriever, "last_attempts", []) if isinstance(attempt, ProviderAttempt)
            )
        attempts.extend(
            attempt
            for attempt in getattr(self.reranker.reranker, "last_attempts", [])
            if isinstance(attempt, ProviderAttempt)
        )
        return attempts


@dataclass(slots=True)
class RAGQueryPipeline:
    retrieval: RetrievalExecutor
    context_merger: ContextEvidenceMerger
    truncator: EvidenceTruncator
    prompt_builder: ContextPromptBuilder
    answer_generator: AnswerGenerator

    def run(
        self,
        query: str,
        *,
        options: QueryOptions,
    ) -> RAGQueryResult:
        retrieval = self.retrieval.retrieve(
            query,
            access_policy=options.access_policy,
            source_scope=options.source_scope,
            execution_location_preference=options.execution_location_preference,
            query_mode=options.mode,
            query_options=options,
        )
        if normalize_query_mode(options.mode) is QueryMode.BYPASS:
            prompt = self.prompt_builder.answer_generation_service.build_direct_prompt(
                query=query,
                response_type=options.response_type,
                user_prompt=options.user_prompt,
                conversation_history=options.conversation_history,
            )
            generated = self.answer_generator.generate_direct(
                query=query,
                prompt=prompt,
                access_policy=options.access_policy,
                execution_location_preference=options.execution_location_preference,
            )
            return RAGQueryResult(
                query=query,
                mode=str(options.mode),
                answer=generated.answer,
                retrieval=retrieval,
                context=BuiltContext(
                    evidence=[],
                    token_budget=options.max_context_tokens,
                    token_count=self.prompt_builder.token_accounting.count(prompt),
                    truncated_count=0,
                    grounded_candidate="Bypass mode does not use retrieved evidence.",
                    prompt=prompt,
                ),
                generation_provider=generated.provider,
                generation_model=generated.model,
                generation_attempts=generated.attempts,
            )
        merged_evidence = self.context_merger.merge(retrieval)
        total_budget = max(options.max_context_tokens, 1)
        evidence_budget = self.truncator.token_accounting.prompt_budget(total_budget)
        truncated, prompt_build = self._build_bounded_context(
            query=query,
            options=options,
            retrieval=retrieval,
            merged_evidence=merged_evidence,
            total_budget=total_budget,
            evidence_budget=evidence_budget,
        )
        context_evidence_items = [item.as_evidence_item() for item in truncated.evidence]
        generated = self.answer_generator.generate(
            query=query,
            prompt=prompt_build.prompt,
            evidence_pack=context_evidence_items,
            grounded_candidate=prompt_build.grounded_candidate,
            runtime_mode=retrieval.decision.runtime_mode,
            access_policy=options.access_policy,
            execution_location_preference=options.execution_location_preference,
        )
        return RAGQueryResult(
            query=query,
            mode=str(options.mode),
            answer=generated.answer,
            retrieval=retrieval,
            context=BuiltContext(
                evidence=truncated.evidence,
                token_budget=total_budget,
                token_count=prompt_build.token_count,
                truncated_count=truncated.truncated_count,
                grounded_candidate=prompt_build.grounded_candidate,
                prompt=prompt_build.prompt,
            ),
            generation_provider=generated.provider,
            generation_model=generated.model,
            generation_attempts=generated.attempts,
        )

    def _build_bounded_context(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: RetrievalResult,
        merged_evidence: list[EvidenceItem],
        total_budget: int,
        evidence_budget: int,
    ) -> tuple[ContextTruncationResult, ContextPromptBuildResult]:
        current_budget = max(evidence_budget, 1)
        truncated = self.truncator.truncate(
            merged_evidence,
            token_budget=current_budget,
            max_evidence_chunks=options.max_evidence_chunks,
            mode=options.mode,
        )
        prompt_build = self._build_prompt_from_truncation(
            query=query,
            options=options,
            retrieval=retrieval,
            truncated=truncated,
            prompt_style="full",
            user_prompt=options.user_prompt,
            conversation_history=options.conversation_history,
        )
        while prompt_build.token_count > total_budget and truncated.evidence and current_budget > 1:
            overflow = prompt_build.token_count - total_budget
            current_budget = max(current_budget - max(overflow, 1), 1)
            retruncated = self.truncator.truncate(
                merged_evidence,
                token_budget=current_budget,
                max_evidence_chunks=options.max_evidence_chunks,
                mode=options.mode,
            )
            if (
                retruncated.token_count >= truncated.token_count
                and len(retruncated.evidence) >= len(truncated.evidence)
            ):
                break
            truncated = retruncated
            prompt_build = self._build_prompt_from_truncation(
                query=query,
                options=options,
                retrieval=retrieval,
                truncated=truncated,
                prompt_style="full",
                user_prompt=options.user_prompt,
                conversation_history=options.conversation_history,
            )
        if prompt_build.token_count > total_budget:
            prompt_build = self._build_reduced_prompt(
                query=query,
                options=options,
                retrieval=retrieval,
                truncated=truncated,
            )
            while prompt_build.token_count > total_budget and truncated.evidence and current_budget > 1:
                overflow = prompt_build.token_count - total_budget
                current_budget = max(current_budget - max(overflow, 1), 1)
                retruncated = self.truncator.truncate(
                    merged_evidence,
                    token_budget=current_budget,
                    max_evidence_chunks=options.max_evidence_chunks,
                    mode=options.mode,
                )
                if (
                    retruncated.token_count >= truncated.token_count
                    and len(retruncated.evidence) >= len(truncated.evidence)
                ):
                    break
                truncated = retruncated
                prompt_build = self._build_reduced_prompt(
                    query=query,
                    options=options,
                    retrieval=retrieval,
                    truncated=truncated,
                )
        return truncated, prompt_build

    def _build_reduced_prompt(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: RetrievalResult,
        truncated: ContextTruncationResult,
    ) -> ContextPromptBuildResult:
        prompt_build = self._build_prompt_from_truncation(
            query=query,
            options=options,
            retrieval=retrieval,
            truncated=truncated,
            prompt_style="compact",
            user_prompt=options.user_prompt,
            conversation_history=options.conversation_history,
        )
        if prompt_build.token_count <= options.max_context_tokens:
            return prompt_build
        prompt_build = self._build_prompt_from_truncation(
            query=query,
            options=options,
            retrieval=retrieval,
            truncated=truncated,
            prompt_style="compact",
            user_prompt=options.user_prompt,
            conversation_history=(),
        )
        if prompt_build.token_count <= options.max_context_tokens:
            return prompt_build
        prompt_build = self._build_prompt_from_truncation(
            query=query,
            options=options,
            retrieval=retrieval,
            truncated=truncated,
            prompt_style="compact",
            user_prompt=None,
            conversation_history=(),
        )
        if prompt_build.token_count <= options.max_context_tokens:
            return prompt_build
        return self._build_prompt_from_truncation(
            query=query,
            options=options,
            retrieval=retrieval,
            truncated=truncated,
            prompt_style="minimal",
            user_prompt=None,
            conversation_history=(),
        )

    def _build_prompt_from_truncation(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: RetrievalResult,
        truncated: ContextTruncationResult,
        prompt_style: Literal["full", "compact", "minimal"],
        user_prompt: str | None,
        conversation_history: Sequence[tuple[str, str]],
    ) -> ContextPromptBuildResult:
        context_evidence_items = [item.as_evidence_item() for item in truncated.evidence]
        grounded_candidate = self.answer_generator.grounded_candidate(
            query,
            context_evidence_items,
            query_understanding=retrieval.diagnostics.query_understanding,
        )
        return self.prompt_builder.build(
            query=query,
            grounded_candidate=grounded_candidate,
            evidence=truncated.evidence,
            runtime_mode=retrieval.decision.runtime_mode,
            response_type=options.response_type,
            user_prompt=user_prompt,
            conversation_history=conversation_history,
            prompt_style=prompt_style,
        )


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
        self._query_pipeline: QueryPipeline = QueryPipeline(
            branch_registry=self._branch_registry,
            routing_service=self._routing_service,
            query_understanding_service=self._query_understanding_service,
            evidence_service=self._evidence_service,
            graph_expansion_service=self._graph_expansion_service,
            artifact_service=self._artifact_service,
            fusion=self._fusion,
            reranker=self._unified_reranker,
            graph_expander=self._graph_expander,
            telemetry_service=self._telemetry_service,
        )

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
        result = self._query_pipeline.run(
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
    "FusedCandidateView",
    "GraphExpander",
    "QueryPipeline",
    "RAGQueryPipeline",
    "ReciprocalRankFusion",
    "RetrievalService",
    "Reranker",
    "RetrieverFn",
    "UnifiedReranker",
]
