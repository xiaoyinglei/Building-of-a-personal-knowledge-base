from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pkp.algorithms.retrieval.branch_retrievers import BranchRetrieverRegistry
from pkp.algorithms.retrieval.contracts import GraphExpander
from pkp.algorithms.retrieval.fusion import FusedCandidateView, ReciprocalRankFusion
from pkp.algorithms.retrieval.mode_planner import RetrievalPlanBuilder
from pkp.algorithms.retrieval.rerank import UnifiedReranker
from pkp.core.query_modes import QueryMode
from pkp.service.artifact_service import ArtifactService
from pkp.service.evidence_service import CandidateLike, EvidenceBundle, EvidenceService
from pkp.service.graph_expansion_service import GraphExpansionService
from pkp.service.query_understanding_service import QueryUnderstandingService
from pkp.service.routing_service import RoutingService
from pkp.service.telemetry_service import TelemetryService
from pkp.types.access import AccessPolicy, ExecutionLocationPreference
from pkp.types.diagnostics import RetrievalDiagnostics
from pkp.types.envelope import PreservationSuggestion
from pkp.types.retrieval import RetrievalResult


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
    mode_planner: RetrievalPlanBuilder | None = None

    def __post_init__(self) -> None:
        if self.mode_planner is None:
            self.mode_planner = RetrievalPlanBuilder()

    def run(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_mode: QueryMode | str | None = None,
    ) -> RetrievalResult:
        scope = list(source_scope)
        decision = self.routing_service.route(query, source_scope=scope, access_policy=access_policy)
        query_understanding = self.query_understanding_service.analyze(query)
        plan = self.mode_planner.build(
            query_understanding=query_understanding,
            requested_mode=query_mode,
        )

        raw_internal = self.branch_registry.collect_internal(plan=plan, query=query, source_scope=scope)
        internal_branches: list[tuple[str, list[CandidateLike]]] = []
        branch_hits: dict[str, int] = {}
        for branch_name, candidates in raw_internal.items():
            filtered = self.evidence_service.filter_candidates(
                candidates,
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
            branch_hits[branch_name] = len(filtered)
            if self.telemetry_service is not None:
                self.telemetry_service.record_branch_usage(
                    branch=branch_name,
                    hit_count=len(filtered),
                    runtime_mode=decision.runtime_mode.value,
                )
            if filtered:
                internal_branches.append((branch_name, filtered))

        fused_candidates = self.fusion.fuse(query=query, mode=plan.mode, branches=internal_branches)
        candidate_count = sum(len(branch) for _, branch in internal_branches)
        if self.telemetry_service is not None:
            self.telemetry_service.record_rrf_fusion(
                branch_count=len(internal_branches),
                candidate_count=candidate_count,
                fused_count=len(fused_candidates),
                duplicate_count=max(0, candidate_count - len(fused_candidates)),
            )

        reranked_candidates = self.reranker.rerank(query, fused_candidates)
        reranked_candidates, parent_backfilled_count = self._apply_parent_backfill(reranked_candidates)
        reranked_candidates, collapsed_candidate_count = self._collapse_redundant_candidates(reranked_candidates)
        self._record_rerank_effectiveness(fused_candidates=fused_candidates, reranked_candidates=reranked_candidates)
        evidence = self.evidence_service.assemble_bundle(reranked_candidates)
        self_check = self.evidence_service.evaluate_self_check(
            bundle=evidence,
            task_type=decision.task_type,
            complexity_level=decision.complexity_level,
        )

        web_candidates: list[CandidateLike] = []
        if (
            plan.allow_web
            and decision.web_search_allowed
            and access_policy.external_retrieval.value == "allow"
            and self_check.retrieve_more
        ):
            web_candidates = self.evidence_service.filter_candidates(
                self.branch_registry.collect_web(query=query, source_scope=scope),
                source_scope=scope,
                access_policy=access_policy,
                runtime_mode=decision.runtime_mode,
            )
            branch_hits["web"] = len(web_candidates)
            if self.telemetry_service is not None:
                self.telemetry_service.record_branch_usage(
                    branch="web",
                    hit_count=len(web_candidates),
                    runtime_mode=decision.runtime_mode.value,
                )
            if web_candidates:
                combined_branches = [*internal_branches, ("web", web_candidates)]
                combined_candidate_count = sum(len(branch) for _, branch in combined_branches)
                fused_candidates = self.fusion.fuse(query=query, mode=plan.mode, branches=combined_branches)
                if self.telemetry_service is not None:
                    self.telemetry_service.record_rrf_fusion(
                        branch_count=len(combined_branches),
                        candidate_count=combined_candidate_count,
                        fused_count=len(fused_candidates),
                        duplicate_count=max(0, combined_candidate_count - len(fused_candidates)),
                    )
                reranked_candidates = self.reranker.rerank(query, fused_candidates)
                reranked_candidates, parent_backfilled_count = self._apply_parent_backfill(reranked_candidates)
                reranked_candidates, collapsed_candidate_count = self._collapse_redundant_candidates(
                    reranked_candidates
                )
                self._record_rerank_effectiveness(
                    fused_candidates=fused_candidates,
                    reranked_candidates=reranked_candidates,
                )
                evidence = self.evidence_service.assemble_bundle(reranked_candidates)

        graph_expanded = False
        if plan.allow_graph_expansion and decision.graph_expansion_allowed:
            internal_candidates = [
                candidate for candidate in reranked_candidates if candidate.source_kind == "internal"
            ]
            graph_candidates = self.graph_expansion_service.expand(
                query=query,
                source_scope=scope,
                evidence=evidence,
                graph_candidates=self.graph_expander(query, scope, internal_candidates),
                access_policy=access_policy,
            )
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
                branch_hits=branch_hits,
                reranked_chunk_ids=[candidate.chunk_id for candidate in reranked_candidates],
                embedding_provider=self._embedding_provider(),
                rerank_provider=getattr(self.reranker.reranker, "last_provider", None),
                attempts=self._provider_attempts(),
                fusion_input_count=candidate_count + len(web_candidates),
                fused_count=len(reranked_candidates),
                graph_expanded=graph_expanded,
                query_understanding=query_understanding,
                parent_backfilled_count=parent_backfilled_count,
                collapsed_candidate_count=collapsed_candidate_count,
            ),
            preservation_suggestion=preservation_suggestion,
        )

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
        ):
            provider = getattr(retriever, "last_provider", None)
            if provider:
                return provider
        return None

    def _provider_attempts(self) -> list[object]:
        attempts: list[object] = []
        for retriever in (
            self.branch_registry.local_retriever,
            self.branch_registry.global_retriever,
            self.branch_registry.vector_retriever,
        ):
            attempts.extend(list(getattr(retriever, "last_attempts", [])))
        attempts.extend(list(getattr(self.reranker.reranker, "last_attempts", [])))
        return attempts
