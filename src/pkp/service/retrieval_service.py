from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pkp.algorithms.retrieval.branch_retrievers import BranchRetrieverRegistry
from pkp.algorithms.retrieval.contracts import GraphExpander, Reranker, RetrieverFn
from pkp.algorithms.retrieval.fusion import ReciprocalRankFusion
from pkp.algorithms.retrieval.mode_planner import RetrievalPlanBuilder
from pkp.algorithms.retrieval.rerank import UnifiedReranker
from pkp.config.policies import RoutingThresholds
from pkp.query.query import QueryMode, QueryPipeline
from pkp.service.artifact_service import ArtifactService
from pkp.service.evidence_service import EvidenceService
from pkp.service.graph_expansion_service import GraphExpansionService
from pkp.service.query_understanding_service import QueryUnderstandingService
from pkp.service.routing_service import RoutingService
from pkp.service.telemetry_service import TelemetryService
from pkp.schema.document import AccessPolicy, ExecutionLocationPreference
from pkp.schema.query import RetrievalResult


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
        thresholds: RoutingThresholds | None = None,
    ) -> None:
        self._full_text_retriever = full_text_retriever or (lambda _query, _scope: [])
        self._vector_retriever = vector_retriever or (lambda _query, _scope: [])
        self._local_retriever = local_retriever or (lambda _query, _scope: [])
        self._global_retriever = global_retriever or (lambda _query, _scope: [])
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
        self._mode_planner = RetrievalPlanBuilder()
        self._fusion = ReciprocalRankFusion()
        self._unified_reranker = UnifiedReranker(reranker=self._reranker)
        self.last_result: RetrievalResult | None = None
        self._branch_registry = BranchRetrieverRegistry(
            full_text_retriever=cast(RetrieverFn, self._full_text_retriever),
            vector_retriever=cast(RetrieverFn, self._vector_retriever),
            local_retriever=cast(RetrieverFn, self._local_retriever),
            global_retriever=cast(RetrieverFn, self._global_retriever),
            section_retriever=cast(RetrieverFn, self._section_retriever),
            special_retriever=cast(RetrieverFn, self._special_retriever),
            metadata_retriever=cast(RetrieverFn, self._metadata_retriever),
            web_retriever=cast(RetrieverFn, self._web_retriever),
        )
        self._query_pipeline = QueryPipeline(
            branch_registry=self._branch_registry,
            routing_service=self._routing_service,
            query_understanding_service=self._query_understanding_service,
            evidence_service=self._evidence_service,
            graph_expansion_service=self._graph_expansion_service,
            artifact_service=self._artifact_service,
            fusion=self._fusion,
            reranker=self._unified_reranker,
            graph_expander=cast(GraphExpander, self._graph_expander),
            telemetry_service=self._telemetry_service,
            mode_planner=self._mode_planner,
        )

    def retrieve(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        source_scope: Sequence[str] = (),
        execution_location_preference: ExecutionLocationPreference | None = None,
        query_mode: QueryMode | str | None = None,
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
        )
        self.last_result = result
        return result

    def _prepare_retriever_policies(
        self,
        *,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference | None,
    ) -> None:
        for retriever in (self._vector_retriever, self._local_retriever, self._global_retriever, self._special_retriever):
            prepare_for_policy = getattr(retriever, "prepare_for_policy", None)
            if callable(prepare_for_policy):
                prepare_for_policy(
                    access_policy=access_policy,
                    execution_location_preference=execution_location_preference,
                )
