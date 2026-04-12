from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rag.assembly import (
    AssemblyDiagnostics,
    AssemblyRequest,
    CapabilityAssemblyService,
    CapabilityBundle,
    CapabilityCatalog,
    CapabilityRequirements,
    TokenAccountingService,
    TokenizerContract,
)
from rag.ingest.chunking import ChunkingService, DocumentProcessingService, TOCService
from rag.ingest.parser import (
    DoclingParserRepo,
    HttpWebFetchRepo,
    ImageParserRepo,
    MarkdownParserRepo,
    PDFParserRepo,
    PlainTextParserRepo,
    WebParserRepo,
    create_default_ocr_repo,
)
from rag.ingest.pipeline import (
    BatchIngestResult,
    DeletePipeline,
    DeletePipelineResult,
    DeleteRequest,
    DirectContentItem,
    IngestPipeline,
    IngestPipelineResult,
    IngestRequest,
    RebuildPipeline,
    RebuildPipelineResult,
    RebuildRequest,
)
from rag.ingest.policy import PolicyResolutionService
from rag.providers.generation import AnswerGenerationService, AnswerGenerator
from rag.providers.rerank import ModelBackedRerankService
from rag.retrieval.analysis import QueryUnderstandingService, RoutingService
from rag.retrieval.context import (
    ContextPromptBuilder,
    ContextPromptBuildResult,
    ContextTruncationResult,
    EvidenceTruncator,
)
from rag.retrieval.evidence import ArtifactService, CandidateLike, ContextEvidenceMerger, EvidenceService
from rag.retrieval.graph import GraphExpansionService, SearchBackedRetrievalFactory
from rag.retrieval.models import BuiltContext, QueryMode, QueryOptions, RAGQueryResult, normalize_query_mode
from rag.retrieval.orchestrator import RetrievalService
from rag.schema.core import GraphEdge, GraphNode
from rag.schema.query import EvidenceItem
from rag.schema.runtime import CacheEntry, ProviderAttempt, VisualDescriptionRepo
from rag.storage import StorageBundle, StorageConfig
from rag.utils.telemetry import TelemetryService

_RUNTIME_CONTRACT_NAMESPACE = "rag_runtime"
_RUNTIME_CONTRACT_KEY = "core_contract_v1"


class _InstrumentedReranker:
    def __init__(self, rerank_service: object) -> None:
        self._rerank_service = rerank_service
        self.provider_name = getattr(rerank_service, "provider_name", "formal-rerank")
        self.rerank_model_name = getattr(rerank_service, "rerank_model_name", "unconfigured-reranker")
        self.last_provider: str | None = self.provider_name
        self.last_attempts: list[ProviderAttempt] = []

    def __call__(self, query: str, candidates: list[CandidateLike]) -> list[CandidateLike]:
        attempt = ProviderAttempt(
            stage="rerank",
            capability="rerank",
            provider=self.provider_name,
            location="core",
            model=self.rerank_model_name,
            status="success",
        )
        rerank = getattr(self._rerank_service, "rerank", None)
        if not callable(rerank):
            self.last_attempts = [attempt.model_copy(update={"status": "failed", "error": "rerank not supported"})]
            return candidates
        try:
            result = list(rerank(query, candidates))
        except RuntimeError as exc:
            self.last_attempts = [attempt.model_copy(update={"status": "failed", "error": str(exc)})]
            raise
        self.provider_name = getattr(self._rerank_service, "provider_name", self.provider_name)
        self.rerank_model_name = getattr(self._rerank_service, "rerank_model_name", self.rerank_model_name)
        self.last_provider = self.provider_name
        self.last_attempts = [
            attempt.model_copy(update={"provider": self.provider_name, "model": self.rerank_model_name})
        ]
        return result


@dataclass(slots=True)
class _QueryPipeline:
    retrieval: RetrievalService
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
        retrieval: object,
        merged_evidence: list[EvidenceItem],
        total_budget: int,
        evidence_budget: int,
    ) -> tuple[ContextTruncationResult, ContextPromptBuildResult]:
        current_budget = max(evidence_budget, 1)
        truncated = self._truncate_evidence(merged_evidence, budget=current_budget, options=options)
        truncated, prompt_build, current_budget = self._shrink_to_budget(
            query=query,
            options=options,
            retrieval=retrieval,
            merged_evidence=merged_evidence,
            total_budget=total_budget,
            current_budget=current_budget,
            truncated=truncated,
            prompt_variants=(("full", options.user_prompt, options.conversation_history),),
        )
        if prompt_build.token_count > total_budget:
            truncated, prompt_build, _current_budget = self._shrink_to_budget(
                query=query,
                options=options,
                retrieval=retrieval,
                merged_evidence=merged_evidence,
                total_budget=total_budget,
                current_budget=current_budget,
                truncated=truncated,
                prompt_variants=(
                    ("compact", options.user_prompt, options.conversation_history),
                    ("compact", options.user_prompt, ()),
                    ("compact", None, ()),
                    ("minimal", None, ()),
                ),
            )
        return truncated, prompt_build

    def _shrink_to_budget(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: object,
        merged_evidence: list[EvidenceItem],
        total_budget: int,
        current_budget: int,
        truncated: ContextTruncationResult,
        prompt_variants: Sequence[tuple[str, str | None, Sequence[tuple[str, str]]]],
    ) -> tuple[ContextTruncationResult, ContextPromptBuildResult, int]:
        prompt_build = self._build_prompt_variants(
            query=query,
            options=options,
            retrieval=retrieval,
            truncated=truncated,
            prompt_variants=prompt_variants,
        )
        while prompt_build.token_count > total_budget and truncated.evidence and current_budget > 1:
            overflow = prompt_build.token_count - total_budget
            next_budget = max(current_budget - max(overflow, 1), 1)
            retruncated = self._truncate_evidence(merged_evidence, budget=next_budget, options=options)
            if (
                retruncated.token_count >= truncated.token_count
                and len(retruncated.evidence) >= len(truncated.evidence)
            ):
                break
            truncated = retruncated
            current_budget = next_budget
            prompt_build = self._build_prompt_variants(
                query=query,
                options=options,
                retrieval=retrieval,
                truncated=truncated,
                prompt_variants=prompt_variants,
            )
        return truncated, prompt_build, current_budget

    def _truncate_evidence(
        self,
        merged_evidence: list[EvidenceItem],
        *,
        budget: int,
        options: QueryOptions,
    ) -> ContextTruncationResult:
        max_evidence_chunks = options.max_evidence_chunks
        if options.answer_context_top_k is not None:
            max_evidence_chunks = min(max_evidence_chunks, max(options.answer_context_top_k, 1))
        return self.truncator.truncate(
            merged_evidence,
            token_budget=budget,
            max_evidence_chunks=max_evidence_chunks,
            mode=options.mode,
        )

    def _build_prompt_variants(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: object,
        truncated: ContextTruncationResult,
        prompt_variants: Sequence[tuple[str, str | None, Sequence[tuple[str, str]]]],
    ) -> ContextPromptBuildResult:
        last_prompt: ContextPromptBuildResult | None = None
        for prompt_style, user_prompt, conversation_history in prompt_variants:
            last_prompt = self._build_prompt_from_truncation(
                query=query,
                options=options,
                retrieval=retrieval,
                truncated=truncated,
                prompt_style=prompt_style,
                user_prompt=user_prompt,
                conversation_history=conversation_history,
            )
            if last_prompt.token_count <= options.max_context_tokens:
                return last_prompt
        assert last_prompt is not None
        clipped_prompt = self.prompt_builder.token_accounting.clip(
            last_prompt.prompt,
            options.max_context_tokens,
        )
        return ContextPromptBuildResult(
            grounded_candidate=last_prompt.grounded_candidate,
            prompt=clipped_prompt,
            token_count=self.prompt_builder.token_accounting.count(clipped_prompt),
        )

    def _build_prompt_from_truncation(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: object,
        truncated: ContextTruncationResult,
        prompt_style: str,
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


@dataclass(slots=True)
class RAGRuntime:
    storage: StorageConfig
    request: AssemblyRequest = field(default_factory=AssemblyRequest)
    assembly_service: CapabilityAssemblyService = field(default_factory=CapabilityAssemblyService, repr=False)
    telemetry_service: TelemetryService | None = None
    vlm_repo: VisualDescriptionRepo | None = None
    capability_bundle: CapabilityBundle = field(init=False, repr=False)
    token_contract: TokenizerContract = field(init=False, repr=False)
    token_accounting: TokenAccountingService = field(init=False, repr=False)
    stores: StorageBundle = field(init=False)
    ingest_pipeline: IngestPipeline = field(init=False, repr=False)
    delete_pipeline: DeletePipeline = field(init=False, repr=False)
    rebuild_pipeline: RebuildPipeline = field(init=False, repr=False)
    retrieval_service: RetrievalService = field(init=False, repr=False)
    query_pipeline: _QueryPipeline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.capability_bundle = self.assembly_service.assemble_request(self.request)
        self.token_contract = self.capability_bundle.token_contract
        self.token_accounting = self.capability_bundle.token_accounting
        self.stores = self.storage.build()
        self._register_or_validate_runtime_contract()
        self._build_pipelines()

    @classmethod
    def from_request(
        cls,
        *,
        storage: StorageConfig,
        request: AssemblyRequest,
        assembly_service: CapabilityAssemblyService | None = None,
        telemetry_service: TelemetryService | None = None,
        vlm_repo: VisualDescriptionRepo | None = None,
    ) -> RAGRuntime:
        return cls(
            storage=storage,
            request=request,
            assembly_service=assembly_service or CapabilityAssemblyService(),
            telemetry_service=telemetry_service,
            vlm_repo=vlm_repo,
        )

    @classmethod
    def from_profile(
        cls,
        *,
        storage: StorageConfig,
        profile_id: str,
        requirements: CapabilityRequirements | None = None,
        assembly_service: CapabilityAssemblyService | None = None,
        telemetry_service: TelemetryService | None = None,
        vlm_repo: VisualDescriptionRepo | None = None,
    ) -> RAGRuntime:
        service = assembly_service or CapabilityAssemblyService()
        request = service.request_for_profile(profile_id, requirements=requirements)
        return cls.from_request(
            storage=storage,
            request=request,
            assembly_service=service,
            telemetry_service=telemetry_service,
            vlm_repo=vlm_repo,
        )

    @property
    def diagnostics(self) -> AssemblyDiagnostics:
        return self.capability_bundle.diagnostics

    @property
    def catalog(self) -> CapabilityCatalog:
        return self.assembly_service.catalog_from_environment(config=self.request.config)

    @property
    def runtime_contract_payload(self) -> dict[str, str | int | bool]:
        return self.capability_bundle.runtime_contract_payload

    @property
    def selected_profile_id(self) -> str | None:
        return self.capability_bundle.selected_profile_id

    def diagnostics_payload(self) -> dict[str, object]:
        return {
            "status": self.diagnostics.status,
            "selected_profile_id": self.selected_profile_id,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                }
                for issue in self.diagnostics.issues
            ],
            "decisions": [
                {
                    "capability": decision.capability,
                    "source": decision.source,
                    "provider_kind": decision.provider_kind,
                    "provider_name": decision.provider_name,
                    "model_name": decision.model_name,
                    "location": decision.location,
                    "reason": decision.reason,
                    "selected": decision.selected,
                }
                for decision in self.diagnostics.decisions
            ],
            "runtime_contract": self.runtime_contract_payload,
        }

    def close(self) -> None:
        self.stores.close()

    def __enter__(self) -> RAGRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def insert(self, request: IngestRequest | None = None, /, **kwargs: Any) -> IngestPipelineResult:
        self._register_or_validate_runtime_contract()
        return self.ingest_pipeline.run(self._coerce_ingest_request(request, **kwargs))

    def insert_many(
        self,
        requests: list[IngestRequest],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        self._register_or_validate_runtime_contract()
        return self.ingest_pipeline.run_many(requests, continue_on_error=continue_on_error)

    def insert_content_list(
        self,
        items: list[DirectContentItem],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        self._register_or_validate_runtime_contract()
        return self.ingest_pipeline.run_content_list(items, continue_on_error=continue_on_error)

    def query(
        self,
        *args: Any,
        options: QueryOptions | None = None,
        **kwargs: Any,
    ) -> RAGQueryResult:
        query_text = self._coerce_query_text(*args, **kwargs)
        self._register_or_validate_runtime_contract()
        return self.query_pipeline.run(query_text, options=self._normalize_query_options(options))

    def delete(self, *args: Any, **kwargs: Any) -> DeletePipelineResult:
        request = self._coerce_delete_request(*args, **kwargs)
        return self.delete_pipeline.run(request)

    def rebuild(self, *args: Any, **kwargs: Any) -> RebuildPipelineResult:
        self._register_or_validate_runtime_contract()
        request = self._coerce_rebuild_request(*args, **kwargs)
        return self.rebuild_pipeline.run(request)

    def upsert_node(self, node: GraphNode, *, evidence_chunk_ids: list[str] | None = None) -> GraphNode:
        self.stores.graph.save_node(node, evidence_chunk_ids=evidence_chunk_ids)
        return node

    def upsert_edge(self, edge: GraphEdge, *, candidate: bool = False) -> GraphEdge:
        if candidate:
            self.stores.graph.save_candidate_edge(edge)
        else:
            self.stores.graph.save_edge(edge)
        return edge

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.stores.graph.get_node(node_id)

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]:
        return self.stores.graph.list_nodes(node_type=node_type)

    def delete_node(self, node_id: str) -> None:
        self.stores.graph.delete_node(node_id)

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None:
        return self.stores.graph.get_edge(edge_id, include_candidates=include_candidates)

    def list_edges(self) -> list[GraphEdge]:
        return self.stores.graph.list_edges()

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None:
        self.stores.graph.delete_edge(edge_id, include_candidates=include_candidates)

    def insert_custom_kg(
        self,
        *,
        nodes: list[GraphNode] | None = None,
        edges: list[GraphEdge] | None = None,
    ) -> dict[str, int]:
        for node in nodes or []:
            self.upsert_node(node)
        for edge in edges or []:
            self.upsert_edge(edge)
        return {
            "node_count": len(nodes or []),
            "edge_count": len(edges or []),
        }

    def _build_pipelines(self) -> None:
        ocr_repo = create_default_ocr_repo()
        self.ingest_pipeline = IngestPipeline(
            documents=self.stores.documents,
            chunks=self.stores.chunks,
            vectors=self.stores.vectors,
            graph=self.stores.graph,
            status=self.stores.status,
            cache=self.stores.cache,
            fts_repo=self.stores.fts_repo,
            object_store=self.stores.object_store,
            markdown_parser=MarkdownParserRepo(),
            pdf_parser=PDFParserRepo(),
            plain_text_parser=PlainTextParserRepo(),
            image_parser=ImageParserRepo(ocr_repo),
            web_parser=WebParserRepo(),
            web_fetch_repo=HttpWebFetchRepo(),
            docling_parser=DoclingParserRepo(ocr_repo, self.vlm_repo),
            policy_resolution_service=PolicyResolutionService(),
            toc_service=TOCService(),
            chunking_service=ChunkingService(token_accounting=self.token_accounting),
            document_processing_service=DocumentProcessingService(
                toc_service=TOCService(),
                token_accounting=self.token_accounting,
                tokenizer_contract=self.token_contract,
            ),
            embedding_capabilities=self.capability_bundle.embedding_bindings,
            chat_capabilities=self.capability_bundle.chat_bindings,
        )
        self.delete_pipeline = DeletePipeline(
            documents=self.stores.documents,
            chunks=self.stores.chunks,
            vectors=self.stores.vectors,
            graph=self.stores.graph,
            status=self.stores.status,
            fts_repo=self.stores.fts_repo,
            ingest_pipeline=self.ingest_pipeline,
        )
        self.rebuild_pipeline = RebuildPipeline(
            ingest_pipeline=self.ingest_pipeline,
            delete_pipeline=self.delete_pipeline,
            object_store=self.stores.object_store,
        )
        self.retrieval_service = self._build_retrieval_service()
        answer_generation_service = AnswerGenerationService()
        self.query_pipeline = _QueryPipeline(
            retrieval=self.retrieval_service,
            context_merger=ContextEvidenceMerger(),
            truncator=EvidenceTruncator(token_accounting=self.token_accounting),
            prompt_builder=ContextPromptBuilder(
                answer_generation_service=answer_generation_service,
                token_accounting=self.token_accounting,
            ),
            answer_generator=AnswerGenerator(
                answer_generation_service=answer_generation_service,
                chat_bindings=self.capability_bundle.chat_bindings,
            ),
        )

    def _build_retrieval_service(self) -> RetrievalService:
        bundle = self.capability_bundle
        query_understanding_service = QueryUnderstandingService(chat_bindings=bundle.chat_bindings)
        retrieval_factory = SearchBackedRetrievalFactory(
            metadata_repo=self.stores.metadata_repo,
            fts_repo=self.stores.fts_repo,
            graph_repo=self.stores.graph_repo,
        )
        reranker_service = (
            ModelBackedRerankService(
                binding=bundle.rerank_bindings[0],
                query_understanding_service=query_understanding_service,
            )
            if bundle.rerank_bindings
            else None
        )
        instrumented_reranker = None if reranker_service is None else _InstrumentedReranker(reranker_service)
        return RetrievalService(
            full_text_retriever=retrieval_factory.full_text_retriever,
            vector_retriever=retrieval_factory.vector_retriever_from_repo(
                self.stores.vector_repo,
                bundle.embedding_bindings,
            ),
            local_retriever=retrieval_factory.local_retriever_from_repo(
                self.stores.vector_repo,
                bundle.embedding_bindings,
            ),
            global_retriever=retrieval_factory.global_retriever_from_repo(
                self.stores.vector_repo,
                bundle.embedding_bindings,
            ),
            section_retriever=retrieval_factory.section_retriever,
            special_retriever=retrieval_factory.special_retriever_from_repo(
                self.stores.vector_repo,
                bundle.embedding_bindings,
            ),
            metadata_retriever=retrieval_factory.metadata_retriever,
            graph_expander=retrieval_factory.graph_expander,
            web_retriever=retrieval_factory.web_retriever,
            reranker=instrumented_reranker,
            routing_service=RoutingService(),
            query_understanding_service=query_understanding_service,
            evidence_service=EvidenceService(),
            graph_expansion_service=GraphExpansionService(),
            artifact_service=ArtifactService(),
            telemetry_service=self.telemetry_service,
        )

    def _register_or_validate_runtime_contract(self) -> None:
        payload = dict(self.capability_bundle.runtime_contract_payload)
        existing = self.stores.cache.get(_RUNTIME_CONTRACT_KEY, namespace=_RUNTIME_CONTRACT_NAMESPACE)
        stored_payload = existing.payload if existing is not None and isinstance(existing.payload, dict) else None
        governance = self.assembly_service.govern_runtime_contract(
            bundle=self.capability_bundle,
            stored_payload=stored_payload,
        )
        if governance.should_persist:
            self.stores.cache.save(
                CacheEntry(
                    namespace=_RUNTIME_CONTRACT_NAMESPACE,
                    cache_key=_RUNTIME_CONTRACT_KEY,
                    payload=payload,
                )
            )
            return
        governance.raise_for_invalid()

    def _normalize_query_options(self, options: QueryOptions | None) -> QueryOptions:
        if options is None:
            return QueryOptions(max_context_tokens=self.token_contract.max_context_tokens)
        if (
            options.max_context_tokens == QueryOptions().max_context_tokens
            and self.token_contract.max_context_tokens != QueryOptions().max_context_tokens
        ):
            return replace(options, max_context_tokens=self.token_contract.max_context_tokens)
        return options

    @staticmethod
    def _coerce_query_text(*args: Any, **kwargs: Any) -> str:
        query_text = kwargs.pop("query", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if args:
            if len(args) != 1:
                raise TypeError("query accepts exactly one positional query string")
            if query_text is not None:
                raise TypeError("query was provided both positionally and by keyword")
            query_text = args[0]
        if not isinstance(query_text, str) or not query_text.strip():
            raise TypeError("query requires a non-empty string")
        return query_text

    @staticmethod
    def _coerce_ingest_request(request: IngestRequest | None = None, /, **kwargs: Any) -> IngestRequest:
        if request is not None:
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"insert request was provided both positionally and by keyword: {unexpected}")
            return request
        normalized_kwargs = {"owner": "user", **kwargs}
        if "file_path" in normalized_kwargs and normalized_kwargs["file_path"] is not None:
            normalized_kwargs["file_path"] = Path(normalized_kwargs["file_path"])
        return IngestRequest(**normalized_kwargs)

    @staticmethod
    def _coerce_request(expected_type: type[Any], action: str, *args: Any, **kwargs: Any) -> Any:
        if args:
            if len(args) != 1 or not isinstance(args[0], expected_type):
                raise TypeError(f"{action} accepts either a {expected_type.__name__} or keyword selectors")
            if kwargs:
                raise TypeError(f"{action} request was provided both positionally and by keyword")
            return args[0]
        return expected_type(**kwargs)

    @staticmethod
    def _coerce_delete_request(*args: Any, **kwargs: Any) -> DeleteRequest:
        return RAGRuntime._coerce_request(DeleteRequest, "delete", *args, **kwargs)

    @staticmethod
    def _coerce_rebuild_request(*args: Any, **kwargs: Any) -> RebuildRequest:
        return RAGRuntime._coerce_request(RebuildRequest, "rebuild", *args, **kwargs)


__all__ = ["RAGRuntime", "_RUNTIME_CONTRACT_KEY", "_RUNTIME_CONTRACT_NAMESPACE"]
