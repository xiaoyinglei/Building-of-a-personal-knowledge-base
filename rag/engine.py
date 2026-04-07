from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rag.document._vision.ocr_vision_repo import create_default_ocr_repo
from rag.document.loader import HttpWebFetchRepo
from rag.document.parser import (
    DoclingParserRepo,
    ImageParserRepo,
    MarkdownParserRepo,
    PDFParserRepo,
    PlainTextParserRepo,
    WebParserRepo,
)
from rag.ingest._policy.service import PolicyResolutionService
from rag.ingest.chunk import ChunkingService, DocumentProcessingService, TOCService
from rag.ingest.ingest import (
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
from rag.llm._generation.answer_generator import AnswerGenerator
from rag.llm.assembly import CapabilityAssemblyService, CapabilityBundle
from rag.llm.generation import AnswerGenerationService
from rag.llm.rerank import ModelBackedRerankService
from rag.query.analysis import QueryUnderstandingService, RoutingService
from rag.query.artifact import ArtifactService
from rag.query.context import (
    CandidateLike,
    ContextEvidenceMerger,
    ContextPromptBuilder,
    EvidenceService,
    EvidenceTruncator,
)
from rag.query.graph import GraphExpansionService, SearchBackedRetrievalFactory
from rag.query.query import QueryOptions, RAGQueryResult
from rag.query.retrieve import RAGQueryPipeline, RetrievalService
from rag.schema._types.diagnostics import ProviderAttempt
from rag.schema._types.storage import CacheEntry
from rag.schema._types.text import (
    TokenAccountingService,
    TokenizerContract,
)
from rag.schema.graph import GraphEdge, GraphNode
from rag.storage import StorageBundle, StorageConfig
from rag.utils._contracts import FullTextSearchRepo, ObjectStore, VisualDescriptionRepo
from rag.utils._telemetry import TelemetryService

_RUNTIME_CONTRACT_NAMESPACE = "rag_runtime"
_RUNTIME_CONTRACT_KEY = "core_contract_v1"


class _CoreInstrumentedReranker:
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
class _CoreRAG:
    """Internal core implementation.

    Construct through `RAGRuntime` so assembly remains the only model
    decision and contract-governance entrypoint.
    """

    storage: StorageConfig
    assembly_service: CapabilityAssemblyService = field(repr=False)
    capability_bundle: CapabilityBundle = field(repr=False)
    telemetry_service: TelemetryService | None = None
    vlm_repo: VisualDescriptionRepo | None = None
    token_contract: TokenizerContract = field(init=False, repr=False)
    token_accounting: TokenAccountingService = field(init=False, repr=False)
    stores: StorageBundle = field(init=False)
    ingest_pipeline: IngestPipeline = field(init=False)
    delete_pipeline: DeletePipeline = field(init=False, repr=False)
    rebuild_pipeline: RebuildPipeline = field(init=False, repr=False)
    retrieval_service: RetrievalService = field(init=False, repr=False)
    query_pipeline: RAGQueryPipeline = field(init=False, repr=False)
    _fts_repo: FullTextSearchRepo = field(init=False, repr=False)
    _object_store: ObjectStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.token_contract = self.capability_bundle.token_contract
        self.token_accounting = self.capability_bundle.token_accounting
        self.stores = self.storage.build()
        self._register_or_validate_runtime_contract()
        ocr_repo = create_default_ocr_repo()
        self._fts_repo = self.stores.fts_repo
        self._object_store = self.stores.object_store
        self.ingest_pipeline = IngestPipeline(
            documents=self.stores.documents,
            chunks=self.stores.chunks,
            vectors=self.stores.vectors,
            graph=self.stores.graph,
            status=self.stores.status,
            cache=self.stores.cache,
            fts_repo=self._fts_repo,
            object_store=self._object_store,
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
            fts_repo=self._fts_repo,
            ingest_pipeline=self.ingest_pipeline,
        )
        self.rebuild_pipeline = RebuildPipeline(
            ingest_pipeline=self.ingest_pipeline,
            delete_pipeline=self.delete_pipeline,
            object_store=self._object_store,
        )
        self.retrieval_service = self._build_retrieval_service()
        answer_generation_service = AnswerGenerationService()
        self.query_pipeline = RAGQueryPipeline(
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

    def _build_retrieval_service(self) -> RetrievalService:
        bundle = self.capability_bundle
        query_understanding_service = QueryUnderstandingService(chat_bindings=bundle.chat_bindings)
        retrieval_factory = SearchBackedRetrievalFactory(
            metadata_repo=self.stores.metadata_repo,
            fts_repo=self._fts_repo,
            graph_repo=self.stores.graph_repo,
        )
        reranker_service = self._build_reranker_service(query_understanding_service=query_understanding_service)
        instrumented_reranker = None if reranker_service is None else _CoreInstrumentedReranker(reranker_service)
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

    def _build_reranker_service(
        self,
        *,
        query_understanding_service: QueryUnderstandingService,
    ) -> object | None:
        if not self.capability_bundle.rerank_bindings:
            return None
        binding = self.capability_bundle.rerank_bindings[0]
        if binding is not None:
            return ModelBackedRerankService(
                binding=binding,
                query_understanding_service=query_understanding_service,
            )
        return None

    def _register_or_validate_runtime_contract(self) -> None:
        payload = self._runtime_contract_payload()
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

    def _runtime_contract_payload(self) -> dict[str, str | int | bool]:
        return dict(self.capability_bundle.runtime_contract_payload)

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
    def _coerce_delete_request(*args: Any, **kwargs: Any) -> DeleteRequest:
        if args:
            if len(args) != 1 or not isinstance(args[0], DeleteRequest):
                raise TypeError("delete accepts either a DeleteRequest or keyword selectors")
            if kwargs:
                raise TypeError("delete request was provided both positionally and by keyword")
            return args[0]
        return DeleteRequest(**kwargs)

    @staticmethod
    def _coerce_rebuild_request(*args: Any, **kwargs: Any) -> RebuildRequest:
        if args:
            if len(args) != 1 or not isinstance(args[0], RebuildRequest):
                raise TypeError("rebuild accepts either a RebuildRequest or keyword selectors")
            if kwargs:
                raise TypeError("rebuild request was provided both positionally and by keyword")
            return args[0]
        return RebuildRequest(**kwargs)


__all__ = ["_CoreRAG"]
