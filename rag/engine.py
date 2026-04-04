from __future__ import annotations

import os
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
from rag.llm._providers.local_bge_provider_repo import LocalBgeProviderRepo
from rag.llm.embedding import EmbeddingProviderBinding
from rag.llm.generation import AnswerGenerationService
from rag.llm.rerank import ModelBackedRerankService
from rag.query._artifact.service import ArtifactService
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
from rag.query.routing import RoutingService
from rag.query.understanding import QueryUnderstandingService
from rag.schema._types.diagnostics import ProviderAttempt
from rag.schema._types.storage import CacheEntry
from rag.schema._types.text import (
    DEFAULT_TOKENIZER_FALLBACK_MODEL,
    TokenAccountingService,
    TokenizerContract,
    load_env_file,
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
class RAG:
    storage: StorageConfig
    embedding_bindings: tuple[EmbeddingProviderBinding, ...] = ()
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
        resolved_embedding_model = self._resolve_embedding_model_name(self.embedding_bindings)
        self.token_contract = TokenizerContract.from_env(
            embedding_model_name=resolved_embedding_model,
            default_context_tokens=QueryOptions().max_context_tokens,
        )
        self.token_accounting = TokenAccountingService(self.token_contract)
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
            embedding_bindings=self.embedding_bindings,
        )
        self.embedding_bindings = self.ingest_pipeline.embedding_bindings
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
                provider_bindings=self.embedding_bindings,
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
        retrieval_factory = SearchBackedRetrievalFactory(
            metadata_repo=self.stores.metadata_repo,
            fts_repo=self._fts_repo,
            graph_repo=self.stores.graph_repo,
        )
        reranker_service = self._build_reranker_service()
        instrumented_reranker = None if reranker_service is None else _CoreInstrumentedReranker(reranker_service)
        return RetrievalService(
            full_text_retriever=retrieval_factory.full_text_retriever,
            vector_retriever=retrieval_factory.vector_retriever_from_repo(
                self.stores.vector_repo,
                self.embedding_bindings,
            ),
            local_retriever=retrieval_factory.local_retriever_from_repo(
                self.stores.vector_repo,
                self.embedding_bindings,
            ),
            global_retriever=retrieval_factory.global_retriever_from_repo(
                self.stores.vector_repo,
                self.embedding_bindings,
            ),
            section_retriever=retrieval_factory.section_retriever,
            special_retriever=retrieval_factory.special_retriever_from_repo(
                self.stores.vector_repo,
                self.embedding_bindings,
            ),
            metadata_retriever=retrieval_factory.metadata_retriever,
            graph_expander=retrieval_factory.graph_expander,
            web_retriever=retrieval_factory.web_retriever,
            reranker=instrumented_reranker,
            routing_service=RoutingService(),
            query_understanding_service=QueryUnderstandingService(),
            evidence_service=EvidenceService(),
            graph_expansion_service=GraphExpansionService(),
            artifact_service=ArtifactService(),
            telemetry_service=self.telemetry_service,
        )

    def _build_reranker_service(self) -> object | None:
        provider = self._default_rerank_provider()
        if provider is not None:
            return ModelBackedRerankService(provider=provider)
        env_rerank_provider = self._env_rerank_provider()
        if env_rerank_provider is not None:
            return ModelBackedRerankService(provider=env_rerank_provider)
        return None

    def _default_rerank_provider(self) -> object | None:
        return next(
            (
                binding.provider
                for binding in self.embedding_bindings
                if callable(getattr(binding.provider, "rerank", None))
                and bool(getattr(binding.provider, "is_rerank_configured", True))
            ),
            None,
        )

    @staticmethod
    def _env_rerank_provider() -> object | None:
        rerank_model = os.environ.get("RAG_RERANK_MODEL")
        rerank_model_path = os.environ.get("RAG_RERANK_MODEL_PATH")
        if not rerank_model and not rerank_model_path:
            return None
        return LocalBgeProviderRepo(
            rerank_model=rerank_model or "BAAI/bge-reranker-v2-m3",
            rerank_model_path=rerank_model_path,
        )

    def _register_or_validate_runtime_contract(self) -> None:
        payload = self._runtime_contract_payload()
        existing = self.stores.cache.get(_RUNTIME_CONTRACT_KEY, namespace=_RUNTIME_CONTRACT_NAMESPACE)
        if existing is None or not isinstance(existing.payload, dict):
            self.stores.cache.save(
                CacheEntry(
                    namespace=_RUNTIME_CONTRACT_NAMESPACE,
                    cache_key=_RUNTIME_CONTRACT_KEY,
                    payload=payload,
                )
            )
            return
        mismatches = [
            key
            for key in (
                "embedding_model_name",
                "tokenizer_model_name",
                "chunking_tokenizer_model_name",
                "tokenizer_backend",
                "chunk_token_size",
                "chunk_overlap_tokens",
            )
            if existing.payload.get(key) != payload.get(key)
        ]
        if mismatches:
            details = ", ".join(
                f"{key}: current={payload.get(key)!r} stored={existing.payload.get(key)!r}"
                for key in mismatches
            )
            raise RuntimeError(
                "RAG runtime contract does not match the existing index. "
                f"Mismatched fields: {details}. Use the same embedding/tokenizer contract or rebuild the index."
            )

    def _runtime_contract_payload(self) -> dict[str, str | int | bool]:
        tokenizer_backend, _tokenizer_model = self.token_accounting.backend_descriptor()
        return {
            "embedding_model_name": self.token_contract.embedding_model_name,
            "tokenizer_model_name": self.token_contract.tokenizer_model_name,
            "chunking_tokenizer_model_name": self.token_contract.chunking_tokenizer_model_name,
            "tokenizer_backend": tokenizer_backend,
            "chunk_token_size": self.token_contract.chunk_token_size,
            "chunk_overlap_tokens": self.token_contract.normalized_chunk_overlap_tokens(),
        }

    @staticmethod
    def _resolve_embedding_model_name(bindings: tuple[EmbeddingProviderBinding, ...]) -> str:
        load_env_file()
        configured = next(
            (
                str(model_name)
                for binding in bindings
                for model_name in [getattr(binding.provider, "embedding_model_name", None)]
                if isinstance(model_name, str) and model_name.strip()
            ),
            "",
        )
        env_locked_model = os.environ.get("RAG_EMBEDDING_MODEL") or os.environ.get("RAG_INDEX_EMBEDDING_MODEL")
        if env_locked_model and not configured:
            raise RuntimeError(
                "RAG_EMBEDDING_MODEL is set, but no embedding-capable provider "
                "with embedding_model_name was configured."
            )
        contract = TokenizerContract.from_env(
            embedding_model_name=configured or DEFAULT_TOKENIZER_FALLBACK_MODEL
        )
        if configured and contract.embedding_model_name and contract.embedding_model_name != configured:
            raise RuntimeError(
                "Configured embedding model does not match RAG_EMBEDDING_MODEL/RAG_TOKENIZER contract: "
                f"{configured!r} != {contract.embedding_model_name!r}"
            )
        if configured:
            return configured
        return contract.embedding_model_name or DEFAULT_TOKENIZER_FALLBACK_MODEL

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


__all__ = ["RAG"]
