from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pkp.algorithms.generation.answer_generator import AnswerGenerator
from pkp.algorithms.retrieval.search_backed_factory import SearchBackedRetrievalFactory
from pkp.document.loader import HttpWebFetchRepo
from pkp.document.parser import (
    DoclingParserRepo,
    ImageParserRepo,
    MarkdownParserRepo,
    PDFParserRepo,
    PlainTextParserRepo,
    WebParserRepo,
)
from pkp.ingest.ingest import (
    DeletePipeline,
    DeletePipelineResult,
    DeleteRequest,
    IngestPipeline,
    IngestPipelineResult,
    IngestRequest,
    RebuildPipeline,
    RebuildPipelineResult,
    RebuildRequest,
)
from pkp.llm.embedding import EmbeddingProviderBinding
from pkp.llm.generation import AnswerGenerationService
from pkp.llm.rerank import HeuristicRerankService
from pkp.query.context import ContextEvidenceMerger, ContextPromptBuilder, EvidenceTruncator
from pkp.query.graph import GraphExpansionService
from pkp.query.query import QueryOptions, RAGQueryResult
from pkp.query.retrieve import RAGQueryPipeline, RetrievalService
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.storage.file_object_store import FileObjectStore
from pkp.repo.vision.ocr_vision_repo import create_default_ocr_repo
from pkp.service.artifact_service import ArtifactService
from pkp.service.chunking_service import ChunkingService
from pkp.service.document_processing_service import DocumentProcessingService
from pkp.service.evidence_service import CandidateLike, EvidenceService
from pkp.service.policy_resolution_service import PolicyResolutionService
from pkp.service.query_understanding_service import QueryUnderstandingService
from pkp.service.routing_service import RoutingService
from pkp.service.telemetry_service import TelemetryService
from pkp.service.toc_service import TOCService
from pkp.storage import StorageBundle, StorageConfig
from pkp.types.diagnostics import ProviderAttempt


class _CoreInstrumentedReranker:
    def __init__(self, rerank_service: object) -> None:
        self._rerank_service = rerank_service
        self.provider_name = getattr(rerank_service, "provider_name", "heuristic")
        self.rerank_model_name = getattr(rerank_service, "rerank_model_name", "heuristic")
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
class RAGCore:
    storage: StorageConfig
    embedding_bindings: tuple[EmbeddingProviderBinding, ...] = ()
    telemetry_service: TelemetryService | None = None
    stores: StorageBundle = field(init=False)
    ingest_pipeline: IngestPipeline = field(init=False)
    delete_pipeline: DeletePipeline = field(init=False, repr=False)
    rebuild_pipeline: RebuildPipeline = field(init=False, repr=False)
    retrieval_service: RetrievalService = field(init=False, repr=False)
    query_pipeline: object = field(init=False, repr=False)
    _fts_repo: SQLiteFTSRepo = field(init=False, repr=False)
    _object_store: FileObjectStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.stores = self.storage.build()
        ocr_repo = create_default_ocr_repo()
        self._fts_repo = SQLiteFTSRepo(self.stores.root / "fts.sqlite3")
        self._object_store = FileObjectStore(self.stores.root / "objects")
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
            docling_parser=DoclingParserRepo(ocr_repo),
            policy_resolution_service=PolicyResolutionService(),
            toc_service=TOCService(),
            chunking_service=ChunkingService(),
            document_processing_service=DocumentProcessingService(toc_service=TOCService()),
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
            truncator=EvidenceTruncator(),
            prompt_builder=ContextPromptBuilder(
                answer_generation_service=answer_generation_service,
            ),
            answer_generator=AnswerGenerator(
                answer_generation_service=answer_generation_service,
                provider_bindings=self.embedding_bindings,
            ),
        )

    def insert(self, request: IngestRequest | None = None, /, **kwargs: Any) -> IngestPipelineResult:
        if request is None:
            normalized_kwargs = {"owner": "user", **kwargs}
            if "file_path" in normalized_kwargs and normalized_kwargs["file_path"] is not None:
                normalized_kwargs["file_path"] = Path(normalized_kwargs["file_path"])
            request = IngestRequest(**normalized_kwargs)
        return self.ingest_pipeline.run(request)

    def query(
        self,
        *args: Any,
        options: QueryOptions | None = None,
        **kwargs: Any,
    ) -> RAGQueryResult:
        query_text = self._coerce_query_text(*args, **kwargs)
        return self.query_pipeline.run(query_text, options=options or QueryOptions())

    def delete(self, *args: Any, **kwargs: Any) -> DeletePipelineResult:
        request = self._coerce_delete_request(*args, **kwargs)
        return self.delete_pipeline.run(request)

    def rebuild(self, *args: Any, **kwargs: Any) -> RebuildPipelineResult:
        request = self._coerce_rebuild_request(*args, **kwargs)
        return self.rebuild_pipeline.run(request)

    def _build_retrieval_service(self) -> RetrievalService:
        retrieval_factory = SearchBackedRetrievalFactory(
            metadata_repo=self.stores.metadata_repo,
            fts_repo=self._fts_repo,
            graph_repo=self.stores.graph_repo,
        )
        instrumented_reranker = _CoreInstrumentedReranker(
            HeuristicRerankService(provider=self._default_rerank_provider()),
        )
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

    def _default_rerank_provider(self) -> object | None:
        return next(
            (
                binding.provider
                for binding in self.embedding_bindings
                if callable(getattr(binding.provider, "rerank", None))
            ),
            None,
        )

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


__all__ = ["RAGCore"]
