from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import httpx

from pkp.config import AppSettings
from pkp.config.policies import RoutingThresholds
from pkp.core.rag_core import RAGCore
from pkp.core.storage_config import StorageConfig
from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import EmbeddingProviderBinding, OcrResult, OcrVisionRepo, VectorRepo
from pkp.repo.models.fallback_embedding_repo import FallbackEmbeddingRepo
from pkp.repo.models.local_bge_provider_repo import LocalBgeProviderRepo
from pkp.repo.models.ollama_provider_repo import OllamaProviderRepo
from pkp.repo.models.openai_provider_repo import OpenAIProviderRepo
from pkp.repo.parse.docling_parser_repo import DoclingParserRepo
from pkp.repo.parse.image_parser_repo import ImageParserRepo
from pkp.repo.parse.markdown_parser_repo import MarkdownParserRepo
from pkp.repo.parse.pdf_parser_repo import PDFParserRepo
from pkp.repo.parse.plain_text_parser_repo import PlainTextParserRepo
from pkp.repo.parse.web_fetch_repo import WebFetchRepo as HttpWebFetchRepo
from pkp.repo.parse.web_parser_repo import WebParserRepo
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.search.sqlite_vector_repo import SQLiteVectorRepo
from pkp.repo.storage.file_object_store import FileObjectStore
from pkp.repo.storage.sqlite_memory_repo import SQLiteMemoryRepo
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.repo.vision.ocr_vision_repo import DeterministicOcrVisionRepo, create_default_ocr_repo
from pkp.rerank.cross_encoder import CrossEncoderConfig
from pkp.rerank.pipeline import RerankPipelineConfig
from pkp.runtime.adapters import (
    ArtifactApprovalAdapter,
    ArtifactIndexerAdapter,
    InstrumentedReranker,
    ResearchPlannerAdapter,
    RetrievalRuntimeAdapter,
    RuntimeEvidenceAdapter,
    SearchBackedRetrievalFactory,
)
from pkp.runtime.artifact_promotion_runtime import ArtifactPromotionRuntime
from pkp.runtime.container import RuntimeContainer
from pkp.runtime.deep_research_runtime import DeepResearchRuntime
from pkp.runtime.diagnostics_runtime import DiagnosticsRuntime, ProviderBinding
from pkp.runtime.fast_query_runtime import FastQueryRuntime
from pkp.runtime.ingest_runtime import IngestRuntime
from pkp.runtime.provider_metadata import capability_configured, embedding_space
from pkp.runtime.session_runtime import SessionRuntime
from pkp.service.artifact_service import ArtifactService
from pkp.service.chunking_service import ChunkingService
from pkp.service.document_processing_service import DocumentProcessingService
from pkp.service.evidence_service import CandidateLike, EvidenceService
from pkp.service.graph_expansion_service import GraphExpansionService
from pkp.service.ingest_service import IngestService
from pkp.service.memory_service import MemoryService
from pkp.service.policy_resolution_service import PolicyResolutionService
from pkp.service.rerank_service import HeuristicRerankService
from pkp.service.retrieval_service import Reranker, RetrievalService
from pkp.service.routing_service import RoutingService
from pkp.service.telemetry_service import TelemetryService
from pkp.service.toc_service import TOCService


def load_settings() -> AppSettings:
    return AppSettings()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sample_ocr_repo() -> DeterministicOcrVisionRepo:
    sample_image = _project_root() / "data/samples/sample-ui.png"
    return DeterministicOcrVisionRepo(
        mapping={
            sample_image.as_posix(): OcrResult(
                visible_text="Sample UI Fast Path Deep Path Preserve Artifact",
                visual_semantics="application card with labels for Fast Path and Deep Path",
            )
        }
    )


def _sample_web_fetch_repo() -> HttpWebFetchRepo:
    sample_html = (_project_root() / "data/samples/sample-article.html").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            content=sample_html.encode("utf-8"),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)
    return HttpWebFetchRepo(http_client=httpx.Client(transport=transport))


class _DeterministicModelProvider:
    def __init__(self) -> None:
        self._fallback = FallbackEmbeddingRepo()
        self.provider_name = "deterministic"
        self.chat_model_name = "deterministic-chat"
        self.embedding_model_name = "deterministic-embed"
        self.is_chat_configured = True
        self.is_embed_configured = True
        self.is_rerank_configured = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._fallback.embed(texts)

    def chat(self, prompt: str) -> str:
        query = next((line.strip() for line in prompt.splitlines() if line.strip()), "summary")
        return f"Deterministic synthesis for: {query}"

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        del query
        return list(range(len(candidates)))


@dataclass(frozen=True, slots=True)
class _ProviderStack:
    cloud_providers: tuple[object, ...]
    local_providers: tuple[object, ...]
    rerank_service: object


def _embedding_bindings(
    *,
    local_providers: Sequence[object],
    cloud_providers: Sequence[object],
) -> tuple[EmbeddingProviderBinding, ...]:
    bindings: list[EmbeddingProviderBinding] = []
    for provider in cloud_providers:
        if capability_configured(provider, "embed"):
            bindings.append(
                EmbeddingProviderBinding(
                    provider=provider,
                    space=embedding_space(provider),
                    location="cloud",
                )
            )
    for provider in local_providers:
        if capability_configured(provider, "embed"):
            bindings.append(
                EmbeddingProviderBinding(
                    provider=provider,
                    space=embedding_space(provider),
                    location="local",
                )
            )
    if bindings:
        return tuple(bindings)
    return (EmbeddingProviderBinding(provider=FallbackEmbeddingRepo(), space="default", location="runtime"),)


def _resolve_provider_stack(settings: AppSettings) -> _ProviderStack:
    local_bge_provider: object | None = None
    if settings.local_bge.enabled:
        local_bge_provider = LocalBgeProviderRepo(
            embedding_model=settings.local_bge.embedding_model,
            embedding_model_path=(
                None
                if settings.local_bge.embedding_model_path is None
                else settings.local_bge.embedding_model_path.as_posix()
            ),
            rerank_model=settings.local_bge.rerank_model,
            rerank_model_path=(
                None
                if settings.local_bge.rerank_model_path is None
                else settings.local_bge.rerank_model_path.as_posix()
            ),
        )

    rerank_service = (
        HeuristicRerankService(
            provider=local_bge_provider,
            config=RerankPipelineConfig(
                cross_encoder=CrossEncoderConfig(
                    model_name=settings.local_bge.rerank_model,
                    model_path=(
                        None
                        if settings.local_bge.rerank_model_path is None
                        else settings.local_bge.rerank_model_path.as_posix()
                    ),
                )
            ),
        )
        if settings.local_bge.enabled
        else HeuristicRerankService()
    )

    cloud_providers = (
        OpenAIProviderRepo(
            api_key=settings.openai.api_key.get_secret_value(),
            base_url=settings.openai.base_url,
            model=settings.openai.model,
            embedding_model=settings.openai.embedding_model,
        ),
    )
    local_providers = tuple(
        provider
        for provider in (
            local_bge_provider,
            OllamaProviderRepo(
                base_url=settings.ollama.base_url,
                chat_model=settings.ollama.chat_model,
                embedding_model=None if settings.local_bge.enabled else settings.ollama.embedding_model,
            ),
        )
        if provider is not None
    )
    return _ProviderStack(
        cloud_providers=cloud_providers,
        local_providers=local_providers,
        rerank_service=rerank_service,
    )


def _build_ingest_service(
    runtime_root: Path,
    object_store_root: Path,
    *,
    ocr_repo: OcrVisionRepo,
    web_fetch_repo: HttpWebFetchRepo,
    embedding_bindings: tuple[EmbeddingProviderBinding, ...],
) -> IngestService:
    runtime_root.mkdir(parents=True, exist_ok=True)
    object_store_root.mkdir(parents=True, exist_ok=True)
    return IngestService(
        metadata_repo=SQLiteMetadataRepo(runtime_root / "metadata.sqlite3"),
        fts_repo=SQLiteFTSRepo(runtime_root / "fts.sqlite3"),
        vector_repo=cast(VectorRepo, SQLiteVectorRepo(runtime_root / "vectors.sqlite3")),
        graph_repo=SQLiteGraphRepo(runtime_root / "graph.sqlite3"),
        object_store=FileObjectStore(object_store_root),
        markdown_parser=MarkdownParserRepo(),
        pdf_parser=PDFParserRepo(),
        plain_text_parser=PlainTextParserRepo(),
        image_parser=ImageParserRepo(ocr_repo),
        web_parser=WebParserRepo(),
        web_fetch_repo=web_fetch_repo,
        docling_parser=DoclingParserRepo(ocr_repo),
        policy_resolution_service=PolicyResolutionService(),
        toc_service=TOCService(),
        chunking_service=ChunkingService(),
        document_processing_service=DocumentProcessingService(toc_service=TOCService()),
        embedding_bindings=embedding_bindings,
    )


def _build_runtime_container(
    *,
    runtime_root: Path,
    object_store_root: Path,
    max_retrieval_rounds: int,
    max_recursive_depth: int,
    thresholds: RoutingThresholds,
    ocr_repo: OcrVisionRepo,
    web_fetch_repo: HttpWebFetchRepo,
    cloud_providers: Sequence[object] = (),
    local_providers: Sequence[object] = (),
    rerank_service: object | None = None,
) -> RuntimeContainer:
    embedding_bindings = _embedding_bindings(
        local_providers=local_providers,
        cloud_providers=cloud_providers,
    )
    ingest_service = _build_ingest_service(
        runtime_root,
        object_store_root,
        ocr_repo=ocr_repo,
        web_fetch_repo=web_fetch_repo,
        embedding_bindings=embedding_bindings,
    )
    ingest_service.repair_indexes()
    telemetry_service = TelemetryService.create_jsonl(runtime_root / "telemetry" / "events.jsonl")
    metadata_repo: SQLiteMetadataRepo = ingest_service.metadata_repo
    fts_repo: SQLiteFTSRepo = ingest_service.fts_repo
    vector_repo = cast(SQLiteVectorRepo, ingest_service.vector_repo)
    graph_repo: SQLiteGraphRepo = ingest_service.graph_repo

    retrieval_factory = SearchBackedRetrievalFactory(
        metadata_repo=metadata_repo,
        fts_repo=fts_repo,
        graph_repo=graph_repo,
    )
    standard_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.full_text_retriever,
    )
    vector_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.vector_retriever_from_repo(
            vector_repo,
            embedding_bindings,
        ),
    )
    local_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.local_retriever_from_repo(
            vector_repo,
            embedding_bindings,
        ),
    )
    global_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.global_retriever_from_repo(
            vector_repo,
            embedding_bindings,
        ),
    )
    section_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.section_retriever,
    )
    special_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.special_retriever,
    )
    metadata_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.metadata_retriever,
    )
    graph_expander = cast(
        Callable[[str, list[str], list[CandidateLike]], Sequence[CandidateLike]],
        retrieval_factory.graph_expander,
    )
    web_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.web_retriever,
    )
    routing_service = RoutingService(thresholds)
    evidence_service = EvidenceService(thresholds)
    resolved_rerank_service = rerank_service or HeuristicRerankService()
    instrumented_reranker = InstrumentedReranker(resolved_rerank_service)
    retrieval_service = RetrievalService(
        full_text_retriever=standard_retriever,
        vector_retriever=vector_retriever,
        local_retriever=local_retriever,
        global_retriever=global_retriever,
        section_retriever=section_retriever,
        special_retriever=special_retriever,
        metadata_retriever=metadata_retriever,
        graph_expander=graph_expander,
        web_retriever=web_retriever,
        reranker=cast(Reranker, instrumented_reranker),
        routing_service=routing_service,
        evidence_service=evidence_service,
        graph_expansion_service=GraphExpansionService(),
        artifact_service=ArtifactService(),
        telemetry_service=telemetry_service,
        thresholds=thresholds,
    )
    retrieval_adapter = RetrievalRuntimeAdapter(retrieval_service)
    evidence_adapter = RuntimeEvidenceAdapter(
        evidence_service=evidence_service,
        artifact_service=ArtifactService(),
        metadata_repo=metadata_repo,
        ingest_service=ingest_service,
        telemetry_service=telemetry_service,
        cloud_providers=cloud_providers,
        local_providers=local_providers,
    )
    session_runtime = SessionRuntime()
    memory_service = MemoryService(SQLiteMemoryRepo(runtime_root / "memory.sqlite3"))
    deep_runtime = DeepResearchRuntime(
        routing_service=ResearchPlannerAdapter(),
        retrieval_service=retrieval_adapter,
        evidence_service=evidence_adapter,
        session_runtime=session_runtime,
        memory_service=memory_service,
        max_rounds=max_retrieval_rounds,
        max_recursive_depth=max_recursive_depth,
    )
    fast_runtime = FastQueryRuntime(
        retrieval_service=retrieval_adapter,
        evidence_service=evidence_adapter,
        deep_runtime=deep_runtime,
        telemetry_service=telemetry_service,
    )
    artifact_promotion_runtime = ArtifactPromotionRuntime(
        ArtifactApprovalAdapter(metadata_repo),
        ArtifactIndexerAdapter(ingest_service),
        telemetry_service=telemetry_service,
    )
    ingest_runtime = IngestRuntime(ingest_service=ingest_service, base_path=_project_root())
    diagnostics_runtime = DiagnosticsRuntime(
        providers=[
            *[ProviderBinding(provider=provider, location="cloud") for provider in cloud_providers],
            *[ProviderBinding(provider=provider, location="local") for provider in local_providers],
            ProviderBinding(provider=instrumented_reranker, location="runtime"),
        ],
        metadata_repo=metadata_repo,
        vector_repo=vector_repo,
    )
    return RuntimeContainer(
        ingest_runtime=ingest_runtime,
        fast_query_runtime=fast_runtime,
        deep_research_runtime=deep_runtime,
        artifact_promotion_runtime=artifact_promotion_runtime,
        session_runtime=session_runtime,
        diagnostics_runtime=diagnostics_runtime,
        metadata_repo=metadata_repo,
        telemetry_service=telemetry_service,
    )


def build_runtime_container(settings: AppSettings) -> RuntimeContainer:
    thresholds = RoutingThresholds(
        fast_min_evidence_chunks=settings.runtime.fast_min_evidence_chunks,
        fast_min_sections=1,
        deep_min_evidence_chunks=settings.runtime.deep_min_evidence_chunks,
        deep_min_supporting_units=2,
        max_retrieval_rounds=settings.runtime.max_retrieval_rounds,
        max_recursive_depth=settings.runtime.max_recursive_depth,
        default_wall_clock_budget_seconds=settings.runtime.default_wall_clock_budget_seconds,
        default_synthesis_retry_count=settings.runtime.default_synthesis_retry_count,
    )
    provider_stack = _resolve_provider_stack(settings)
    return _build_runtime_container(
        runtime_root=settings.runtime.data_dir,
        object_store_root=settings.runtime.object_store_dir,
        max_retrieval_rounds=settings.runtime.max_retrieval_rounds,
        max_recursive_depth=settings.runtime.max_recursive_depth,
        thresholds=thresholds,
        ocr_repo=create_default_ocr_repo(),
        web_fetch_repo=HttpWebFetchRepo(),
        cloud_providers=provider_stack.cloud_providers,
        local_providers=provider_stack.local_providers,
        rerank_service=provider_stack.rerank_service,
    )


def build_rag_core(settings: AppSettings) -> RAGCore:
    provider_stack = _resolve_provider_stack(settings)
    return RAGCore(
        storage=StorageConfig(backend="sqlite", root=settings.runtime.data_dir),
        embedding_bindings=_embedding_bindings(
            local_providers=provider_stack.local_providers,
            cloud_providers=provider_stack.cloud_providers,
        ),
    )


def build_test_container(root: Path) -> RuntimeContainer:
    return _build_runtime_container(
        runtime_root=root,
        object_store_root=root / "objects",
        max_retrieval_rounds=4,
        max_recursive_depth=2,
        thresholds=RoutingThresholds(),
        ocr_repo=_sample_ocr_repo(),
        web_fetch_repo=_sample_web_fetch_repo(),
        cloud_providers=(),
        local_providers=(_DeterministicModelProvider(),),
    )
