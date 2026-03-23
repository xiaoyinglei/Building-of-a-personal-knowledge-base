from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import httpx

from pkp.config import AppSettings
from pkp.config.policies import RoutingThresholds
from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import OcrResult
from pkp.repo.models.fallback_embedding_repo import FallbackEmbeddingRepo
from pkp.repo.models.ollama_provider_repo import OllamaProviderRepo
from pkp.repo.models.openai_provider_repo import OpenAIProviderRepo
from pkp.repo.parse.image_parser_repo import ImageParserRepo
from pkp.repo.parse.markdown_parser_repo import MarkdownParserRepo
from pkp.repo.parse.pdf_parser_repo import PDFParserRepo
from pkp.repo.parse.plain_text_parser_repo import PlainTextParserRepo
from pkp.repo.parse.web_fetch_repo import WebFetchRepo as HttpWebFetchRepo
from pkp.repo.parse.web_parser_repo import WebParserRepo
from pkp.repo.search.in_memory_vector_repo import InMemoryVectorRepo
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.storage.file_object_store import FileObjectStore
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.repo.vision.ocr_vision_repo import DeterministicOcrVisionRepo
from pkp.runtime.adapters import (
    ArtifactApprovalAdapter,
    ArtifactIndexerAdapter,
    ResearchPlannerAdapter,
    RetrievalRuntimeAdapter,
    RuntimeEvidenceAdapter,
    SearchBackedRetrievalFactory,
)
from pkp.runtime.artifact_promotion_runtime import ArtifactPromotionRuntime
from pkp.runtime.container import RuntimeContainer
from pkp.runtime.deep_research_runtime import DeepResearchRuntime
from pkp.runtime.fast_query_runtime import FastQueryRuntime
from pkp.runtime.ingest_runtime import IngestRuntime
from pkp.runtime.session_runtime import SessionRuntime
from pkp.service.artifact_service import ArtifactService
from pkp.service.chunking_service import ChunkingService
from pkp.service.evidence_service import CandidateLike, EvidenceService
from pkp.service.graph_expansion_service import GraphExpansionService
from pkp.service.ingest_service import IngestService
from pkp.service.policy_resolution_service import PolicyResolutionService
from pkp.service.retrieval_service import RetrievalService
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


class _DeterministicChatProvider:
    def chat(self, prompt: str) -> str:
        query = next((line.strip() for line in prompt.splitlines() if line.strip()), "summary")
        return f"Deterministic synthesis for: {query}"


def _build_ingest_service(
    runtime_root: Path,
    object_store_root: Path,
    *,
    ocr_repo: DeterministicOcrVisionRepo,
    web_fetch_repo: HttpWebFetchRepo,
) -> IngestService:
    runtime_root.mkdir(parents=True, exist_ok=True)
    object_store_root.mkdir(parents=True, exist_ok=True)
    return IngestService(
        metadata_repo=SQLiteMetadataRepo(runtime_root / "metadata.sqlite3"),
        fts_repo=SQLiteFTSRepo(runtime_root / "fts.sqlite3"),
        vector_repo=InMemoryVectorRepo(),
        graph_repo=SQLiteGraphRepo(runtime_root / "graph.sqlite3"),
        object_store=FileObjectStore(object_store_root),
        markdown_parser=MarkdownParserRepo(),
        pdf_parser=PDFParserRepo(),
        plain_text_parser=PlainTextParserRepo(),
        image_parser=ImageParserRepo(ocr_repo),
        web_parser=WebParserRepo(),
        web_fetch_repo=web_fetch_repo,
        policy_resolution_service=PolicyResolutionService(),
        toc_service=TOCService(),
        chunking_service=ChunkingService(),
        embedding_repo=FallbackEmbeddingRepo(),
    )


def _build_runtime_container(
    *,
    runtime_root: Path,
    object_store_root: Path,
    max_retrieval_rounds: int,
    thresholds: RoutingThresholds,
    ocr_repo: DeterministicOcrVisionRepo,
    web_fetch_repo: HttpWebFetchRepo,
    cloud_providers: Sequence[object] = (),
    local_providers: Sequence[object] = (),
) -> RuntimeContainer:
    ingest_service = _build_ingest_service(
        runtime_root,
        object_store_root,
        ocr_repo=ocr_repo,
        web_fetch_repo=web_fetch_repo,
    )
    telemetry_service = TelemetryService.create_jsonl(runtime_root / "telemetry" / "events.jsonl")
    metadata_repo: SQLiteMetadataRepo = ingest_service.metadata_repo
    fts_repo: SQLiteFTSRepo = ingest_service.fts_repo
    vector_repo: InMemoryVectorRepo = ingest_service.vector_repo
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
        retrieval_factory.vector_retriever_from_repo(vector_repo),
    )
    section_retriever = cast(
        Callable[[str, list[str]], Sequence[CandidateLike]],
        retrieval_factory.section_retriever,
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
    retrieval_service = RetrievalService(
        full_text_retriever=standard_retriever,
        vector_retriever=vector_retriever,
        section_retriever=section_retriever,
        graph_expander=graph_expander,
        web_retriever=web_retriever,
        reranker=lambda _query, candidates: sorted(
            candidates,
            key=lambda candidate: (-candidate.score, candidate.chunk_id),
        ),
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
    deep_runtime = DeepResearchRuntime(
        routing_service=ResearchPlannerAdapter(),
        retrieval_service=retrieval_adapter,
        evidence_service=evidence_adapter,
        session_runtime=session_runtime,
        max_rounds=max_retrieval_rounds,
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
    return RuntimeContainer(
        ingest_runtime=ingest_runtime,
        fast_query_runtime=fast_runtime,
        deep_research_runtime=deep_runtime,
        artifact_promotion_runtime=artifact_promotion_runtime,
        session_runtime=session_runtime,
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
    return _build_runtime_container(
        runtime_root=settings.runtime.data_dir,
        object_store_root=settings.runtime.object_store_dir,
        max_retrieval_rounds=settings.runtime.max_retrieval_rounds,
        thresholds=thresholds,
        ocr_repo=_sample_ocr_repo(),
        web_fetch_repo=HttpWebFetchRepo(),
        cloud_providers=(
            OpenAIProviderRepo(
                api_key=settings.openai.api_key.get_secret_value(),
                base_url=settings.openai.base_url,
                model=settings.openai.model,
                embedding_model=settings.openai.embedding_model,
            ),
        ),
        local_providers=(
            OllamaProviderRepo(
                base_url=settings.ollama.base_url,
                chat_model=settings.ollama.chat_model,
                embedding_model=settings.ollama.embedding_model,
            ),
        ),
    )


def build_test_container(root: Path) -> RuntimeContainer:
    return _build_runtime_container(
        runtime_root=root,
        object_store_root=root / "objects",
        max_retrieval_rounds=4,
        thresholds=RoutingThresholds(),
        ocr_repo=_sample_ocr_repo(),
        web_fetch_repo=_sample_web_fetch_repo(),
        cloud_providers=(),
        local_providers=(_DeterministicChatProvider(),),
    )
