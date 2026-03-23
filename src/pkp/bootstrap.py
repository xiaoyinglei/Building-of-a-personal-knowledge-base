from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from pkp.config import AppSettings
from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import OcrResult
from pkp.repo.search.in_memory_vector_repo import InMemoryVectorRepo
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
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
from pkp.service.evidence_service import CandidateLike, EvidenceService
from pkp.service.graph_expansion_service import GraphExpansionService
from pkp.service.ingest_service import IngestService
from pkp.service.retrieval_service import RetrievalService
from pkp.service.routing_service import RoutingService


def load_settings() -> AppSettings:
    return AppSettings()


def build_test_container(root: Path) -> RuntimeContainer:
    root.mkdir(parents=True, exist_ok=True)
    ocr_repo = DeterministicOcrVisionRepo(
        mapping={
            (Path.cwd() / "data/samples/sample-ui.png").as_posix(): OcrResult(
                visible_text="Sample UI Fast Path Deep Path Preserve Artifact",
                visual_semantics="application card with labels for Fast Path and Deep Path",
            )
        }
    )
    ingest_service = IngestService.create_in_memory(root, ocr_repo=ocr_repo)
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
        routing_service=RoutingService(),
        evidence_service=EvidenceService(),
        graph_expansion_service=GraphExpansionService(),
        artifact_service=ArtifactService(),
    )
    retrieval_adapter = RetrievalRuntimeAdapter(retrieval_service)
    evidence_adapter = RuntimeEvidenceAdapter(
        evidence_service=EvidenceService(),
        artifact_service=ArtifactService(),
        metadata_repo=metadata_repo,
        ingest_service=ingest_service,
    )
    session_runtime = SessionRuntime()
    deep_runtime = DeepResearchRuntime(
        routing_service=ResearchPlannerAdapter(),
        retrieval_service=retrieval_adapter,
        evidence_service=evidence_adapter,
        session_runtime=session_runtime,
        max_rounds=4,
    )
    fast_runtime = FastQueryRuntime(
        retrieval_service=retrieval_adapter,
        evidence_service=evidence_adapter,
        deep_runtime=deep_runtime,
    )
    artifact_promotion_runtime = ArtifactPromotionRuntime(
        ArtifactApprovalAdapter(metadata_repo),
        ArtifactIndexerAdapter(ingest_service),
    )
    ingest_runtime = IngestRuntime(ingest_service=ingest_service, base_path=Path.cwd())
    return RuntimeContainer(
        ingest_runtime=ingest_runtime,
        fast_query_runtime=fast_runtime,
        deep_research_runtime=deep_runtime,
        artifact_promotion_runtime=artifact_promotion_runtime,
        session_runtime=session_runtime,
        metadata_repo=metadata_repo,
    )
