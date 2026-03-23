from __future__ import annotations

from dataclasses import dataclass, field

from pkp.runtime.adapters import RuntimeEvidenceAdapter
from pkp.runtime.deep_research_runtime import DeepResearchRuntime
from pkp.runtime.session_runtime import SessionRuntime
from pkp.service.artifact_service import ArtifactService
from pkp.service.evidence_service import EvidenceService
from pkp.types import (
    AccessPolicy,
    EvidenceItem,
    ExecutionLocationPreference,
    ExecutionPolicy,
    RuntimeMode,
    TaskType,
)
from pkp.types.query import ComplexityLevel


def make_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        effective_access_policy=AccessPolicy.default(),
        task_type=TaskType.RESEARCH,
        complexity_level=ComplexityLevel.L4_RESEARCH,
        latency_budget=60,
        cost_budget=3.0,
        execution_location_preference=ExecutionLocationPreference.CLOUD_FIRST,
        fallback_allowed=True,
    )


def hit(text: str = "retrieved evidence") -> EvidenceItem:
    return EvidenceItem(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_id="src-1",
        citation_anchor="Section 1",
        text=text,
        score=0.8,
    )


@dataclass
class StubMetadataRepo:
    saved_artifacts: list[object] = field(default_factory=list)

    def save_artifact(self, artifact: object) -> None:
        self.saved_artifacts.append(artifact)


@dataclass
class StubIngestService:
    indexed: list[str] = field(default_factory=list)

    def ingest_plain_text(self, *, location: str, text: str, owner: str, title: str | None = None) -> object:
        self.indexed.append(location)
        return object()


class FailingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def chat(self, prompt: str) -> str:
        raise RuntimeError("provider unavailable")

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        return list(range(len(candidates)))


class EchoProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def chat(self, prompt: str) -> str:
        return f"{self.name}: {prompt.splitlines()[0]}"

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        return list(range(len(candidates)))


@dataclass
class FakeRoutingService:
    def decompose(self, query: str) -> list[str]:
        return [query]

    def expand(self, query: str, evidence_matrix: list[dict[str, object]], round_index: int) -> list[str]:
        return []


@dataclass
class FakeRetrievalService:
    def retrieve(self, query: str, policy: ExecutionPolicy, mode: RuntimeMode, round_index: int) -> list[EvidenceItem]:
        return [hit()]


def test_runtime_evidence_adapter_uses_backup_cloud_provider() -> None:
    adapter = RuntimeEvidenceAdapter(
        evidence_service=EvidenceService(),
        artifact_service=ArtifactService(),
        metadata_repo=StubMetadataRepo(),
        ingest_service=StubIngestService(),
        cloud_providers=[FailingProvider(), EchoProvider("backup-cloud")],
        local_providers=[EchoProvider("local")],
    )

    matrix = adapter.build_evidence_matrix([hit()])
    response = adapter.build_deep_response("Explain Alpha", matrix, location="cloud")

    assert response.conclusion.startswith("backup-cloud:")


def test_deep_research_runtime_falls_back_from_cloud_to_local_provider() -> None:
    adapter = RuntimeEvidenceAdapter(
        evidence_service=EvidenceService(),
        artifact_service=ArtifactService(),
        metadata_repo=StubMetadataRepo(),
        ingest_service=StubIngestService(),
        cloud_providers=[FailingProvider(), FailingProvider()],
        local_providers=[EchoProvider("local")],
    )
    runtime = DeepResearchRuntime(
        routing_service=FakeRoutingService(),
        retrieval_service=FakeRetrievalService(),
        evidence_service=adapter,
        session_runtime=SessionRuntime(),
        max_rounds=2,
    )

    response = runtime.run("research", make_policy(), session_id="fallback-to-local")

    assert response.conclusion.startswith("local:")
    assert response.runtime_mode is RuntimeMode.DEEP
