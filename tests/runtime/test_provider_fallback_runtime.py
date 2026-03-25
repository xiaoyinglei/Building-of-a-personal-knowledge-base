from __future__ import annotations

from dataclasses import dataclass, field

from pkp.runtime.adapters import RuntimeEvidenceAdapter
from pkp.runtime.deep_research_runtime import DeepResearchRuntime
from pkp.runtime.session_runtime import SessionRuntime
from pkp.service.artifact_service import ArtifactService
from pkp.service.evidence_service import EvidenceService
from pkp.service.telemetry_service import TelemetryService
from pkp.types import (
    AccessPolicy,
    ArtifactStatus,
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


def multi_doc_hit(chunk_id: str, doc_id: str, text: str = "Grounded repeated evidence.") -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_id=f"src-{doc_id}",
        citation_anchor=f"{doc_id}#1",
        text=text,
        score=0.8,
    )


@dataclass
class StubMetadataRepo:
    saved_artifacts: list[object] = field(default_factory=list)

    def save_artifact(self, artifact: object) -> None:
        self.saved_artifacts.append(artifact)

    def list_artifacts(self) -> list[object]:
        return list(self.saved_artifacts)


class DeduplicatingMetadataRepo(StubMetadataRepo):
    def save_artifact(self, artifact: object) -> None:
        artifact_id = getattr(artifact, "artifact_id", None)
        if artifact_id is None:
            super().save_artifact(artifact)
            return
        for index, existing in enumerate(self.saved_artifacts):
            if getattr(existing, "artifact_id", None) == artifact_id:
                self.saved_artifacts[index] = artifact
                return
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


class HallucinatingProvider(EchoProvider):
    def chat(self, prompt: str) -> str:
        del prompt
        return "This unsupported conclusion is not grounded in the cited evidence."


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
    telemetry = TelemetryService.create_in_memory()
    adapter = RuntimeEvidenceAdapter(
        evidence_service=EvidenceService(),
        artifact_service=ArtifactService(),
        metadata_repo=StubMetadataRepo(),
        ingest_service=StubIngestService(),
        cloud_providers=[FailingProvider(), FailingProvider()],
        local_providers=[EchoProvider("local")],
        telemetry_service=telemetry,
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
    assert [event.name for event in telemetry.list_events()] == ["runtime.local_fallback"]
    assert telemetry.list_events()[0].payload["to_location"] == "local"


def test_deep_research_runtime_returns_retrieval_only_when_no_evidence_exists() -> None:
    telemetry = TelemetryService.create_in_memory()
    adapter = RuntimeEvidenceAdapter(
        evidence_service=EvidenceService(),
        artifact_service=ArtifactService(),
        metadata_repo=StubMetadataRepo(),
        ingest_service=StubIngestService(),
        cloud_providers=[FailingProvider()],
        local_providers=[FailingProvider()],
        telemetry_service=telemetry,
    )

    @dataclass
    class EmptyRetrievalService:
        def retrieve(
            self,
            query: str,
            policy: ExecutionPolicy,
            mode: RuntimeMode,
            round_index: int,
        ) -> list[EvidenceItem]:
            return []

    runtime = DeepResearchRuntime(
        routing_service=FakeRoutingService(),
        retrieval_service=EmptyRetrievalService(),
        evidence_service=adapter,
        session_runtime=SessionRuntime(),
        max_rounds=2,
    )

    response = runtime.run("research", make_policy(), session_id="no-evidence")

    assert response.runtime_mode is RuntimeMode.DEEP
    assert response.evidence == []
    assert response.uncertainty == "high"


def test_runtime_evidence_adapter_prefers_explanatory_sentence_over_command_example() -> None:
    adapter = RuntimeEvidenceAdapter(
        evidence_service=EvidenceService(),
        artifact_service=ArtifactService(),
        metadata_repo=StubMetadataRepo(),
        ingest_service=StubIngestService(),
        local_providers=[EchoProvider("local")],
    )

    response = adapter.build_fast_response(
        "这个项目做什么？",
        [
            hit('uv run pkp query --mode fast --query "这个项目做什么？"'),
            hit("一个以可靠性为优先的个人知识平台，用来完成资料接入、索引构建、检索问答、深度研究和知识沉淀。"),
        ],
    )

    assert response.conclusion.startswith("一个以可靠性为优先的个人知识平台")


def test_runtime_evidence_adapter_downgrades_unsupported_deep_synthesis_to_retrieval_only() -> None:
    telemetry = TelemetryService.create_in_memory()
    adapter = RuntimeEvidenceAdapter(
        evidence_service=EvidenceService(),
        artifact_service=ArtifactService(),
        metadata_repo=StubMetadataRepo(),
        ingest_service=StubIngestService(),
        local_providers=[HallucinatingProvider("hallucinating")],
        telemetry_service=telemetry,
    )

    matrix = adapter.build_evidence_matrix([hit("Grounded retrieved evidence.")])
    response = adapter.build_deep_response("Explain Alpha", matrix, location="local")

    assert response.runtime_mode is RuntimeMode.DEEP
    assert response.uncertainty == "high"
    assert response.conclusion == "Grounded retrieved evidence."
    assert [event.name for event in telemetry.list_events()] == ["runtime.claim_citation_failed"]


def test_runtime_evidence_adapter_applies_artifact_lifecycle_before_save() -> None:
    metadata_repo = DeduplicatingMetadataRepo()
    adapter = RuntimeEvidenceAdapter(
        evidence_service=EvidenceService(),
        artifact_service=ArtifactService(),
        metadata_repo=metadata_repo,
        ingest_service=StubIngestService(),
        local_providers=[EchoProvider("local")],
    )

    first_matrix = adapter.build_evidence_matrix(
        [
            multi_doc_hit("chunk-a1", "doc-a"),
            multi_doc_hit("chunk-a2", "doc-a"),
            multi_doc_hit("chunk-a3", "doc-a"),
            multi_doc_hit("chunk-a4", "doc-a"),
        ]
    )
    first = adapter.build_deep_response("reliability architecture", first_matrix, location="local")
    second_matrix = adapter.build_evidence_matrix(
        [
            multi_doc_hit("chunk-a1", "doc-a"),
            multi_doc_hit("chunk-a2", "doc-a"),
            multi_doc_hit("chunk-a3", "doc-a"),
            multi_doc_hit("chunk-a4", "doc-a"),
            multi_doc_hit("chunk-b1", "doc-b"),
            multi_doc_hit("chunk-b2", "doc-b"),
            multi_doc_hit("chunk-b3", "doc-b"),
            multi_doc_hit("chunk-b4", "doc-b"),
        ]
    )
    second = adapter.build_deep_response("reliability architecture", second_matrix, location="local")

    artifacts_by_id = {artifact.artifact_id: artifact for artifact in metadata_repo.list_artifacts()}
    first_artifact = artifacts_by_id[first.preservation_suggestion.artifact_id]
    second_artifact = artifacts_by_id[second.preservation_suggestion.artifact_id]

    assert first.preservation_suggestion.artifact_id != second.preservation_suggestion.artifact_id
    assert first_artifact.status is ArtifactStatus.CONFLICTED
    assert second_artifact.status is ArtifactStatus.SUGGESTED
