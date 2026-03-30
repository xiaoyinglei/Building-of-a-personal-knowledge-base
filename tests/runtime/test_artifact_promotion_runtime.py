from dataclasses import dataclass, field
from datetime import UTC, datetime

from pkp.interfaces._runtime.artifact_promotion_runtime import ArtifactPromotionRuntime
from pkp.utils._telemetry import TelemetryService
from pkp.schema._types.artifact import ArtifactStatus, ArtifactType, KnowledgeArtifact


@dataclass
class FakeArtifactService:
    approvals: list[str] = field(default_factory=list)

    def list_artifacts(self) -> list[KnowledgeArtifact]:
        return [
            KnowledgeArtifact(
                artifact_id="artifact-1",
                artifact_type=ArtifactType.TOPIC_PAGE,
                title="Artifact 1",
                supported_chunk_ids=["chunk-1"],
                confidence=0.9,
                status=ArtifactStatus.SUGGESTED,
                last_reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
                body_markdown="# Artifact 1",
                source_scope=["doc-1"],
            )
        ]

    def get_artifact(self, artifact_id: str) -> KnowledgeArtifact | None:
        if artifact_id != "artifact-1":
            return None
        return self.list_artifacts()[0]

    def approve(self, artifact_id: str) -> KnowledgeArtifact:
        self.approvals.append(artifact_id)
        return KnowledgeArtifact(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.TOPIC_PAGE,
            title="Artifact",
            supported_chunk_ids=["chunk-1"],
            confidence=0.9,
            status=ArtifactStatus.APPROVED,
            last_reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
            body_markdown="# Artifact",
            source_scope=["doc-1"],
        )


@dataclass
class FakeArtifactIndex:
    indexed_ids: list[str] = field(default_factory=list)

    def index_artifact(self, artifact: KnowledgeArtifact) -> None:
        self.indexed_ids.append(artifact.artifact_id)


def test_artifact_promotion_runtime_records_approval_event() -> None:
    telemetry = TelemetryService.create_in_memory()
    artifact_service = FakeArtifactService()
    artifact_index = FakeArtifactIndex()
    runtime = ArtifactPromotionRuntime(
        artifact_service=artifact_service,
        artifact_index=artifact_index,
        telemetry_service=telemetry,
    )

    artifact = runtime.approve("artifact-123")

    assert artifact.artifact_id == "artifact-123"
    assert artifact_service.approvals == ["artifact-123"]
    assert artifact_index.indexed_ids == ["artifact-123"]
    events = telemetry.list_events()
    assert [event.name for event in events] == ["artifact.approved"]
    assert events[0].payload["artifact_id"] == "artifact-123"


def test_artifact_promotion_runtime_lists_and_shows_artifacts() -> None:
    runtime = ArtifactPromotionRuntime(
        artifact_service=FakeArtifactService(),
        artifact_index=FakeArtifactIndex(),
    )

    artifacts = runtime.list_artifacts()
    artifact = runtime.get_artifact("artifact-1")

    assert len(artifacts) == 1
    assert artifacts[0].artifact_id == "artifact-1"
    assert artifact.artifact_id == "artifact-1"
