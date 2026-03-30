from __future__ import annotations

from typing import Protocol

from pkp.utils._telemetry import TelemetryService
from pkp.schema._types.artifact import KnowledgeArtifact


class ArtifactServiceProtocol(Protocol):
    def list_artifacts(self) -> list[KnowledgeArtifact]: ...

    def get_artifact(self, artifact_id: str) -> KnowledgeArtifact | None: ...

    def approve(self, artifact_id: str) -> KnowledgeArtifact: ...


class ArtifactIndexProtocol(Protocol):
    def index_artifact(self, artifact: KnowledgeArtifact) -> None: ...


class ArtifactPromotionRuntime:
    def __init__(
        self,
        artifact_service: ArtifactServiceProtocol,
        artifact_index: ArtifactIndexProtocol,
        telemetry_service: TelemetryService | None = None,
    ) -> None:
        self._artifact_service = artifact_service
        self._artifact_index = artifact_index
        self._telemetry_service = telemetry_service

    def list_artifacts(self) -> list[KnowledgeArtifact]:
        return self._artifact_service.list_artifacts()

    def get_artifact(self, artifact_id: str) -> KnowledgeArtifact:
        artifact = self._artifact_service.get_artifact(artifact_id)
        if artifact is None:
            raise ValueError(f"Unknown artifact: {artifact_id}")
        return artifact

    def approve(self, artifact_id: str) -> KnowledgeArtifact:
        artifact = self._artifact_service.approve(artifact_id)
        self._artifact_index.index_artifact(artifact)
        if self._telemetry_service is not None:
            self._telemetry_service.record_artifact_approved(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type.value,
            )
        return artifact
