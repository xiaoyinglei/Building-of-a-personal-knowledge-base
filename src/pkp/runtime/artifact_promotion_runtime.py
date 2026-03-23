from __future__ import annotations

from typing import Protocol

from pkp.types.artifact import KnowledgeArtifact


class ArtifactServiceProtocol(Protocol):
    def approve(self, artifact_id: str) -> KnowledgeArtifact: ...


class ArtifactIndexProtocol(Protocol):
    def index_artifact(self, artifact: KnowledgeArtifact) -> None: ...


class ArtifactPromotionRuntime:
    def __init__(
        self,
        artifact_service: ArtifactServiceProtocol,
        artifact_index: ArtifactIndexProtocol,
    ) -> None:
        self._artifact_service = artifact_service
        self._artifact_index = artifact_index

    def approve(self, artifact_id: str) -> KnowledgeArtifact:
        artifact = self._artifact_service.approve(artifact_id)
        self._artifact_index.index_artifact(artifact)
        return artifact
