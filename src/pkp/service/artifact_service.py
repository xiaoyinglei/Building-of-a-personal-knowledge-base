from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pkp.types.access import RuntimeMode
from pkp.types.artifact import ArtifactStatus, ArtifactType, KnowledgeArtifact
from pkp.types.envelope import EvidenceItem, PreservationSuggestion


class ArtifactService:
    @staticmethod
    def _unique_docs(evidence: Sequence[EvidenceItem]) -> set[str]:
        return {item.doc_id for item in evidence}

    def suggest_preservation(
        self,
        *,
        query: str,
        runtime_mode: RuntimeMode,
        evidence: Sequence[EvidenceItem],
        differences_or_conflicts: Sequence[str] | None = None,
    ) -> PreservationSuggestion:
        doc_count = len(self._unique_docs(evidence))
        conflict_count = len(differences_or_conflicts or ())
        reusable = runtime_mode is RuntimeMode.DEEP and (doc_count >= 2 or conflict_count > 0 or len(evidence) >= 4)
        if not reusable:
            return PreservationSuggestion(suggested=False)

        lowered = query.lower()
        if "compare" in lowered or conflict_count > 0:
            artifact_type = ArtifactType.COMPARISON_PAGE.value
        elif "timeline" in lowered or "trend" in lowered:
            artifact_type = ArtifactType.TIMELINE.value
        else:
            artifact_type = ArtifactType.TOPIC_PAGE.value

        title = query.strip().rstrip("?") or "Reusable knowledge"
        rationale = "Evidence spans multiple documents and is likely reusable."
        if conflict_count > 0:
            rationale = "Evidence captures a stable comparison or conflict map."

        return PreservationSuggestion(
            suggested=True,
            artifact_type=artifact_type,
            title=title,
            rationale=rationale,
        )

    def apply_lifecycle(
        self,
        *,
        proposed: KnowledgeArtifact,
        existing_artifacts: Sequence[KnowledgeArtifact],
    ) -> list[KnowledgeArtifact]:
        updated_existing: list[KnowledgeArtifact] = []
        proposed_chunks = set(proposed.supported_chunk_ids)
        for artifact in existing_artifacts:
            status = artifact.status
            if artifact.artifact_type is proposed.artifact_type and artifact.title == proposed.title:
                existing_chunks = set(artifact.supported_chunk_ids)
                overlap = bool(existing_chunks & proposed_chunks)
                if overlap and artifact.body_markdown != proposed.body_markdown:
                    status = ArtifactStatus.CONFLICTED
                elif proposed_chunks > existing_chunks:
                    status = ArtifactStatus.STALE
            updated_existing.append(artifact.model_copy(update={"status": status}))

        return [*updated_existing, proposed]

    @staticmethod
    def build_timestamp() -> datetime:
        return datetime.now(UTC)
