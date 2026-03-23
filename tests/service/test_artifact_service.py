from __future__ import annotations

from datetime import UTC, datetime

from pkp.service.artifact_service import ArtifactService
from pkp.types.access import RuntimeMode
from pkp.types.artifact import ArtifactStatus, ArtifactType, KnowledgeArtifact
from pkp.types.envelope import EvidenceItem


def _evidence(chunk_id: str, doc_id: str, evidence_kind: str = "internal") -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        citation_anchor=f"#{chunk_id}",
        text=f"text for {chunk_id}",
        score=0.9,
        evidence_kind=evidence_kind,
    )


def _artifact(
    *,
    artifact_id: str,
    title: str,
    supported_chunk_ids: list[str],
    body_markdown: str,
    status: ArtifactStatus,
) -> KnowledgeArtifact:
    return KnowledgeArtifact(
        artifact_id=artifact_id,
        artifact_type=ArtifactType.TOPIC_PAGE,
        title=title,
        supported_chunk_ids=supported_chunk_ids,
        confidence=0.8,
        status=status,
        last_reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
        body_markdown=body_markdown,
        source_scope=["doc-a"],
    )


def test_artifact_service_suggests_preservation_for_reusable_multi_doc_evidence() -> None:
    service = ArtifactService()

    suggestion = service.suggest_preservation(
        query="Compare Alpha and Beta",
        runtime_mode=RuntimeMode.DEEP,
        evidence=[
            _evidence("chunk-a", "doc-a"),
            _evidence("chunk-b", "doc-b"),
        ],
        differences_or_conflicts=["Alpha and Beta diverge on retention"],
    )

    assert suggestion.suggested is True
    assert suggestion.artifact_type == ArtifactType.COMPARISON_PAGE.value


def test_artifact_service_marks_superseded_artifacts_stale_instead_of_overwriting() -> None:
    service = ArtifactService()
    existing = _artifact(
        artifact_id="artifact-old",
        title="Alpha topic",
        supported_chunk_ids=["chunk-a"],
        body_markdown="old summary",
        status=ArtifactStatus.APPROVED,
    )
    proposed = _artifact(
        artifact_id="artifact-new",
        title="Alpha topic",
        supported_chunk_ids=["chunk-a", "chunk-b"],
        body_markdown="old summary",
        status=ArtifactStatus.SUGGESTED,
    )

    updated = service.apply_lifecycle(proposed=proposed, existing_artifacts=[existing])

    assert updated[0].status is ArtifactStatus.STALE
    assert updated[1].status is ArtifactStatus.SUGGESTED


def test_artifact_service_marks_overlapping_conflicts_as_conflicted() -> None:
    service = ArtifactService()
    existing = _artifact(
        artifact_id="artifact-old",
        title="Alpha topic",
        supported_chunk_ids=["chunk-a"],
        body_markdown="old summary",
        status=ArtifactStatus.APPROVED,
    )
    proposed = _artifact(
        artifact_id="artifact-new",
        title="Alpha topic",
        supported_chunk_ids=["chunk-a"],
        body_markdown="new conflicting summary",
        status=ArtifactStatus.SUGGESTED,
    )

    updated = service.apply_lifecycle(proposed=proposed, existing_artifacts=[existing])

    assert updated[0].status is ArtifactStatus.CONFLICTED
    assert updated[1].status is ArtifactStatus.SUGGESTED
