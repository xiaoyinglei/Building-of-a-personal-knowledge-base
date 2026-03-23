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


def test_artifact_service_builds_topic_page_with_stable_sections() -> None:
    service = ArtifactService()
    reviewed_at = datetime(2026, 2, 3, 4, 5, tzinfo=UTC)
    suggestion = service.suggest_preservation(
        query="Alpha retrieval path",
        runtime_mode=RuntimeMode.DEEP,
        evidence=[
            _evidence("chunk-a", "doc-a"),
            _evidence("chunk-b", "doc-b"),
        ],
    )

    artifact = service.build_artifact(
        query="Alpha retrieval path",
        suggestion=suggestion,
        evidence=[
            _evidence("chunk-a", "doc-a"),
            _evidence("chunk-b", "doc-b"),
        ],
        differences_or_conflicts=[],
        reviewed_at=reviewed_at,
    )

    sections = [
        "## Topic Definition",
        "## Key Conclusions",
        "## Key Evidence",
        "## Boundaries and Failure Cases",
        "## Disagreements",
        "## Related Documents",
        "## Related Concepts and Entities",
        "## Open Questions",
        "## Last Reviewed",
        "## Confidence and Coverage",
    ]
    positions = [artifact.body_markdown.index(section) for section in sections]

    assert artifact.artifact_type is ArtifactType.TOPIC_PAGE
    assert positions == sorted(positions)
    assert "Alpha retrieval path" in artifact.body_markdown
    assert "doc-a" in artifact.body_markdown
    assert "text for chunk-a" in artifact.body_markdown
    assert "2026-02-03T04:05:00+00:00" in artifact.body_markdown
    assert "2 evidence item(s) across 2 document(s)." in artifact.body_markdown


def test_artifact_service_builds_comparison_page_with_stable_sections() -> None:
    service = ArtifactService()
    reviewed_at = datetime(2026, 2, 3, 4, 5, tzinfo=UTC)
    suggestion = service.suggest_preservation(
        query="Compare Alpha and Beta",
        runtime_mode=RuntimeMode.DEEP,
        evidence=[
            _evidence("chunk-a", "doc-a"),
            _evidence("chunk-b", "doc-b"),
        ],
        differences_or_conflicts=["Alpha and Beta diverge on retention"],
    )

    artifact = service.build_artifact(
        query="Compare Alpha and Beta",
        suggestion=suggestion,
        evidence=[
            _evidence("chunk-a", "doc-a"),
            _evidence("chunk-b", "doc-b"),
        ],
        differences_or_conflicts=["Alpha and Beta diverge on retention"],
        reviewed_at=reviewed_at,
    )

    assert artifact.artifact_type is ArtifactType.COMPARISON_PAGE
    assert "## Comparison Definition" in artifact.body_markdown
    assert "## Disagreements" in artifact.body_markdown
    assert "Alpha and Beta diverge on retention" in artifact.body_markdown
    assert "## Confidence and Coverage" in artifact.body_markdown
