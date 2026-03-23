from pathlib import Path

from pkp.service.ingest_service import IngestService


def test_markdown_ingest_builds_toc_paths_and_stable_anchors(tmp_path: Path) -> None:
    service = IngestService.create_in_memory(tmp_path)

    first = service.ingest_markdown(
        location="notes/topic.md",
        markdown=("# Topic\n\nIntro.\n\n## Section One\n\nAlpha.\n\n### Detail\n\nBeta.\n"),
        owner="user",
    )
    second = service.ingest_markdown(
        location="notes/topic.md",
        markdown=("# Topic\n\nIntro updated.\n\n## Section One\n\nAlpha updated.\n\n### Detail\n\nBeta updated.\n"),
        owner="user",
    )

    assert first.segments[1].toc_path == ["Topic", "Section One"]
    assert first.segments[1].anchor == second.segments[1].anchor
    assert first.chunks[0].citation_anchor.startswith("notes/topic.md#")
