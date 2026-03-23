from pathlib import Path

from pkp.service.ingest_service import IngestService


def test_duplicate_ingest_does_not_create_duplicate_active_documents(tmp_path: Path) -> None:
    service = IngestService.create_in_memory(tmp_path)

    first = service.ingest_markdown(
        location="notes/topic.md",
        markdown="# Topic\n\nAlpha.",
        owner="user",
    )
    second = service.ingest_markdown(
        location="notes/topic.md",
        markdown="# Topic\n\nAlpha.",
        owner="user",
    )

    assert second.is_duplicate is True
    assert second.document.doc_id == first.document.doc_id
    assert len(service.metadata_repo.list_documents(active_only=True)) == 1
