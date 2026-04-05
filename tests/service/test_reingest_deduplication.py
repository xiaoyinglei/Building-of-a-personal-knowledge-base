import sqlite3
from pathlib import Path

from tests.support import make_ingest_service


def test_duplicate_ingest_does_not_create_duplicate_active_documents(tmp_path: Path) -> None:
    service = make_ingest_service(tmp_path)

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


def test_duplicate_ingest_restores_missing_vectors_for_existing_chunks(tmp_path: Path) -> None:
    service = make_ingest_service(tmp_path)

    first = service.ingest_markdown(
        location="notes/topic.md",
        markdown="# Topic\n\nAlpha.\n\n## Detail\n\nBeta.",
        owner="user",
    )
    vector_db = sqlite3.connect(tmp_path / "vectors.sqlite3")
    vector_db.execute("DELETE FROM vectors")
    vector_db.commit()

    second = service.ingest_markdown(
        location="notes/topic.md",
        markdown="# Topic\n\nAlpha.\n\n## Detail\n\nBeta.",
        owner="user",
    )
    vector_count = vector_db.execute("SELECT count(*) FROM vectors WHERE item_kind = 'chunk'").fetchone()[0]

    assert second.is_duplicate is True
    assert second.document.doc_id == first.document.doc_id
    assert vector_count == len(first.chunks)


def test_repair_indexes_restores_missing_vectors_without_reingest(tmp_path: Path) -> None:
    service = make_ingest_service(tmp_path)

    first = service.ingest_markdown(
        location="notes/topic.md",
        markdown="# Topic\n\nAlpha.\n\n## Detail\n\nBeta.",
        owner="user",
    )
    vector_db = sqlite3.connect(tmp_path / "vectors.sqlite3")
    vector_db.execute("DELETE FROM vectors")
    vector_db.commit()

    summary = service.repair_indexes()
    vector_count = vector_db.execute("SELECT count(*) FROM vectors WHERE item_kind = 'chunk'").fetchone()[0]

    assert summary["document_count"] == 1
    assert summary["chunk_count"] == len(first.chunks)
    assert summary["repaired_vector_count"] == len(first.chunks)
    assert vector_count == len(first.chunks)
