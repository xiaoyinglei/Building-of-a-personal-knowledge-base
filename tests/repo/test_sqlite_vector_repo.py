from pathlib import Path

from pkp.repo.search.sqlite_vector_repo import SQLiteVectorRepo


def test_sqlite_vector_repo_persists_vectors_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.sqlite3"
    repo = SQLiteVectorRepo(db_path)
    repo.upsert(
        "chunk-a",
        [1.0, 0.0],
        metadata={"doc_id": "doc-a", "segment_id": "seg-a", "text": "alpha evidence"},
    )
    repo.upsert(
        "chunk-b",
        [0.0, 1.0],
        metadata={"doc_id": "doc-b", "segment_id": "seg-b", "text": "beta evidence"},
    )

    reloaded = SQLiteVectorRepo(db_path)
    results = reloaded.search([0.9, 0.1], limit=2, doc_ids=["doc-a", "doc-b"])

    assert [item.chunk_id for item in results] == ["chunk-a", "chunk-b"]
    assert results[0].score > results[1].score
