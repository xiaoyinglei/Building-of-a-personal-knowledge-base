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


def test_sqlite_vector_repo_scopes_vectors_by_embedding_space(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.sqlite3"
    repo = SQLiteVectorRepo(db_path)
    repo.upsert(
        "chunk-a",
        [1.0, 0.0],
        metadata={"doc_id": "doc-a", "segment_id": "seg-a", "text": "alpha evidence"},
        embedding_space="cloud::embed",
    )
    repo.upsert(
        "chunk-a",
        [0.0, 1.0],
        metadata={"doc_id": "doc-a", "segment_id": "seg-a", "text": "alpha evidence"},
        embedding_space="local::embed",
    )

    cloud_results = repo.search([1.0, 0.0], limit=1, embedding_space="cloud::embed")
    local_results = repo.search([0.0, 1.0], limit=1, embedding_space="local::embed")

    assert [item.chunk_id for item in cloud_results] == ["chunk-a"]
    assert [item.chunk_id for item in local_results] == ["chunk-a"]


def test_sqlite_vector_repo_separates_chunk_entity_and_relation_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.sqlite3"
    repo = SQLiteVectorRepo(db_path)
    repo.upsert(
        "chunk-a",
        [1.0, 0.0],
        metadata={"doc_id": "doc-a", "segment_id": "seg-a", "text": "chunk evidence"},
        item_kind="chunk",
    )
    repo.upsert(
        "entity-a",
        [0.0, 1.0],
        metadata={"doc_id": "doc-a", "segment_id": "seg-a", "text": "entity alpha"},
        item_kind="entity",
    )
    repo.upsert(
        "relation-a",
        [0.6, 0.4],
        metadata={"doc_id": "doc-a", "segment_id": "seg-a", "text": "supports relation"},
        item_kind="relation",
    )

    chunk_results = repo.search([1.0, 0.0], limit=1, item_kind="chunk")
    entity_results = repo.search([0.0, 1.0], limit=1, item_kind="entity")
    relation_results = repo.search([0.6, 0.4], limit=1, item_kind="relation")

    assert [item.item_id for item in chunk_results] == ["chunk-a"]
    assert [item.item_id for item in entity_results] == ["entity-a"]
    assert [item.item_id for item in relation_results] == ["relation-a"]
    assert repo.count_vectors(item_kind="chunk") == 1
    assert repo.count_vectors(item_kind="entity") == 1
    assert repo.count_vectors(item_kind="relation") == 1


def test_sqlite_vector_repo_scopes_aggregated_vectors_by_doc_ids_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.sqlite3"
    repo = SQLiteVectorRepo(db_path)
    repo.upsert(
        "entity-alpha",
        [1.0, 0.0],
        metadata={
            "doc_id": "doc-b",
            "doc_ids": "doc-a,doc-b",
            "segment_id": "",
            "text": "Alpha Engine",
        },
        item_kind="entity",
    )

    results = repo.search([1.0, 0.0], limit=1, doc_ids=["doc-a"], item_kind="entity")

    assert [item.item_id for item in results] == ["entity-alpha"]
