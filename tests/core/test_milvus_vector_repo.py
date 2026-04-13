from __future__ import annotations

import sys
import types

import pytest

from rag.storage.search_backends.milvus_vector_repo import MilvusVectorRepo


def test_milvus_vector_repo_creates_missing_database(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str, str | None]] = []

    class _Connections:
        def connect(self, *, alias: str, uri: str, token: str = "", db_name: str = "default", **kwargs) -> None:
            events.append(("connect", alias, db_name))

        def disconnect(self, alias: str) -> None:
            events.append(("disconnect", alias, None))

    class _Db:
        def list_database(self, *, using: str = "default", timeout=None) -> list[str]:
            events.append(("list_database", using, None))
            return ["default"]

        def create_database(self, db_name: str, *, using: str = "default", timeout=None, **kwargs) -> None:
            events.append(("create_database", using, db_name))

    fake_pymilvus = types.SimpleNamespace(connections=_Connections(), db=_Db())
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    repo = MilvusVectorRepo(
        "http://127.0.0.1:19530",
        db_name="rag_benchmarks",
        collection_prefix="medical_retrieval_mini",
    )
    try:
        assert ("list_database", f"{repo._alias}_bootstrap", None) in events
        assert ("create_database", f"{repo._alias}_bootstrap", "rag_benchmarks") in events
        assert ("connect", repo._alias, "rag_benchmarks") in events
    finally:
        repo.close()


def test_milvus_vector_repo_batches_upserts_before_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MilvusVectorRepo, "_connect", lambda self: setattr(self, "_connected", True))

    class _Connections:
        def disconnect(self, alias: str) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pymilvus", types.SimpleNamespace(connections=_Connections()))

    class _FakeCollection:
        name = "rag_vectors__chunk__default"

        def __init__(self) -> None:
            self.upsert_calls: list[list[dict[str, object]]] = []
            self.flush_count = 0

        def upsert(self, rows: list[dict[str, object]]) -> None:
            self.upsert_calls.append([dict(row) for row in rows])

        def flush(self) -> None:
            self.flush_count += 1

        def release(self) -> None:
            return None

    repo = MilvusVectorRepo("http://127.0.0.1:19530")
    fake_collection = _FakeCollection()
    monkeypatch.setattr(repo, "_collection", lambda **kwargs: fake_collection)
    monkeypatch.setattr(repo, "_UPSERT_BUFFER_SIZE", 2)
    repo._collections[fake_collection.name] = fake_collection
    try:
        repo.upsert("chunk-1", [0.1, 0.2], metadata={"doc_id": "d1"})
        assert fake_collection.upsert_calls == []

        repo.upsert("chunk-2", [0.3, 0.4], metadata={"doc_id": "d2"})
        assert len(fake_collection.upsert_calls) == 1
        assert [row["item_id"] for row in fake_collection.upsert_calls[0]] == ["chunk-1", "chunk-2"]

        repo._flush_dirty_collections()
        assert fake_collection.flush_count == 1
    finally:
        repo.close()
