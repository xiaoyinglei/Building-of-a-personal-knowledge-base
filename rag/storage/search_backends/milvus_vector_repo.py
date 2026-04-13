from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, cast

from rag.schema.runtime import StoredVectorEntry, VectorSearchResult


class MilvusVectorRepo:
    _UPSERT_BUFFER_SIZE = 128

    def __init__(
        self,
        uri: str,
        *,
        token: str | None = None,
        db_name: str | None = None,
        collection_prefix: str = "rag_vectors",
    ) -> None:
        self._uri = uri
        self._token = token
        self._db_name = db_name
        self._collection_prefix = collection_prefix
        self._alias = f"rag_milvus_{abs(hash((uri, collection_prefix))) % 1000000}"
        self._connected = False
        self._collections: dict[str, Any] = {}
        self._dirty_collections: set[str] = set()
        self._pending_upserts: dict[str, list[dict[str, Any]]] = {}
        self._connect()

    def upsert(
        self,
        item_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> None:
        payload = dict(metadata or {})
        vector_values = [float(value) for value in vector]
        collection = self._collection(
            item_kind=item_kind,
            embedding_space=embedding_space,
            dimension=len(vector_values),
        )
        row = {
            "item_id": item_id,
            "doc_id": payload.get("doc_id", ""),
            "source_id": payload.get("source_id", ""),
            "segment_id": payload.get("segment_id", ""),
            "text": payload.get("text", ""),
            "doc_ids": payload.get("doc_ids", ""),
            "source_ids": payload.get("source_ids", ""),
            "metadata_json": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            "embedding": vector_values,
        }
        pending = self._pending_upserts.setdefault(collection.name, [])
        pending.append(row)
        if len(pending) >= self._UPSERT_BUFFER_SIZE:
            self._drain_pending_collection(collection.name)

    def search(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> list[VectorSearchResult]:
        query_vector = [float(value) for value in query]
        if not query_vector:
            return []
        self._flush_dirty_collections()
        if not self._has_collection(item_kind=item_kind, embedding_space=embedding_space):
            return []
        collection = self._collection(item_kind=item_kind, embedding_space=embedding_space)
        raw_limit = max(limit, limit * 8 if doc_ids else limit)
        results = cast(
            list[Any],
            cast(Any, collection).search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {}},
                limit=raw_limit,
                output_fields=[
                    "item_id",
                    "doc_id",
                    "source_id",
                    "segment_id",
                    "text",
                    "doc_ids",
                    "source_ids",
                    "metadata_json",
                ],
            ),
        )
        hits = [] if not results else list(results[0])
        scoped = [self._vector_result_from_hit(hit, item_kind=item_kind) for hit in hits]
        if doc_ids:
            allowed = set(doc_ids)
            scoped = [result for result in scoped if self._result_scope_tokens(result) & allowed]
        scoped.sort(key=lambda item: (-item.score, item.item_id))
        return scoped[:limit]

    def get_entry(
        self,
        item_id: str,
        *,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> StoredVectorEntry | None:
        self._flush_dirty_collections()
        if not self._has_collection(item_kind=item_kind, embedding_space=embedding_space):
            return None
        collection = self._collection(item_kind=item_kind, embedding_space=embedding_space)
        rows = cast(
            list[dict[str, Any]],
            cast(Any, collection).query(
                expr=f'item_id == "{self._escape(item_id)}"',
                output_fields=[
                    "item_id",
                    "doc_id",
                    "source_id",
                    "segment_id",
                    "text",
                    "doc_ids",
                    "source_ids",
                    "metadata_json",
                    "embedding",
                ],
            ),
        )
        if not rows:
            return None
        row = rows[0]
        metadata = self._load_metadata(str(row["metadata_json"]))
        return StoredVectorEntry(
            item_id=str(row["item_id"]),
            item_kind=item_kind,
            embedding_space=embedding_space,
            doc_id=str(row["doc_id"]),
            segment_id=str(row["segment_id"]),
            text=str(row["text"]),
            metadata=metadata,
            vector=[float(value) for value in cast(list[float], row.get("embedding", []))],
        )

    def existing_item_ids(
        self,
        item_ids: Sequence[str],
        *,
        embedding_space: str | None = None,
        item_kind: str | None = "chunk",
    ) -> set[str]:
        normalized_ids = tuple(dict.fromkeys(item_ids))
        if not normalized_ids:
            return set()
        self._flush_dirty_collections()
        collected: set[str] = set()
        for collection in self._iter_target_collections(item_kind=item_kind, embedding_space=embedding_space):
            expr = " or ".join(f'item_id == "{self._escape(item_id)}"' for item_id in normalized_ids)
            rows = cast(
                list[dict[str, Any]],
                cast(Any, collection).query(expr=expr, output_fields=["item_id"]),
            )
            collected.update(str(row["item_id"]) for row in rows)
        return collected

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_chunks: bool = False,
    ) -> int:
        self._flush_dirty_collections()
        chunk_ids: set[str] = set()
        total = 0
        for collection in self._iter_target_collections(item_kind=item_kind, embedding_space=embedding_space):
            rows = cast(
                list[dict[str, Any]],
                cast(Any, collection).query(expr='item_id != ""', output_fields=["item_id"]),
            )
            if distinct_chunks:
                chunk_ids.update(str(row["item_id"]) for row in rows)
                continue
            total += len(rows)
        return len(chunk_ids) if distinct_chunks else total

    def delete_for_documents(
        self,
        doc_ids: Sequence[str],
        *,
        item_kind: str | None = None,
    ) -> int:
        normalized_ids = tuple(dict.fromkeys(doc_ids))
        if not normalized_ids:
            return 0
        self._flush_dirty_collections()
        deleted = 0
        expr = " or ".join(f'doc_id == "{self._escape(doc_id)}"' for doc_id in normalized_ids)
        for collection in self._iter_target_collections(item_kind=item_kind, embedding_space=None):
            result = cast(Any, collection).delete(expr)
            cast(Any, collection).flush()
            deleted += int(getattr(result, "delete_count", 0))
        return deleted

    def close(self) -> None:
        if not self._connected:
            return
        from pymilvus import connections  # type: ignore[import-untyped]

        self._flush_dirty_collections()
        for collection in self._collections.values():
            release = getattr(collection, "release", None)
            if callable(release):
                release()
        connections.disconnect(self._alias)
        self._collections.clear()
        self._connected = False

    def _connect(self) -> None:
        if self._connected:
            return
        from pymilvus import connections

        kwargs: dict[str, object] = {"alias": self._alias, "uri": self._uri}
        if self._token:
            kwargs["token"] = self._token
        if self._db_name:
            self._ensure_database_exists()
            kwargs["db_name"] = self._db_name
        connections.connect(**kwargs)
        self._connected = True

    def _ensure_database_exists(self) -> None:
        from pymilvus import connections, db

        if not self._db_name or self._db_name == "default":
            return
        bootstrap_alias = f"{self._alias}_bootstrap"
        kwargs: dict[str, object] = {"alias": bootstrap_alias, "uri": self._uri}
        if self._token:
            kwargs["token"] = self._token
        connections.connect(**kwargs)
        try:
            existing = set(cast(list[str], db.list_database(using=bootstrap_alias)))
            if self._db_name not in existing:
                db.create_database(self._db_name, using=bootstrap_alias)
        finally:
            connections.disconnect(bootstrap_alias)

    def _collection(self, *, item_kind: str, embedding_space: str, dimension: int | None = None) -> Any:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        name = self._collection_name(item_kind=item_kind, embedding_space=embedding_space)
        cached = self._collections.get(name)
        if cached is not None:
            return cached
        if not utility.has_collection(name, using=self._alias):
            if dimension is None:
                raise RuntimeError(f"Milvus collection {name} does not exist and no vector dimension was provided")
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="item_id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
                    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="segment_id", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="doc_ids", dtype=DataType.VARCHAR, max_length=4096),
                    FieldSchema(name="source_ids", dtype=DataType.VARCHAR, max_length=4096),
                    FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                ],
                description=f"RAG vectors for {item_kind}/{embedding_space}",
                enable_dynamic_field=False,
            )
            collection = Collection(name=name, schema=schema, using=self._alias)
            collection.create_index(
                field_name="embedding",
                index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
            )
        else:
            collection = Collection(name=name, using=self._alias)
        collection.load()
        self._collections[name] = collection
        return collection

    def _has_collection(self, *, item_kind: str, embedding_space: str) -> bool:
        from pymilvus import utility

        name = self._collection_name(item_kind=item_kind, embedding_space=embedding_space)
        return bool(utility.has_collection(name, using=self._alias))

    def _flush_dirty_collections(self) -> None:
        pending_names = list(self._pending_upserts)
        for name in pending_names:
            self._drain_pending_collection(name)
        if not self._dirty_collections:
            return
        pending = list(self._dirty_collections)
        for name in pending:
            collection = self._collections.get(name)
            if collection is None:
                continue
            cast(Any, collection).flush()
            self._dirty_collections.discard(name)

    def _drain_pending_collection(self, name: str) -> None:
        rows = self._pending_upserts.get(name)
        if not rows:
            return
        collection = self._collections.get(name)
        if collection is None:
            return
        cast(Any, collection).upsert(rows)
        self._dirty_collections.add(name)
        rows.clear()

    def _iter_target_collections(self, *, item_kind: str | None, embedding_space: str | None) -> list[Any]:
        from pymilvus import utility

        names = list(cast(list[str], utility.list_collections(using=self._alias)))
        targets: list[Any] = []
        prefix = f"{self._sanitize(self._collection_prefix)}__"
        for name in names:
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            parts = suffix.split("__", 1)
            if len(parts) != 2:
                continue
            collection_kind, collection_space = parts
            if item_kind is not None and collection_kind != self._sanitize(item_kind):
                continue
            if embedding_space is not None and collection_space != self._sanitize(embedding_space):
                continue
            targets.append(self._collection(item_kind=collection_kind, embedding_space=collection_space))
        return targets

    def _collection_name(self, *, item_kind: str, embedding_space: str) -> str:
        return (
            f"{self._sanitize(self._collection_prefix)}__"
            f"{self._sanitize(item_kind)}__"
            f"{self._sanitize(embedding_space)}"
        )

    @staticmethod
    def _sanitize(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
        return cleaned[:200] or "default"

    @staticmethod
    def _vector_result_from_hit(hit: object, *, item_kind: str) -> VectorSearchResult:
        entity = getattr(hit, "entity", None)
        if entity is None:
            raise RuntimeError("Milvus search hit did not include entity payload")
        metadata = MilvusVectorRepo._load_metadata(str(entity.get("metadata_json", "{}")))
        return VectorSearchResult(
            item_id=str(entity.get("item_id", "")),
            score=float(getattr(hit, "score", 0.0) or 0.0),
            item_kind=item_kind,
            doc_id=str(entity.get("doc_id", "")),
            source_id=str(entity.get("source_id", "")),
            segment_id=str(entity.get("segment_id", "")),
            text=str(entity.get("text", "")),
            metadata=metadata,
        )

    @staticmethod
    def _result_scope_tokens(result: VectorSearchResult) -> set[str]:
        tokens = {result.doc_id}
        metadata = result.metadata
        for key in ("source_id", "doc_id"):
            value = metadata.get(key)
            if value:
                tokens.add(value)
        for key in ("doc_ids", "source_ids"):
            value = metadata.get(key)
            if value:
                tokens.update(item.strip() for item in value.split(",") if item.strip())
        return tokens

    @staticmethod
    def _load_metadata(payload: str) -> dict[str, str]:
        return cast(dict[str, str], json.loads(payload))

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
