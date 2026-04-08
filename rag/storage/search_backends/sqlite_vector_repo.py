from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from math import sqrt
from pathlib import Path

from rag.schema.runtime import StoredVectorEntry, VectorSearchResult


class SQLiteVectorRepo:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(vectors)").fetchall()}
        if columns and ("item_id" not in columns or "item_kind" not in columns or "metadata_json" not in columns):
            self._conn.execute("ALTER TABLE vectors RENAME TO vectors_legacy")

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                item_id TEXT NOT NULL,
                item_kind TEXT NOT NULL,
                embedding_space TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY(item_id, item_kind, embedding_space)
            );

            CREATE INDEX IF NOT EXISTS idx_vectors_doc_id
            ON vectors(doc_id);

            CREATE INDEX IF NOT EXISTS idx_vectors_space_doc_id
            ON vectors(embedding_space, doc_id);

            CREATE INDEX IF NOT EXISTS idx_vectors_kind_space
            ON vectors(item_kind, embedding_space, doc_id);
            """
        )
        legacy_exists = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vectors_legacy'"
        ).fetchone()
        if legacy_exists is not None:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO vectors (
                    item_id,
                    item_kind,
                    embedding_space,
                    doc_id,
                    segment_id,
                    text,
                    metadata_json,
                    vector_json
                )
                SELECT chunk_id, 'chunk', 'default', doc_id, segment_id, text, '{}', vector_json
                FROM vectors_legacy
                """
            )
            self._conn.execute("DROP TABLE vectors_legacy")
        self._conn.commit()

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
        self._conn.execute(
            """
            INSERT INTO vectors (
                item_id,
                item_kind,
                embedding_space,
                doc_id,
                segment_id,
                text,
                metadata_json,
                vector_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id, item_kind, embedding_space) DO UPDATE SET
                doc_id=excluded.doc_id,
                segment_id=excluded.segment_id,
                text=excluded.text,
                metadata_json=excluded.metadata_json,
                vector_json=excluded.vector_json
            """,
            (
                item_id,
                item_kind,
                embedding_space,
                payload.get("doc_id", ""),
                payload.get("segment_id", ""),
                payload.get("text", ""),
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
                json.dumps([float(value) for value in vector], ensure_ascii=True),
            ),
        )
        self._conn.commit()

    def search(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> list[VectorSearchResult]:
        query_vector = tuple(float(value) for value in query)
        if not query_vector:
            return []

        sql = """
            SELECT item_id, doc_id, segment_id, text, metadata_json, vector_json
            FROM vectors
            WHERE embedding_space = ? AND item_kind = ?
        """
        params: list[object] = [embedding_space, item_kind]
        rows = self._conn.execute(sql, tuple(params)).fetchall()

        scored: list[VectorSearchResult] = []
        requested_scope = set(doc_ids or [])
        for row in rows:
            vector = tuple(float(value) for value in json.loads(row["vector_json"]))
            metadata = json.loads(row["metadata_json"])
            if requested_scope and not (
                self._vector_scope_tokens(metadata=metadata, row_doc_id=row["doc_id"]) & requested_scope
            ):
                continue
            scored.append(
                VectorSearchResult(
                    item_id=row["item_id"],
                    score=self._cosine_similarity(query_vector, vector),
                    item_kind=str(item_kind),
                    doc_id=str(row["doc_id"]),
                    source_id=str(metadata.get("source_id", "")),
                    segment_id=str(row["segment_id"]),
                    text=str(row["text"]),
                    metadata=metadata
                    | {
                        "doc_id": row["doc_id"],
                        "source_id": metadata.get("source_id", ""),
                        "segment_id": row["segment_id"],
                        "text": row["text"],
                    },
                )
            )

        scored.sort(key=lambda result: (-result.score, result.item_id))
        return scored[:limit]

    def get_entry(
        self,
        item_id: str,
        *,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> StoredVectorEntry | None:
        row = self._conn.execute(
            """
            SELECT item_id, item_kind, embedding_space, doc_id, segment_id, text, metadata_json, vector_json
            FROM vectors
            WHERE item_id = ? AND item_kind = ? AND embedding_space = ?
            """,
            (item_id, item_kind, embedding_space),
        ).fetchone()
        if row is None:
            return None
        return StoredVectorEntry(
            item_id=str(row["item_id"]),
            item_kind=str(row["item_kind"]),
            embedding_space=str(row["embedding_space"]),
            doc_id=str(row["doc_id"]),
            segment_id=str(row["segment_id"]),
            text=str(row["text"]),
            metadata=json.loads(row["metadata_json"]),
            vector=[float(value) for value in json.loads(row["vector_json"])],
        )

    def existing_item_ids(
        self,
        item_ids: tuple[str, ...] | list[str],
        *,
        embedding_space: str | None = None,
        item_kind: str | None = "chunk",
    ) -> set[str]:
        normalized_ids = tuple(dict.fromkeys(item_ids))
        if not normalized_ids:
            return set()
        placeholders = ", ".join("?" for _ in normalized_ids)
        sql = f"SELECT item_id FROM vectors WHERE item_id IN ({placeholders})"
        params: list[object] = list(normalized_ids)
        if item_kind is not None:
            sql += " AND item_kind = ?"
            params.append(item_kind)
        if embedding_space is not None:
            sql += " AND embedding_space = ?"
            params.append(embedding_space)
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return {str(row["item_id"]) for row in rows}

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_chunks: bool = False,
    ) -> int:
        select = "COUNT(DISTINCT item_id)" if distinct_chunks else "COUNT(*)"
        sql = f"SELECT {select} AS count FROM vectors"
        clauses = []
        params: list[object] = []
        if embedding_space is not None:
            clauses.append("embedding_space = ?")
            params.append(embedding_space)
        if item_kind is not None:
            clauses.append("item_kind = ?")
            params.append(item_kind)
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        row = self._conn.execute(sql, params).fetchone()
        return int(row["count"]) if row is not None else 0

    def delete_for_documents(
        self,
        doc_ids: tuple[str, ...] | list[str],
        *,
        item_kind: str | None = None,
    ) -> int:
        normalized_ids = tuple(dict.fromkeys(doc_ids))
        if not normalized_ids:
            return 0
        placeholders = ", ".join("?" for _ in normalized_ids)
        sql = f"DELETE FROM vectors WHERE doc_id IN ({placeholders})"
        params: list[object] = list(normalized_ids)
        if item_kind is not None:
            sql += " AND item_kind = ?"
            params.append(item_kind)
        cursor = self._conn.execute(sql, tuple(params))
        self._conn.commit()
        return int(cursor.rowcount)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _vector_scope_tokens(*, metadata: dict[str, str], row_doc_id: str) -> set[str]:
        tokens = {row_doc_id}
        for key in ("doc_id", "source_id"):
            value = metadata.get(key)
            if value:
                tokens.add(value)
        for key in ("doc_ids", "source_ids"):
            value = metadata.get(key)
            if not value:
                continue
            tokens.update(item.strip() for item in value.split(",") if item.strip())
        return tokens

    @staticmethod
    def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            return 0.0
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot = sum(lv * rv for lv, rv in zip(left, right, strict=True))
        return dot / (left_norm * right_norm)
