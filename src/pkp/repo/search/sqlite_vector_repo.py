from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from math import sqrt
from pathlib import Path

from pkp.repo.interfaces import VectorSearchResult


class SQLiteVectorRepo:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(vectors)").fetchall()
        }
        if columns and "embedding_space" not in columns:
            self._conn.execute("ALTER TABLE vectors RENAME TO vectors_legacy")

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT NOT NULL,
                embedding_space TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY(chunk_id, embedding_space)
            );

            CREATE INDEX IF NOT EXISTS idx_vectors_doc_id
            ON vectors(doc_id);

            CREATE INDEX IF NOT EXISTS idx_vectors_space_doc_id
            ON vectors(embedding_space, doc_id);
            """
        )
        legacy_exists = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vectors_legacy'"
        ).fetchone()
        if legacy_exists is not None:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO vectors (
                    chunk_id,
                    embedding_space,
                    doc_id,
                    segment_id,
                    text,
                    vector_json
                )
                SELECT chunk_id, 'default', doc_id, segment_id, text, vector_json
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
    ) -> None:
        payload = dict(metadata or {})
        self._conn.execute(
            """
            INSERT INTO vectors (chunk_id, embedding_space, doc_id, segment_id, text, vector_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id, embedding_space) DO UPDATE SET
                doc_id=excluded.doc_id,
                segment_id=excluded.segment_id,
                text=excluded.text,
                vector_json=excluded.vector_json
            """,
            (
                item_id,
                embedding_space,
                payload.get("doc_id", ""),
                payload.get("segment_id", ""),
                payload.get("text", ""),
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
    ) -> list[VectorSearchResult]:
        query_vector = tuple(float(value) for value in query)
        if not query_vector:
            return []

        sql = "SELECT chunk_id, doc_id, segment_id, text, vector_json FROM vectors WHERE embedding_space = ?"
        params: list[object] = [embedding_space]
        if doc_ids:
            placeholders = ", ".join("?" for _ in doc_ids)
            sql += f" AND doc_id IN ({placeholders})"
            params.extend(doc_ids)
        rows = self._conn.execute(sql, tuple(params)).fetchall()

        scored: list[VectorSearchResult] = []
        for row in rows:
            vector = tuple(float(value) for value in json.loads(row["vector_json"]))
            scored.append(
                VectorSearchResult(
                    item_id=row["chunk_id"],
                    score=self._cosine_similarity(query_vector, vector),
                    metadata={
                        "doc_id": row["doc_id"],
                        "segment_id": row["segment_id"],
                        "text": row["text"],
                    },
                )
            )

        scored.sort(key=lambda result: (-result.score, result.item_id))
        return scored[:limit]

    def existing_item_ids(
        self,
        item_ids: tuple[str, ...] | list[str],
        *,
        embedding_space: str | None = None,
    ) -> set[str]:
        normalized_ids = tuple(dict.fromkeys(item_ids))
        if not normalized_ids:
            return set()
        placeholders = ", ".join("?" for _ in normalized_ids)
        sql = f"SELECT chunk_id FROM vectors WHERE chunk_id IN ({placeholders})"
        params: list[object] = list(normalized_ids)
        if embedding_space is not None:
            sql += " AND embedding_space = ?"
            params.append(embedding_space)
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return {str(row["chunk_id"]) for row in rows}

    def count_vectors(self, *, embedding_space: str | None = None, distinct_chunks: bool = False) -> int:
        select = "COUNT(DISTINCT chunk_id)" if distinct_chunks else "COUNT(*)"
        sql = f"SELECT {select} AS count FROM vectors"
        params: tuple[object, ...] = ()
        if embedding_space is not None:
            sql += " WHERE embedding_space = ?"
            params = (embedding_space,)
        row = self._conn.execute(sql, params).fetchone()
        return int(row["count"]) if row is not None else 0

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
