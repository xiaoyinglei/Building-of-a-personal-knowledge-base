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
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_vectors_doc_id
            ON vectors(doc_id);
            """
        )
        self._conn.commit()

    def upsert(
        self,
        item_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        payload = dict(metadata or {})
        self._conn.execute(
            """
            INSERT INTO vectors (chunk_id, doc_id, segment_id, text, vector_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                segment_id=excluded.segment_id,
                text=excluded.text,
                vector_json=excluded.vector_json
            """,
            (
                item_id,
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
    ) -> list[VectorSearchResult]:
        query_vector = tuple(float(value) for value in query)
        if not query_vector:
            return []

        sql = "SELECT chunk_id, doc_id, segment_id, text, vector_json FROM vectors"
        params: list[object] = []
        if doc_ids:
            placeholders = ", ".join("?" for _ in doc_ids)
            sql += f" WHERE doc_id IN ({placeholders})"
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
