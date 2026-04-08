from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, cast

from rag.schema.runtime import StoredVectorEntry, VectorSearchResult


class PgvectorVectorRepo:
    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        self._dsn = dsn
        self._schema = schema
        self._conn: Any = self._connect()
        self._ensure_schema()

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
            f"""
            INSERT INTO {self._schema}.vectors (
                item_id,
                item_kind,
                embedding_space,
                doc_id,
                segment_id,
                text,
                scope_tokens,
                metadata_json,
                vector
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT(item_id, item_kind, embedding_space) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                segment_id = EXCLUDED.segment_id,
                text = EXCLUDED.text,
                scope_tokens = EXCLUDED.scope_tokens,
                metadata_json = EXCLUDED.metadata_json,
                vector = EXCLUDED.vector
            """,
            (
                item_id,
                item_kind,
                embedding_space,
                payload.get("doc_id", ""),
                payload.get("segment_id", ""),
                payload.get("text", ""),
                list(self._scope_tokens(payload)),
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
                self._vector_literal(vector),
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
        clauses = ["embedding_space = %s", "item_kind = %s"]
        params: list[object] = [self._vector_literal(query), embedding_space, item_kind]
        if doc_ids:
            clauses.append("scope_tokens && %s::text[]")
            params.append(doc_ids)
        params.extend([self._vector_literal(query), limit])
        rows = self._conn.execute(
            f"""
            SELECT
                item_id,
                doc_id,
                segment_id,
                text,
                metadata_json,
                1 - (vector <=> %s::vector) AS score
            FROM {self._schema}.vectors
            WHERE {' AND '.join(clauses)}
            ORDER BY vector <=> %s::vector ASC, item_id ASC
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
        return [
            VectorSearchResult(
                item_id=str(row["item_id"]),
                score=float(row["score"] or 0.0),
                item_kind=item_kind,
                doc_id=str(row["doc_id"]),
                source_id=str(self._load_metadata(str(row["metadata_json"])).get("source_id", "")),
                segment_id=str(row["segment_id"]),
                text=str(row["text"]),
                metadata=self._load_metadata(str(row["metadata_json"])),
            )
            for row in cast(list[dict[str, Any]], rows)
        ]

    def get_entry(
        self,
        item_id: str,
        *,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> StoredVectorEntry | None:
        row = self._conn.execute(
            f"""
            SELECT
                item_id,
                item_kind,
                embedding_space,
                doc_id,
                segment_id,
                text,
                metadata_json,
                vector::text AS vector_text
            FROM {self._schema}.vectors
            WHERE item_id = %s AND item_kind = %s AND embedding_space = %s
            """,
            (item_id, item_kind, embedding_space),
        ).fetchone()
        if row is None:
            return None
        metadata = self._load_metadata(str(row["metadata_json"]))
        return StoredVectorEntry(
            item_id=str(row["item_id"]),
            item_kind=str(row["item_kind"]),
            embedding_space=str(row["embedding_space"]),
            doc_id=str(row["doc_id"]),
            segment_id=str(row["segment_id"]),
            text=str(row["text"]),
            metadata=metadata,
            vector=self._parse_vector_text(str(row["vector_text"])),
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
        clauses = ["item_id = ANY(%s)"]
        params: list[object] = [list(normalized_ids)]
        if item_kind is not None:
            clauses.append("item_kind = %s")
            params.append(item_kind)
        if embedding_space is not None:
            clauses.append("embedding_space = %s")
            params.append(embedding_space)
        rows = self._conn.execute(
            f"SELECT item_id FROM {self._schema}.vectors WHERE {' AND '.join(clauses)}",
            tuple(params),
        ).fetchall()
        return {str(row["item_id"]) for row in cast(list[dict[str, Any]], rows)}

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_chunks: bool = False,
    ) -> int:
        select = "COUNT(DISTINCT item_id)" if distinct_chunks else "COUNT(*)"
        clauses: list[str] = []
        params: list[object] = []
        if embedding_space is not None:
            clauses.append("embedding_space = %s")
            params.append(embedding_space)
        if item_kind is not None:
            clauses.append("item_kind = %s")
            params.append(item_kind)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        row = self._conn.execute(
            f"SELECT {select} AS count FROM {self._schema}.vectors{where_sql}",
            tuple(params),
        ).fetchone()
        return int(row["count"]) if row is not None else 0

    def delete_for_documents(
        self,
        doc_ids: Sequence[str],
        *,
        item_kind: str | None = None,
    ) -> int:
        normalized_ids = tuple(dict.fromkeys(doc_ids))
        if not normalized_ids:
            return 0
        clauses = ["doc_id = ANY(%s)"]
        params: list[object] = [list(normalized_ids)]
        if item_kind is not None:
            clauses.append("item_kind = %s")
            params.append(item_kind)
        cursor = self._conn.execute(
            f"DELETE FROM {self._schema}.vectors WHERE {' AND '.join(clauses)}",
            tuple(params),
        )
        self._conn.commit()
        return int(cursor.rowcount)

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        statements = (
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"CREATE SCHEMA IF NOT EXISTS {self._schema}",
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.vectors (
                item_id TEXT NOT NULL,
                item_kind TEXT NOT NULL,
                embedding_space TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                text TEXT NOT NULL,
                scope_tokens TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                metadata_json TEXT NOT NULL,
                vector vector NOT NULL,
                PRIMARY KEY(item_id, item_kind, embedding_space)
            )
            """,
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_vectors_doc_id "
                f"ON {self._schema}.vectors(doc_id)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_vectors_scope_tokens "
                f"ON {self._schema}.vectors USING GIN(scope_tokens)"
            ),
        )
        for statement in statements:
            self._conn.execute(statement)
        self._conn.commit()

    def _connect(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self._dsn, row_factory=dict_row)

    @staticmethod
    def _vector_literal(vector: Iterable[float]) -> str:
        return "[" + ",".join(f"{float(value):.12g}" for value in vector) + "]"

    @staticmethod
    def _parse_vector_text(value: str) -> list[float]:
        stripped = value.strip().removeprefix("[").removesuffix("]")
        if not stripped:
            return []
        return [float(item.strip()) for item in stripped.split(",") if item.strip()]

    @staticmethod
    def _scope_tokens(metadata: dict[str, str]) -> tuple[str, ...]:
        tokens: list[str] = []
        for key in ("doc_id", "source_id"):
            value = metadata.get(key)
            if value:
                tokens.append(value)
        for key in ("doc_ids", "source_ids"):
            value = metadata.get(key)
            if value:
                tokens.extend(item.strip() for item in value.split(",") if item.strip())
        return tuple(sorted(dict.fromkeys(tokens)))

    @staticmethod
    def _load_metadata(payload: str) -> dict[str, str]:
        return cast(dict[str, str], json.loads(payload))
