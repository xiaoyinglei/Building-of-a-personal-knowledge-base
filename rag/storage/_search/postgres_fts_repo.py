from __future__ import annotations

from typing import Any, cast

from rag.utils._contracts import ChunkSearchResult


class PostgresFTSRepo:
    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        self._dsn = dsn
        self._schema = schema
        self._conn: Any = self._connect()
        self._ensure_schema()

    def index_chunk(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        source_id: str,
        title: str,
        toc_path: list[str],
        text: str,
    ) -> None:
        search_text = " ".join(part for part in (title, " ".join(toc_path), text) if part)
        self._conn.execute(
            f"""
            INSERT INTO {self._schema}.fts_chunks (
                chunk_id,
                doc_id,
                source_id,
                title,
                toc_path,
                text,
                search_vector
            )
            VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                source_id = EXCLUDED.source_id,
                title = EXCLUDED.title,
                toc_path = EXCLUDED.toc_path,
                text = EXCLUDED.text,
                search_vector = EXCLUDED.search_vector
            """,
            (chunk_id, doc_id, source_id, title, toc_path, text, search_text),
        )
        self._conn.commit()

    def search(self, query: str, *, limit: int = 10, doc_ids: list[str] | None = None) -> list[ChunkSearchResult]:
        normalized = query.strip()
        if not normalized:
            return []
        clauses = ["search_vector @@ websearch_to_tsquery('simple', %s)"]
        params: list[object] = [normalized, normalized, normalized]
        if doc_ids:
            clauses.append("doc_id = ANY(%s)")
            params.append(doc_ids)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT
                chunk_id,
                doc_id,
                source_id,
                title,
                toc_path,
                ts_headline(
                    'simple',
                    text,
                    websearch_to_tsquery('simple', %s),
                    'MaxFragments=2, MaxWords=12, MinWords=3'
                ) AS snippet,
                ts_rank_cd(search_vector, websearch_to_tsquery('simple', %s)) AS score
            FROM {self._schema}.fts_chunks
            WHERE {' AND '.join(clauses)}
            ORDER BY score DESC, chunk_id ASC
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
        return [
            ChunkSearchResult(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                source_id=str(row["source_id"]),
                title=str(row["title"]),
                toc_path=tuple(cast(list[str], row["toc_path"])),
                snippet=str(row["snippet"] or ""),
                score=float(row["score"] or 0.0),
            )
            for row in cast(list[dict[str, Any]], rows)
        ]

    def delete_by_chunk_ids(self, chunk_ids: list[str] | tuple[str, ...]) -> int:
        normalized_ids = tuple(dict.fromkeys(chunk_ids))
        if not normalized_ids:
            return 0
        cursor = self._conn.execute(
            f"DELETE FROM {self._schema}.fts_chunks WHERE chunk_id = ANY(%s)",
            (list(normalized_ids),),
        )
        self._conn.commit()
        return int(cursor.rowcount)

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
        statements = (
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.fts_chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                toc_path TEXT[] NOT NULL,
                text TEXT NOT NULL,
                search_vector tsvector NOT NULL
            )
            """,
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_fts_chunks_doc_id "
                f"ON {self._schema}.fts_chunks(doc_id)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_fts_chunks_vector "
                f"ON {self._schema}.fts_chunks USING GIN(search_vector)"
            ),
        )
        for statement in statements:
            self._conn.execute(statement)
        self._conn.commit()

    def _connect(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self._dsn, row_factory=dict_row)
