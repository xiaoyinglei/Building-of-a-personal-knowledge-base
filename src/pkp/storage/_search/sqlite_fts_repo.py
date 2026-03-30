from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

from pkp.utils._contracts import ChunkSearchResult
from pkp.schema._types.text import build_fts_query, search_terms


class FtsChunkRecord(Protocol):
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str


class SQLiteFTSRepo:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                doc_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                toc_path TEXT NOT NULL,
                text TEXT NOT NULL
            );
            """
        )
        if self._fts_schema_needs_rebuild():
            self._conn.execute("DROP TABLE IF EXISTS chunks_fts")
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    title,
                    toc_path,
                    text,
                    search_terms,
                    chunk_pk UNINDEXED,
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    source_id UNINDEXED
                )
                """
            )
            self._rebuild_fts_from_chunks()
        self._conn.commit()

    def _fts_schema_needs_rebuild(self) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
        ).fetchone()
        if row is None:
            return True
        columns = {
            item["name"]
            for item in self._conn.execute("PRAGMA table_info(chunks_fts)").fetchall()
        }
        return "search_terms" not in columns

    def _rebuild_fts_from_chunks(self) -> None:
        rows = self._conn.execute(
            "SELECT chunk_pk, chunk_id, doc_id, source_id, title, toc_path, text FROM chunks ORDER BY chunk_pk"
        ).fetchall()
        for row in rows:
            toc_path = json.loads(row["toc_path"])
            self._conn.execute(
                """
                INSERT INTO chunks_fts (
                    rowid,
                    title,
                    toc_path,
                    text,
                    search_terms,
                    chunk_pk,
                    chunk_id,
                    doc_id,
                    source_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["chunk_pk"],
                    row["title"],
                    " ".join(toc_path),
                    row["text"],
                    self._search_terms_blob(row["title"], toc_path, row["text"]),
                    row["chunk_pk"],
                    row["chunk_id"],
                    row["doc_id"],
                    row["source_id"],
                ),
            )

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
        self._conn.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, source_id, title, toc_path, text)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                source_id=excluded.source_id,
                title=excluded.title,
                toc_path=excluded.toc_path,
                text=excluded.text
            """,
            (chunk_id, doc_id, source_id, title, json.dumps(toc_path), text),
        )
        row = self._conn.execute(
            "SELECT chunk_pk FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"failed to persist chunk {chunk_id}")
        chunk_pk = int(row["chunk_pk"])
        self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_pk,))
        self._conn.execute(
            """
            INSERT INTO chunks_fts (
                rowid,
                title,
                toc_path,
                text,
                search_terms,
                chunk_pk,
                chunk_id,
                doc_id,
                source_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_pk,
                title,
                " ".join(toc_path),
                text,
                self._search_terms_blob(title, toc_path, text),
                chunk_pk,
                chunk_id,
                doc_id,
                source_id,
            ),
        )
        self._conn.commit()

    def index_chunks(self, chunks: list[FtsChunkRecord]) -> None:
        for chunk in chunks:
            chunk_id = chunk.chunk_id
            doc_id = chunk.doc_id
            text = chunk.text
            citation_anchor = chunk.citation_anchor
            raw_source_id = getattr(chunk, "source_id", None)
            source_id = raw_source_id if isinstance(raw_source_id, str) and raw_source_id else doc_id
            self.index_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_id=source_id,
                title=citation_anchor,
                toc_path=[citation_anchor],
                text=text,
            )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkSearchResult]:
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        sql = """
            SELECT
                chunks.chunk_id AS chunk_id,
                chunks.doc_id AS doc_id,
                chunks.source_id AS source_id,
                chunks.title AS title,
                chunks.toc_path AS toc_path,
                snippet(chunks_fts, 2, '[', ']', '…', 8) AS snippet,
                bm25(chunks_fts) AS raw_score
            FROM chunks_fts
            JOIN chunks ON chunks.chunk_pk = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
        """
        params: list[object] = [normalized_query]
        if doc_ids:
            placeholders = ", ".join("?" for _ in doc_ids)
            sql += f" AND chunks.doc_id IN ({placeholders})"
            params.extend(doc_ids)
        sql += " ORDER BY raw_score ASC, chunks.chunk_id ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, tuple(params)).fetchall()

        return [
            ChunkSearchResult(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                source_id=row["source_id"],
                title=row["title"],
                toc_path=tuple(json.loads(row["toc_path"])),
                snippet=row["snippet"] or "",
                score=1.0 / (1.0 + max(float(row["raw_score"]), 0.0)),
            )
            for row in rows
        ]

    def delete_by_chunk_ids(self, chunk_ids: list[str] | tuple[str, ...]) -> int:
        normalized_ids = tuple(dict.fromkeys(chunk_ids))
        if not normalized_ids:
            return 0
        placeholders = ", ".join("?" for _ in normalized_ids)
        rows = self._conn.execute(
            f"SELECT chunk_pk FROM chunks WHERE chunk_id IN ({placeholders})",
            normalized_ids,
        ).fetchall()
        chunk_pks = [int(row["chunk_pk"]) for row in rows]
        if chunk_pks:
            pk_placeholders = ", ".join("?" for _ in chunk_pks)
            self._conn.execute(
                f"DELETE FROM chunks_fts WHERE rowid IN ({pk_placeholders})",
                tuple(chunk_pks),
            )
        cursor = self._conn.execute(
            f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
            normalized_ids,
        )
        self._conn.commit()
        return int(cursor.rowcount)

    @staticmethod
    def _normalize_query(query: str) -> str:
        return build_fts_query(query)

    @staticmethod
    def _search_terms_blob(title: str, toc_path: list[str], text: str) -> str:
        parts = [title, *toc_path, text]
        return " ".join(search_terms(" ".join(parts)))
