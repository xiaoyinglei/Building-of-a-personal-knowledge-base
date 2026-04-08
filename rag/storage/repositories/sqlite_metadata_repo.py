from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from rag.schema.core import Chunk, Document, Segment, Source
from rag.schema.query import KnowledgeArtifact
from rag.schema.runtime import CacheEntry, DocumentStatusRecord

TModel = TypeVar(
    "TModel",
    Source,
    Document,
    Segment,
    Chunk,
    KnowledgeArtifact,
    DocumentStatusRecord,
    CacheEntry,
)


class SQLiteMetadataRepo:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                location TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                ingest_version INTEGER NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sources_location_hash
            ON sources(location, content_hash);

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                location TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                active INTEGER NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES sources(source_id)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_location_hash
            ON documents(location, content_hash, active);

            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            );

            CREATE INDEX IF NOT EXISTS idx_segments_doc_order
            ON segments(doc_id, order_index);

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY(segment_id) REFERENCES segments(segment_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_order
            ON chunks(doc_id, order_index);

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_status (
                doc_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                location TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                error_message TEXT,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_document_status_source
            ON document_status(source_id, status, updated_at);

            CREATE INDEX IF NOT EXISTS idx_document_status_location
            ON document_status(location, content_hash, updated_at);

            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                payload TEXT NOT NULL,
                PRIMARY KEY(namespace, cache_key)
            );

            CREATE INDEX IF NOT EXISTS idx_cache_entries_updated
            ON cache_entries(namespace, updated_at);
            """
        )
        self._conn.commit()

    @staticmethod
    def _dump(model: TModel) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=True)

    @staticmethod
    def _load(model_type: type[TModel], payload: str) -> TModel:
        return model_type.model_validate(json.loads(payload))

    def save_source(self, source: Source) -> None:
        self._conn.execute(
            """
            INSERT INTO sources (
                source_id,
                location,
                content_hash,
                ingest_version,
                saved_at,
                payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                location=excluded.location,
                content_hash=excluded.content_hash,
                ingest_version=excluded.ingest_version,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (
                source.source_id,
                source.location,
                source.content_hash,
                source.ingest_version,
                datetime.now(UTC).isoformat(),
                self._dump(source),
            ),
        )
        self._conn.commit()

    def get_source(self, source_id: str) -> Source | None:
        row = self._conn.execute(
            "SELECT payload FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def get_source_by_location_and_hash(
        self,
        location: str,
        content_hash: str,
    ) -> Source | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM sources
            WHERE location = ? AND content_hash = ?
            ORDER BY ingest_version DESC, saved_at DESC
            LIMIT 1
            """,
            (location, content_hash),
        ).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def find_source_by_content_hash(self, content_hash: str) -> Source | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM sources
            WHERE content_hash = ?
            ORDER BY ingest_version DESC, saved_at DESC
            LIMIT 1
            """,
            (content_hash,),
        ).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def get_latest_source_for_location(self, location: str) -> Source | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM sources
            WHERE location = ?
            ORDER BY ingest_version DESC, saved_at DESC
            LIMIT 1
            """,
            (location,),
        ).fetchone()
        return None if row is None else self._load(Source, row["payload"])

    def list_sources(self, location: str | None = None) -> list[Source]:
        if location is None:
            rows = self._conn.execute(
                "SELECT payload FROM sources ORDER BY saved_at, source_id",
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload FROM sources WHERE location = ? ORDER BY saved_at, source_id",
                (location,),
            ).fetchall()
        return [self._load(Source, row["payload"]) for row in rows]

    def save_document(
        self,
        document: Document,
        *,
        location: str,
        content_hash: str,
        active: bool = True,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO documents (
                doc_id,
                source_id,
                location,
                content_hash,
                active,
                saved_at,
                payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_id=excluded.source_id,
                location=excluded.location,
                content_hash=excluded.content_hash,
                active=excluded.active,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (
                document.doc_id,
                document.source_id,
                location,
                content_hash,
                1 if active else 0,
                datetime.now(UTC).isoformat(),
                self._dump(document),
            ),
        )
        self._conn.commit()

    def save_document_bundle(
        self,
        document: Document,
        segments: list[Segment],
        chunks: list[Chunk],
    ) -> None:
        location = document.metadata.get("location", document.title)
        content_hash = document.metadata.get("content_hash", document.doc_id)
        self.save_document(document, location=location, content_hash=content_hash)
        for segment in segments:
            self.save_segment(segment)
        for chunk in chunks:
            self.save_chunk(chunk)

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT payload FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return None if row is None else self._load(Document, row["payload"])

    def is_document_active(self, doc_id: str) -> bool:
        row = self._conn.execute(
            "SELECT active FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return bool(row["active"]) if row is not None else False

    def list_documents(
        self,
        source_id: str | None = None,
        *,
        active_only: bool = False,
    ) -> list[Document]:
        clauses = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if active_only:
            clauses.append("active = 1")
        where_sql = "" if not clauses else f"WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(
            f"SELECT payload FROM documents {where_sql} ORDER BY saved_at, doc_id",
            tuple(params),
        ).fetchall()
        return [self._load(Document, row["payload"]) for row in rows]

    def get_active_document_by_location_and_hash(
        self,
        location: str,
        content_hash: str,
    ) -> Document | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM documents
            WHERE location = ? AND content_hash = ? AND active = 1
            ORDER BY saved_at DESC, doc_id DESC
            LIMIT 1
            """,
            (location, content_hash),
        ).fetchone()
        return None if row is None else self._load(Document, row["payload"])

    def get_latest_document_for_location(self, location: str) -> Document | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM documents
            WHERE location = ?
            ORDER BY saved_at DESC, doc_id DESC
            LIMIT 1
            """,
            (location,),
        ).fetchone()
        return None if row is None else self._load(Document, row["payload"])

    def deactivate_documents_for_location(self, location: str) -> None:
        self._conn.execute(
            "UPDATE documents SET active = 0 WHERE location = ?",
            (location,),
        )
        self._conn.commit()

    def set_document_active(self, doc_id: str, *, active: bool) -> None:
        self._conn.execute(
            "UPDATE documents SET active = ? WHERE doc_id = ?",
            (1 if active else 0, doc_id),
        )
        self._conn.commit()

    def save_segment(self, segment: Segment) -> None:
        self._conn.execute(
            """
            INSERT INTO segments (segment_id, doc_id, order_index, saved_at, payload)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(segment_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                order_index=excluded.order_index,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (
                segment.segment_id,
                segment.doc_id,
                segment.order_index,
                datetime.now(UTC).isoformat(),
                self._dump(segment),
            ),
        )
        self._conn.commit()

    def get_segment(self, segment_id: str) -> Segment | None:
        row = self._conn.execute(
            "SELECT payload FROM segments WHERE segment_id = ?",
            (segment_id,),
        ).fetchone()
        return None if row is None else self._load(Segment, row["payload"])

    def list_segments(self, doc_id: str) -> list[Segment]:
        rows = self._conn.execute(
            "SELECT payload FROM segments WHERE doc_id = ? ORDER BY order_index, segment_id",
            (doc_id,),
        ).fetchall()
        return [self._load(Segment, row["payload"]) for row in rows]

    def delete_segments_for_document(self, doc_id: str) -> int:
        cursor = self._conn.execute("DELETE FROM segments WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return int(cursor.rowcount)

    def save_chunk(self, chunk: Chunk) -> None:
        order_index = int(
            chunk.order_index if chunk.order_index else chunk.metadata.get("order_index", chunk.citation_span[0])
        )
        self._conn.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, segment_id, order_index, saved_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                segment_id=excluded.segment_id,
                order_index=excluded.order_index,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.segment_id,
                order_index,
                datetime.now(UTC).isoformat(),
                self._dump(chunk),
            ),
        )
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT payload FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return None if row is None else self._load(Chunk, row["payload"])

    def list_chunks(self, doc_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT payload FROM chunks WHERE doc_id = ? ORDER BY order_index, chunk_id",
            (doc_id,),
        ).fetchall()
        return [self._load(Chunk, row["payload"]) for row in rows]

    def delete_chunks_for_document(self, doc_id: str) -> int:
        cursor = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return int(cursor.rowcount)

    def list_chunks_by_ids(self, chunk_ids: list[str] | tuple[str, ...]) -> list[Chunk]:
        normalized_ids = tuple(dict.fromkeys(chunk_ids))
        if not normalized_ids:
            return []
        placeholders = ", ".join("?" for _ in normalized_ids)
        rows = self._conn.execute(
            f"SELECT payload FROM chunks WHERE chunk_id IN ({placeholders})",
            normalized_ids,
        ).fetchall()
        loaded = [self._load(Chunk, row["payload"]) for row in rows]
        chunk_by_id = {chunk.chunk_id: chunk for chunk in loaded}
        return [chunk_by_id[chunk_id] for chunk_id in normalized_ids if chunk_id in chunk_by_id]

    def save_artifact(self, artifact: KnowledgeArtifact) -> None:
        self._conn.execute(
            """
            INSERT INTO artifacts (artifact_id, status, saved_at, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                status=excluded.status,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (
                artifact.artifact_id,
                artifact.status.value,
                datetime.now(UTC).isoformat(),
                self._dump(artifact),
            ),
        )
        self._conn.commit()

    def get_artifact(self, artifact_id: str) -> KnowledgeArtifact | None:
        row = self._conn.execute(
            "SELECT payload FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return None if row is None else self._load(KnowledgeArtifact, row["payload"])

    def list_artifacts(self) -> list[KnowledgeArtifact]:
        rows = self._conn.execute(
            "SELECT payload FROM artifacts ORDER BY saved_at, artifact_id",
        ).fetchall()
        return [self._load(KnowledgeArtifact, row["payload"]) for row in rows]

    def save_document_status(self, status: DocumentStatusRecord) -> DocumentStatusRecord:
        normalized = status.model_copy(update={"updated_at": datetime.now(UTC)})
        self._conn.execute(
            """
            INSERT INTO document_status (
                doc_id,
                source_id,
                location,
                content_hash,
                status,
                stage,
                attempts,
                error_message,
                updated_at,
                payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_id=excluded.source_id,
                location=excluded.location,
                content_hash=excluded.content_hash,
                status=excluded.status,
                stage=excluded.stage,
                attempts=excluded.attempts,
                error_message=excluded.error_message,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                normalized.doc_id,
                normalized.source_id,
                normalized.location,
                normalized.content_hash,
                normalized.status.value,
                str(normalized.stage),
                normalized.attempts,
                normalized.error_message,
                normalized.updated_at.isoformat(),
                self._dump(normalized),
            ),
        )
        self._conn.commit()
        return normalized

    def get_document_status(self, doc_id: str) -> DocumentStatusRecord | None:
        row = self._conn.execute(
            "SELECT payload FROM document_status WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return None if row is None else self._load(DocumentStatusRecord, row["payload"])

    def list_document_statuses(
        self,
        *,
        source_id: str | None = None,
        status: str | None = None,
    ) -> list[DocumentStatusRecord]:
        clauses = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where_sql = "" if not clauses else f"WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(
            f"SELECT payload FROM document_status {where_sql} ORDER BY updated_at DESC, doc_id",
            tuple(params),
        ).fetchall()
        return [self._load(DocumentStatusRecord, row["payload"]) for row in rows]

    def delete_document_status(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM document_status WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def save_cache_entry(self, entry: CacheEntry) -> CacheEntry:
        now = datetime.now(UTC)
        existing = self.get_cache_entry(entry.cache_key, namespace=entry.namespace)
        normalized = entry.model_copy(
            update={
                "created_at": existing.created_at if existing is not None else entry.created_at,
                "updated_at": now,
            }
        )
        self._conn.execute(
            """
            INSERT INTO cache_entries (
                namespace,
                cache_key,
                created_at,
                updated_at,
                expires_at,
                payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, cache_key) DO UPDATE SET
                updated_at=excluded.updated_at,
                expires_at=excluded.expires_at,
                payload=excluded.payload
            """,
            (
                normalized.namespace,
                normalized.cache_key,
                normalized.created_at.isoformat(),
                normalized.updated_at.isoformat(),
                normalized.expires_at.isoformat() if normalized.expires_at is not None else None,
                self._dump(normalized),
            ),
        )
        self._conn.commit()
        return normalized

    def get_cache_entry(self, cache_key: str, *, namespace: str = "default") -> CacheEntry | None:
        row = self._conn.execute(
            """
            SELECT payload
            FROM cache_entries
            WHERE namespace = ? AND cache_key = ?
            """,
            (namespace, cache_key),
        ).fetchone()
        return None if row is None else self._load(CacheEntry, row["payload"])

    def list_cache_entries(self, *, namespace: str | None = None) -> list[CacheEntry]:
        if namespace is None:
            rows = self._conn.execute(
                "SELECT payload FROM cache_entries ORDER BY updated_at DESC, namespace, cache_key",
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT payload
                FROM cache_entries
                WHERE namespace = ?
                ORDER BY updated_at DESC, cache_key
                """,
                (namespace,),
            ).fetchall()
        return [self._load(CacheEntry, row["payload"]) for row in rows]

    def delete_cache_entry(self, cache_key: str, *, namespace: str = "default") -> None:
        self._conn.execute(
            "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
            (namespace, cache_key),
        )
        self._conn.commit()

    def purge_expired_cache_entries(self, *, now: datetime | None = None) -> int:
        effective_now = (now or datetime.now(UTC)).isoformat()
        cursor = self._conn.execute(
            """
            DELETE FROM cache_entries
            WHERE expires_at IS NOT NULL AND expires_at <= ?
            """,
            (effective_now,),
        )
        self._conn.commit()
        return int(cursor.rowcount)

    def close(self) -> None:
        self._conn.close()
