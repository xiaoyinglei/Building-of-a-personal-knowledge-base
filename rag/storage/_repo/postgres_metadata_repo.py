from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

from rag.schema._types.artifact import KnowledgeArtifact
from rag.schema._types.content import Chunk, Document, Segment, Source
from rag.schema._types.storage import DocumentStatusRecord

TModel = TypeVar(
    "TModel",
    Source,
    Document,
    Segment,
    Chunk,
    KnowledgeArtifact,
    DocumentStatusRecord,
)


class PostgresMetadataRepo:
    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        self._dsn = dsn
        self._schema = schema
        self._conn: Any = self._connect()
        self._ensure_schema()

    @staticmethod
    def _dump(model: TModel) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=True)

    @staticmethod
    def _load(model_type: type[TModel], payload: str) -> TModel:
        return model_type.model_validate(json.loads(payload))

    def save_source(self, source: Source) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {self._schema}.sources (
                source_id,
                location,
                content_hash,
                ingest_version,
                saved_at,
                payload
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(source_id) DO UPDATE SET
                location = EXCLUDED.location,
                content_hash = EXCLUDED.content_hash,
                ingest_version = EXCLUDED.ingest_version,
                saved_at = EXCLUDED.saved_at,
                payload = EXCLUDED.payload
            """,
            (
                source.source_id,
                source.location,
                source.content_hash,
                source.ingest_version,
                datetime.now(UTC),
                self._dump(source),
            ),
        )
        self._conn.commit()

    def get_source(self, source_id: str) -> Source | None:
        row = self._fetchone(
            f"SELECT payload FROM {self._schema}.sources WHERE source_id = %s",
            (source_id,),
        )
        return None if row is None else self._load(Source, row["payload"])

    def get_source_by_location_and_hash(self, location: str, content_hash: str) -> Source | None:
        row = self._fetchone(
            f"""
            SELECT payload
            FROM {self._schema}.sources
            WHERE location = %s AND content_hash = %s
            ORDER BY ingest_version DESC, saved_at DESC
            LIMIT 1
            """,
            (location, content_hash),
        )
        return None if row is None else self._load(Source, row["payload"])

    def find_source_by_content_hash(self, content_hash: str) -> Source | None:
        row = self._fetchone(
            f"""
            SELECT payload
            FROM {self._schema}.sources
            WHERE content_hash = %s
            ORDER BY ingest_version DESC, saved_at DESC
            LIMIT 1
            """,
            (content_hash,),
        )
        return None if row is None else self._load(Source, row["payload"])

    def get_latest_source_for_location(self, location: str) -> Source | None:
        row = self._fetchone(
            f"""
            SELECT payload
            FROM {self._schema}.sources
            WHERE location = %s
            ORDER BY ingest_version DESC, saved_at DESC
            LIMIT 1
            """,
            (location,),
        )
        return None if row is None else self._load(Source, row["payload"])

    def list_sources(self, location: str | None = None) -> list[Source]:
        if location is None:
            rows = self._fetchall(f"SELECT payload FROM {self._schema}.sources ORDER BY saved_at, source_id")
        else:
            rows = self._fetchall(
                f"SELECT payload FROM {self._schema}.sources WHERE location = %s ORDER BY saved_at, source_id",
                (location,),
            )
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
            f"""
            INSERT INTO {self._schema}.documents (
                doc_id,
                source_id,
                location,
                content_hash,
                active,
                saved_at,
                payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                location = EXCLUDED.location,
                content_hash = EXCLUDED.content_hash,
                active = EXCLUDED.active,
                saved_at = EXCLUDED.saved_at,
                payload = EXCLUDED.payload
            """,
            (
                document.doc_id,
                document.source_id,
                location,
                content_hash,
                active,
                datetime.now(UTC),
                self._dump(document),
            ),
        )
        self._conn.commit()

    def save_document_bundle(self, document: Document, segments: list[Segment], chunks: list[Chunk]) -> None:
        location = document.metadata.get("location", document.title)
        content_hash = document.metadata.get("content_hash", document.doc_id)
        self.save_document(document, location=location, content_hash=content_hash)
        for segment in segments:
            self.save_segment(segment)
        for chunk in chunks:
            self.save_chunk(chunk)

    def get_document(self, doc_id: str) -> Document | None:
        row = self._fetchone(f"SELECT payload FROM {self._schema}.documents WHERE doc_id = %s", (doc_id,))
        return None if row is None else self._load(Document, row["payload"])

    def is_document_active(self, doc_id: str) -> bool:
        row = self._fetchone(f"SELECT active FROM {self._schema}.documents WHERE doc_id = %s", (doc_id,))
        return bool(row["active"]) if row is not None else False

    def list_documents(self, source_id: str | None = None, *, active_only: bool = False) -> list[Document]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if active_only:
            clauses.append("active = TRUE")
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._fetchall(
            f"SELECT payload FROM {self._schema}.documents{where_sql} ORDER BY saved_at, doc_id",
            tuple(params),
        )
        return [self._load(Document, row["payload"]) for row in rows]

    def get_active_document_by_location_and_hash(self, location: str, content_hash: str) -> Document | None:
        row = self._fetchone(
            f"""
            SELECT payload
            FROM {self._schema}.documents
            WHERE location = %s AND content_hash = %s AND active = TRUE
            ORDER BY saved_at DESC, doc_id DESC
            LIMIT 1
            """,
            (location, content_hash),
        )
        return None if row is None else self._load(Document, row["payload"])

    def get_latest_document_for_location(self, location: str) -> Document | None:
        row = self._fetchone(
            f"""
            SELECT payload
            FROM {self._schema}.documents
            WHERE location = %s
            ORDER BY saved_at DESC, doc_id DESC
            LIMIT 1
            """,
            (location,),
        )
        return None if row is None else self._load(Document, row["payload"])

    def deactivate_documents_for_location(self, location: str) -> None:
        self._conn.execute(f"UPDATE {self._schema}.documents SET active = FALSE WHERE location = %s", (location,))
        self._conn.commit()

    def set_document_active(self, doc_id: str, *, active: bool) -> None:
        self._conn.execute(f"UPDATE {self._schema}.documents SET active = %s WHERE doc_id = %s", (active, doc_id))
        self._conn.commit()

    def save_segment(self, segment: Segment) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {self._schema}.segments (segment_id, doc_id, order_index, saved_at, payload)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT(segment_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                order_index = EXCLUDED.order_index,
                saved_at = EXCLUDED.saved_at,
                payload = EXCLUDED.payload
            """,
            (
                segment.segment_id,
                segment.doc_id,
                segment.order_index,
                datetime.now(UTC),
                self._dump(segment),
            ),
        )
        self._conn.commit()

    def get_segment(self, segment_id: str) -> Segment | None:
        row = self._fetchone(f"SELECT payload FROM {self._schema}.segments WHERE segment_id = %s", (segment_id,))
        return None if row is None else self._load(Segment, row["payload"])

    def list_segments(self, doc_id: str) -> list[Segment]:
        rows = self._fetchall(
            f"SELECT payload FROM {self._schema}.segments WHERE doc_id = %s ORDER BY order_index, segment_id",
            (doc_id,),
        )
        return [self._load(Segment, row["payload"]) for row in rows]

    def delete_segments_for_document(self, doc_id: str) -> int:
        cursor = self._conn.execute(f"DELETE FROM {self._schema}.segments WHERE doc_id = %s", (doc_id,))
        self._conn.commit()
        return int(cursor.rowcount)

    def save_chunk(self, chunk: Chunk) -> None:
        order_index = int(
            chunk.order_index if chunk.order_index else chunk.metadata.get("order_index", chunk.citation_span[0])
        )
        self._conn.execute(
            f"""
            INSERT INTO {self._schema}.chunks (chunk_id, doc_id, segment_id, order_index, saved_at, payload)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                segment_id = EXCLUDED.segment_id,
                order_index = EXCLUDED.order_index,
                saved_at = EXCLUDED.saved_at,
                payload = EXCLUDED.payload
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.segment_id,
                order_index,
                datetime.now(UTC),
                self._dump(chunk),
            ),
        )
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._fetchone(f"SELECT payload FROM {self._schema}.chunks WHERE chunk_id = %s", (chunk_id,))
        return None if row is None else self._load(Chunk, row["payload"])

    def list_chunks(self, doc_id: str) -> list[Chunk]:
        rows = self._fetchall(
            f"SELECT payload FROM {self._schema}.chunks WHERE doc_id = %s ORDER BY order_index, chunk_id",
            (doc_id,),
        )
        return [self._load(Chunk, row["payload"]) for row in rows]

    def list_chunks_by_ids(self, chunk_ids: list[str] | tuple[str, ...]) -> list[Chunk]:
        normalized_ids = tuple(dict.fromkeys(chunk_ids))
        if not normalized_ids:
            return []
        rows = self._fetchall(
            f"SELECT payload FROM {self._schema}.chunks WHERE chunk_id = ANY(%s)",
            (list(normalized_ids),),
        )
        loaded = [self._load(Chunk, row["payload"]) for row in rows]
        chunk_by_id = {chunk.chunk_id: chunk for chunk in loaded}
        return [chunk_by_id[chunk_id] for chunk_id in normalized_ids if chunk_id in chunk_by_id]

    def delete_chunks_for_document(self, doc_id: str) -> int:
        cursor = self._conn.execute(f"DELETE FROM {self._schema}.chunks WHERE doc_id = %s", (doc_id,))
        self._conn.commit()
        return int(cursor.rowcount)

    def save_artifact(self, artifact: KnowledgeArtifact) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {self._schema}.artifacts (artifact_id, status, saved_at, payload)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(artifact_id) DO UPDATE SET
                status = EXCLUDED.status,
                saved_at = EXCLUDED.saved_at,
                payload = EXCLUDED.payload
            """,
            (
                artifact.artifact_id,
                artifact.status.value,
                datetime.now(UTC),
                self._dump(artifact),
            ),
        )
        self._conn.commit()

    def get_artifact(self, artifact_id: str) -> KnowledgeArtifact | None:
        row = self._fetchone(f"SELECT payload FROM {self._schema}.artifacts WHERE artifact_id = %s", (artifact_id,))
        return None if row is None else self._load(KnowledgeArtifact, row["payload"])

    def list_artifacts(self) -> list[KnowledgeArtifact]:
        rows = self._fetchall(f"SELECT payload FROM {self._schema}.artifacts ORDER BY saved_at, artifact_id")
        return [self._load(KnowledgeArtifact, row["payload"]) for row in rows]

    def save_document_status(self, status: DocumentStatusRecord) -> DocumentStatusRecord:
        normalized = status.model_copy(update={"updated_at": datetime.now(UTC)})
        self._conn.execute(
            f"""
            INSERT INTO {self._schema}.document_status (
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                location = EXCLUDED.location,
                content_hash = EXCLUDED.content_hash,
                status = EXCLUDED.status,
                stage = EXCLUDED.stage,
                attempts = EXCLUDED.attempts,
                error_message = EXCLUDED.error_message,
                updated_at = EXCLUDED.updated_at,
                payload = EXCLUDED.payload
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
                normalized.updated_at,
                self._dump(normalized),
            ),
        )
        self._conn.commit()
        return normalized

    def get_document_status(self, doc_id: str) -> DocumentStatusRecord | None:
        row = self._fetchone(
            f"SELECT payload FROM {self._schema}.document_status WHERE doc_id = %s",
            (doc_id,),
        )
        return None if row is None else self._load(DocumentStatusRecord, row["payload"])

    def list_document_statuses(
        self,
        *,
        source_id: str | None = None,
        status: str | None = None,
    ) -> list[DocumentStatusRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._fetchall(
            f"SELECT payload FROM {self._schema}.document_status{where_sql} ORDER BY updated_at DESC, doc_id",
            tuple(params),
        )
        return [self._load(DocumentStatusRecord, row["payload"]) for row in rows]

    def delete_document_status(self, doc_id: str) -> None:
        self._conn.execute(f"DELETE FROM {self._schema}.document_status WHERE doc_id = %s", (doc_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
        statements = (
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.sources (
                source_id TEXT PRIMARY KEY,
                location TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                ingest_version INTEGER NOT NULL,
                saved_at TIMESTAMPTZ NOT NULL,
                payload TEXT NOT NULL
            )
            """,
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_sources_location_hash "
                f"ON {self._schema}.sources(location, content_hash)"
            ),
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.documents (
                doc_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                location TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                active BOOLEAN NOT NULL,
                saved_at TIMESTAMPTZ NOT NULL,
                payload TEXT NOT NULL
            )
            """,
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_documents_location_hash "
                f"ON {self._schema}.documents(location, content_hash, active)"
            ),
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.segments (
                segment_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                saved_at TIMESTAMPTZ NOT NULL,
                payload TEXT NOT NULL
            )
            """,
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_segments_doc_order "
                f"ON {self._schema}.segments(doc_id, order_index)"
            ),
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                saved_at TIMESTAMPTZ NOT NULL,
                payload TEXT NOT NULL
            )
            """,
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_chunks_doc_order "
                f"ON {self._schema}.chunks(doc_id, order_index)"
            ),
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.artifacts (
                artifact_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                saved_at TIMESTAMPTZ NOT NULL,
                payload TEXT NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.document_status (
                doc_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                location TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                error_message TEXT,
                updated_at TIMESTAMPTZ NOT NULL,
                payload TEXT NOT NULL
            )
            """,
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_document_status_source "
                f"ON {self._schema}.document_status(source_id, status, updated_at)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_document_status_location "
                f"ON {self._schema}.document_status(location, content_hash, updated_at)"
            ),
        )
        for statement in statements:
            self._conn.execute(statement)
        self._conn.commit()

    def _connect(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self._dsn, row_factory=dict_row)

    def _fetchone(self, sql: str, params: tuple[object, ...] = ()) -> dict[str, Any] | None:
        row = self._conn.execute(sql, params).fetchone()
        return cast(dict[str, Any] | None, row)

    def _fetchall(self, sql: str, params: tuple[object, ...] = ()) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql, params).fetchall()
        return cast(list[dict[str, Any]], rows)
