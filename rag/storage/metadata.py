from __future__ import annotations

from dataclasses import dataclass

from rag.schema.core import Chunk, Document, Segment, Source
from rag.schema.runtime import DocumentPipelineStage, DocumentProcessingStatus, DocumentStatusRecord, MetadataRepo
from rag.storage.repositories.postgres_metadata_repo import PostgresMetadataRepo
from rag.storage.repositories.sqlite_metadata_repo import SQLiteMetadataRepo


@dataclass(slots=True)
class DocumentStore:
    metadata_repo: MetadataRepo

    def save_source(self, source: Source) -> Source:
        return self.metadata_repo.save_source(source)

    def get_source(self, source_id: int) -> Source | None:
        return self.metadata_repo.get_source(source_id)

    def get_source_by_location_and_hash(self, location: str, content_hash: str) -> Source | None:
        return self.metadata_repo.get_source_by_location_and_hash(location, content_hash)

    def list_sources(self, *, location: str | None = None) -> list[Source]:
        return self.metadata_repo.list_sources(location)

    def find_source_by_content_hash(self, content_hash: str) -> Source | None:
        return self.metadata_repo.find_source_by_content_hash(content_hash)

    def get_latest_source_for_location(self, location: str) -> Source | None:
        return self.metadata_repo.get_latest_source_for_location(location)

    def save_document(self, document: Document) -> Document:
        return self.metadata_repo.save_document(document)

    def get_document(self, doc_id: int) -> Document | None:
        return self.metadata_repo.get_document(doc_id)

    def is_active(self, doc_id: int) -> bool:
        return self.metadata_repo.is_document_active(doc_id)

    def list_documents(
        self,
        source_id: int | None = None,
        *,
        active_only: bool = False,
        version_group_id: int | None = None,
    ) -> list[Document]:
        return self.metadata_repo.list_documents(
            source_id,
            active_only=active_only,
            version_group_id=version_group_id,
        )

    def get_active_document_by_location_and_hash(
        self,
        location: str,
        content_hash: str,
    ) -> Document | None:
        direct = getattr(self.metadata_repo, "get_active_document_by_location_and_hash", None)
        if callable(direct):
            return direct(location, content_hash)
        source = self.metadata_repo.get_source_by_location_and_hash(location, content_hash)
        if source is None:
            return None
        documents = self.metadata_repo.list_documents(source.source_id, active_only=True)
        return documents[0] if documents else None

    def get_latest_document_for_location(self, location: str) -> Document | None:
        direct = getattr(self.metadata_repo, "get_latest_document_for_location", None)
        if callable(direct):
            return direct(location)
        source = self.metadata_repo.get_latest_source_for_location(location)
        if source is None:
            return None
        documents = self.metadata_repo.list_documents(source.source_id, active_only=False)
        return documents[0] if documents else None

    def deactivate_documents_for_location(self, location: str) -> None:
        direct = getattr(self.metadata_repo, "deactivate_documents_for_location", None)
        if callable(direct):
            direct(location)
            return
        for source in self.metadata_repo.list_sources(location):
            for document in self.metadata_repo.list_documents(source.source_id, active_only=False):
                self.set_active(document.doc_id, active=False)

    def set_active(self, doc_id: int, *, active: bool) -> None:
        activate_version = getattr(self.metadata_repo, "activate_document_version", None)
        if active and callable(activate_version):
            activate_version(doc_id)
            return
        self.metadata_repo.set_document_active(doc_id, active=active)

    def save_segment(self, segment: Segment) -> None:
        self.metadata_repo.save_segment(segment)

    def get_segment(self, segment_id: str) -> Segment | None:
        return self.metadata_repo.get_segment(segment_id)

    def list_segments(self, doc_id: str) -> list[Segment]:
        return self.metadata_repo.list_segments(doc_id)

    def delete_segments_for_document(self, doc_id: str) -> int:
        return self.metadata_repo.delete_segments_for_document(doc_id)


@dataclass(slots=True)
class ChunkStore:
    metadata_repo: MetadataRepo

    def save(self, chunk: Chunk) -> None:
        self.metadata_repo.save_chunk(chunk)

    def save_many(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.metadata_repo.save_chunk(chunk)

    def get(self, chunk_id: str) -> Chunk | None:
        return self.metadata_repo.get_chunk(chunk_id)

    def list_by_document(self, doc_id: str) -> list[Chunk]:
        return self.metadata_repo.list_chunks(doc_id)

    def list_by_ids(self, chunk_ids: list[str] | tuple[str, ...]) -> list[Chunk]:
        return self.metadata_repo.list_chunks_by_ids(chunk_ids)

    def delete_for_document(self, doc_id: str) -> int:
        return self.metadata_repo.delete_chunks_for_document(doc_id)


@dataclass(slots=True)
class StatusStore:
    metadata_repo: MetadataRepo

    def save(self, status: DocumentStatusRecord) -> DocumentStatusRecord:
        return self.metadata_repo.save_document_status(status)

    def get(self, doc_id: str) -> DocumentStatusRecord | None:
        return self.metadata_repo.get_document_status(doc_id)

    def list(
        self,
        *,
        source_id: str | None = None,
        status: str | None = None,
    ) -> list[DocumentStatusRecord]:
        return self.metadata_repo.list_document_statuses(source_id=source_id, status=status)

    def delete(self, doc_id: str) -> None:
        self.metadata_repo.delete_document_status(doc_id)


__all__ = [
    "ChunkStore",
    "DocumentPipelineStage",
    "DocumentProcessingStatus",
    "DocumentStatusRecord",
    "DocumentStore",
    "PostgresMetadataRepo",
    "SQLiteMetadataRepo",
    "StatusStore",
]
