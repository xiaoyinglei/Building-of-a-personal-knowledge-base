from __future__ import annotations

from dataclasses import dataclass

from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.types.content import Document, Segment, Source


@dataclass(slots=True)
class DocumentStore:
    metadata_repo: SQLiteMetadataRepo

    def save_source(self, source: Source) -> None:
        self.metadata_repo.save_source(source)

    def get_source(self, source_id: str) -> Source | None:
        return self.metadata_repo.get_source(source_id)

    def get_source_by_location_and_hash(self, location: str, content_hash: str) -> Source | None:
        return self.metadata_repo.get_source_by_location_and_hash(location, content_hash)

    def list_sources(self, *, location: str | None = None) -> list[Source]:
        return self.metadata_repo.list_sources(location)

    def find_source_by_content_hash(self, content_hash: str) -> Source | None:
        return self.metadata_repo.find_source_by_content_hash(content_hash)

    def get_latest_source_for_location(self, location: str) -> Source | None:
        return self.metadata_repo.get_latest_source_for_location(location)

    def save_document(
        self,
        document: Document,
        *,
        location: str,
        content_hash: str,
        active: bool = True,
    ) -> None:
        self.metadata_repo.save_document(
            document,
            location=location,
            content_hash=content_hash,
            active=active,
        )

    def get_document(self, doc_id: str) -> Document | None:
        return self.metadata_repo.get_document(doc_id)

    def is_active(self, doc_id: str) -> bool:
        return self.metadata_repo.is_document_active(doc_id)

    def list_documents(
        self,
        source_id: str | None = None,
        *,
        active_only: bool = False,
    ) -> list[Document]:
        return self.metadata_repo.list_documents(source_id, active_only=active_only)

    def get_active_document_by_location_and_hash(
        self,
        location: str,
        content_hash: str,
    ) -> Document | None:
        return self.metadata_repo.get_active_document_by_location_and_hash(location, content_hash)

    def get_latest_document_for_location(self, location: str) -> Document | None:
        return self.metadata_repo.get_latest_document_for_location(location)

    def deactivate_documents_for_location(self, location: str) -> None:
        self.metadata_repo.deactivate_documents_for_location(location)

    def set_active(self, doc_id: str, *, active: bool) -> None:
        self.metadata_repo.set_document_active(doc_id, active=active)

    def save_segment(self, segment: Segment) -> None:
        self.metadata_repo.save_segment(segment)

    def get_segment(self, segment_id: str) -> Segment | None:
        return self.metadata_repo.get_segment(segment_id)

    def list_segments(self, doc_id: str) -> list[Segment]:
        return self.metadata_repo.list_segments(doc_id)

    def delete_segments_for_document(self, doc_id: str) -> int:
        return self.metadata_repo.delete_segments_for_document(doc_id)
