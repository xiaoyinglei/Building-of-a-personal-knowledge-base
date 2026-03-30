from __future__ import annotations

from dataclasses import dataclass

from rag.schema._types.storage import DocumentPipelineStage, DocumentProcessingStatus, DocumentStatusRecord
from rag.storage._repo.sqlite_metadata_repo import SQLiteMetadataRepo


@dataclass(slots=True)
class StatusStore:
    metadata_repo: SQLiteMetadataRepo

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
    "DocumentPipelineStage",
    "DocumentProcessingStatus",
    "DocumentStatusRecord",
    "StatusStore",
]
