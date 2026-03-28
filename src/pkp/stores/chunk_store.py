from __future__ import annotations

from dataclasses import dataclass

from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.types.content import Chunk


@dataclass(slots=True)
class ChunkStore:
    metadata_repo: SQLiteMetadataRepo

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
