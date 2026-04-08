from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rag.schema.runtime import VectorRepo, VectorSearchResult
from rag.storage.search_backends.in_memory_vector_repo import InMemoryVectorRepo
from rag.storage.search_backends.milvus_vector_repo import MilvusVectorRepo
from rag.storage.search_backends.pgvector_vector_repo import PgvectorVectorRepo
from rag.storage.search_backends.sqlite_vector_repo import SQLiteVectorRepo


@dataclass(slots=True)
class VectorStore:
    vector_repo: VectorRepo

    def upsert_chunk(
        self,
        chunk_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
        embedding_space: str = "default",
    ) -> None:
        self.vector_repo.upsert(
            chunk_id,
            vector,
            metadata=metadata,
            embedding_space=embedding_space,
            item_kind="chunk",
        )

    def upsert_entity(
        self,
        entity_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
        embedding_space: str = "default",
    ) -> None:
        self.vector_repo.upsert(
            entity_id,
            vector,
            metadata=metadata,
            embedding_space=embedding_space,
            item_kind="entity",
        )

    def upsert_relation(
        self,
        relation_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
        embedding_space: str = "default",
    ) -> None:
        self.vector_repo.upsert(
            relation_id,
            vector,
            metadata=metadata,
            embedding_space=embedding_space,
            item_kind="relation",
        )

    def search_chunks(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
    ) -> list[VectorSearchResult]:
        return self.vector_repo.search(
            query,
            limit=limit,
            doc_ids=doc_ids,
            embedding_space=embedding_space,
            item_kind="chunk",
        )

    def search_entities(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
    ) -> list[VectorSearchResult]:
        return self.vector_repo.search(
            query,
            limit=limit,
            doc_ids=doc_ids,
            embedding_space=embedding_space,
            item_kind="entity",
        )

    def search_relations(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
    ) -> list[VectorSearchResult]:
        return self.vector_repo.search(
            query,
            limit=limit,
            doc_ids=doc_ids,
            embedding_space=embedding_space,
            item_kind="relation",
        )

    def existing_chunk_ids(
        self,
        chunk_ids: list[str] | tuple[str, ...],
        *,
        embedding_space: str | None = None,
    ) -> set[str]:
        return self.vector_repo.existing_item_ids(
            chunk_ids,
            embedding_space=embedding_space,
            item_kind="chunk",
        )

    def delete_for_documents(
        self,
        doc_ids: list[str] | tuple[str, ...],
        *,
        item_kind: str | None = None,
    ) -> int:
        return self.vector_repo.delete_for_documents(doc_ids, item_kind=item_kind)


__all__ = [
    "InMemoryVectorRepo",
    "MilvusVectorRepo",
    "PgvectorVectorRepo",
    "SQLiteVectorRepo",
    "VectorStore",
]
