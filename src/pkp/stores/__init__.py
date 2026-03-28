from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.search.sqlite_vector_repo import SQLiteVectorRepo
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.stores.cache_store import CacheStore
from pkp.stores.chunk_store import ChunkStore
from pkp.stores.document_store import DocumentStore
from pkp.stores.graph_store import GraphStore
from pkp.stores.status_store import StatusStore
from pkp.stores.vector_store import VectorStore


@dataclass(slots=True)
class StorageBundle:
    root: Path
    documents: DocumentStore
    chunks: ChunkStore
    vectors: VectorStore
    graph: GraphStore
    status: StatusStore
    cache: CacheStore
    metadata_repo: SQLiteMetadataRepo
    vector_repo: SQLiteVectorRepo
    graph_repo: SQLiteGraphRepo
    _ephemeral_root: TemporaryDirectory[str] | None = field(default=None, repr=False)

    def close(self) -> None:
        self.metadata_repo.close()
        self.vector_repo.close()
        self.graph_repo.close()
        if self._ephemeral_root is not None:
            self._ephemeral_root.cleanup()


__all__ = [
    "CacheStore",
    "ChunkStore",
    "DocumentStore",
    "GraphStore",
    "StatusStore",
    "StorageBundle",
    "VectorStore",
]
