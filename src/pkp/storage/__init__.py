from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory

from pkp.storage._graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.storage._search.sqlite_vector_repo import SQLiteVectorRepo
from pkp.storage._repo.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.storage.doc_status import StatusStore
from pkp.storage.graph_store import GraphStore
from pkp.storage.kv_store import CacheStore, ChunkStore, DocumentStore
from pkp.storage.vector_store import VectorStore


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


@dataclass(frozen=True, slots=True)
class StorageConfig:
    backend: str = "sqlite"
    root: str | PathLike[str] | Path | None = None

    @classmethod
    def in_memory(cls) -> "StorageConfig":
        return cls(backend="in_memory", root=None)

    def build(self) -> StorageBundle:
        ephemeral_root: TemporaryDirectory[str] | None = None
        if self.backend == "in_memory":
            ephemeral_root = TemporaryDirectory(prefix="pkp-ragcore-")
            root = Path(ephemeral_root.name)
        elif self.backend == "sqlite":
            root = Path(self.root) if self.root is not None else Path(".ragcore")
        else:
            raise ValueError(f"Unsupported storage backend: {self.backend}")

        root.mkdir(parents=True, exist_ok=True)

        metadata_repo = SQLiteMetadataRepo(root / "metadata.sqlite3")
        vector_repo = SQLiteVectorRepo(root / "vectors.sqlite3")
        graph_repo = SQLiteGraphRepo(root / "graph.sqlite3")

        return StorageBundle(
            root=root,
            documents=DocumentStore(metadata_repo=metadata_repo),
            chunks=ChunkStore(metadata_repo=metadata_repo),
            vectors=VectorStore(vector_repo=vector_repo),
            graph=GraphStore(graph_repo=graph_repo),
            status=StatusStore(metadata_repo=metadata_repo),
            cache=CacheStore(metadata_repo=metadata_repo),
            metadata_repo=metadata_repo,
            vector_repo=vector_repo,
            graph_repo=graph_repo,
            _ephemeral_root=ephemeral_root,
        )


__all__ = [
    "CacheStore",
    "ChunkStore",
    "DocumentStore",
    "GraphStore",
    "StatusStore",
    "StorageBundle",
    "StorageConfig",
    "VectorStore",
]
