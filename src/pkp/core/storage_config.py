from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.search.sqlite_vector_repo import SQLiteVectorRepo
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.stores import CacheStore, ChunkStore, DocumentStore, GraphStore, StatusStore, StorageBundle, VectorStore


@dataclass(frozen=True, slots=True)
class StorageConfig:
    backend: str = "sqlite"
    root: Path | None = None

    @classmethod
    def in_memory(cls) -> "StorageConfig":
        return cls(backend="in_memory", root=None)

    def build(self) -> StorageBundle:
        ephemeral_root: TemporaryDirectory[str] | None = None
        if self.backend == "in_memory":
            ephemeral_root = TemporaryDirectory(prefix="pkp-ragcore-")
            root = Path(ephemeral_root.name)
        elif self.backend == "sqlite":
            root = self.root or Path(".ragcore")
        else:
            raise ValueError(f"Unsupported storage backend: {self.backend}")

        root.mkdir(parents=True, exist_ok=True)

        metadata_repo = SQLiteMetadataRepo(root / "metadata.sqlite3")
        vector_repo = SQLiteVectorRepo(root / "vectors.sqlite3")
        graph_repo = SQLiteGraphRepo(root / "graph.sqlite3")

        documents = DocumentStore(metadata_repo=metadata_repo)
        chunks = ChunkStore(metadata_repo=metadata_repo)
        vectors = VectorStore(vector_repo=vector_repo)
        graph = GraphStore(graph_repo=graph_repo)
        status = StatusStore(metadata_repo=metadata_repo)
        cache = CacheStore(metadata_repo=metadata_repo)

        return StorageBundle(
            root=root,
            documents=documents,
            chunks=chunks,
            vectors=vectors,
            graph=graph,
            status=status,
            cache=cache,
            metadata_repo=metadata_repo,
            vector_repo=vector_repo,
            graph_repo=graph_repo,
            _ephemeral_root=ephemeral_root,
        )
