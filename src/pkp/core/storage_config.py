from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StorageConfig:
    backend: str = "sqlite"
    root: Path | None = None

    @classmethod
    def in_memory(cls) -> "StorageConfig":
        return cls(backend="in_memory", root=None)
