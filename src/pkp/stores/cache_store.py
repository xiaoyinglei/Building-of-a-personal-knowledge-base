from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.types.storage import CacheEntry


@dataclass(slots=True)
class CacheStore:
    metadata_repo: SQLiteMetadataRepo

    def save(self, entry: CacheEntry) -> CacheEntry:
        return self.metadata_repo.save_cache_entry(entry)

    def get(self, cache_key: str, *, namespace: str = "default") -> CacheEntry | None:
        return self.metadata_repo.get_cache_entry(cache_key, namespace=namespace)

    def list(self, *, namespace: str | None = None) -> list[CacheEntry]:
        return self.metadata_repo.list_cache_entries(namespace=namespace)

    def delete(self, cache_key: str, *, namespace: str = "default") -> None:
        self.metadata_repo.delete_cache_entry(cache_key, namespace=namespace)

    def purge_expired(self, *, now: datetime | None = None) -> int:
        return self.metadata_repo.purge_expired_cache_entries(now=now)
