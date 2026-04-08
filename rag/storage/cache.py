from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rag.schema.runtime import CacheEntry, CacheRepo
from rag.storage.repositories.redis_cache_repo import RedisCacheRepo


@dataclass(slots=True)
class CacheStore:
    cache_repo: CacheRepo

    def save(self, entry: CacheEntry) -> CacheEntry:
        return self.cache_repo.save_cache_entry(entry)

    def get(self, cache_key: str, *, namespace: str = "default") -> CacheEntry | None:
        return self.cache_repo.get_cache_entry(cache_key, namespace=namespace)

    def list(self, *, namespace: str | None = None) -> list[CacheEntry]:
        return self.cache_repo.list_cache_entries(namespace=namespace)

    def delete(self, cache_key: str, *, namespace: str = "default") -> None:
        self.cache_repo.delete_cache_entry(cache_key, namespace=namespace)

    def purge_expired(self, *, now: datetime | None = None) -> int:
        return self.cache_repo.purge_expired_cache_entries(now=now)


__all__ = ["CacheStore", "RedisCacheRepo"]
