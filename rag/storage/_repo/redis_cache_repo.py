from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

from rag.schema._types.storage import CacheEntry


class RedisCacheRepo:
    def __init__(
        self,
        dsn: str,
        *,
        key_prefix: str = "rag-cache",
        client: object | None = None,
    ) -> None:
        self._dsn = dsn
        self._key_prefix = key_prefix.strip(":")
        self._client: Any | None = cast(Any | None, client)

    def save_cache_entry(self, entry: CacheEntry) -> CacheEntry:
        existing = self.get_cache_entry(entry.cache_key, namespace=entry.namespace)
        now = datetime.now(UTC)
        normalized = entry.model_copy(
            update={
                "created_at": existing.created_at if existing is not None else entry.created_at,
                "updated_at": now,
            }
        )
        payload = json.dumps(normalized.model_dump(mode="json"), ensure_ascii=True, sort_keys=True)
        ttl = self._ttl_seconds(normalized.expires_at, now=now)
        key = self._cache_key(normalized.namespace, normalized.cache_key)
        if ttl is None:
            self._redis().set(key, payload)
        else:
            self._redis().set(key, payload, ex=ttl)
        return normalized

    def get_cache_entry(self, cache_key: str, *, namespace: str = "default") -> CacheEntry | None:
        payload = self._redis().get(self._cache_key(namespace, cache_key))
        if payload is None:
            return None
        entry = CacheEntry.model_validate(json.loads(payload))
        if entry.expires_at is not None and entry.expires_at <= datetime.now(UTC):
            self.delete_cache_entry(cache_key, namespace=namespace)
            return None
        return entry

    def list_cache_entries(self, *, namespace: str | None = None) -> list[CacheEntry]:
        pattern = f"{self._key_prefix}:{namespace or '*'}:*"
        entries: list[CacheEntry] = []
        for key in self._redis().scan_iter(match=pattern):
            payload = self._redis().get(key)
            if payload is None:
                continue
            entry = CacheEntry.model_validate(json.loads(payload))
            if entry.expires_at is not None and entry.expires_at <= datetime.now(UTC):
                self._redis().delete(key)
                continue
            entries.append(entry)
        entries.sort(key=lambda item: (item.updated_at, item.namespace, item.cache_key), reverse=True)
        return entries

    def delete_cache_entry(self, cache_key: str, *, namespace: str = "default") -> None:
        self._redis().delete(self._cache_key(namespace, cache_key))

    def purge_expired_cache_entries(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        deleted = 0
        for key in self._redis().scan_iter(match=f"{self._key_prefix}:*"):
            payload = self._redis().get(key)
            if payload is None:
                continue
            entry = CacheEntry.model_validate(json.loads(payload))
            if entry.expires_at is None or entry.expires_at > current:
                continue
            deleted += int(self._redis().delete(key) or 0)
        return deleted

    def close(self) -> None:
        close = getattr(self._redis(), "close", None)
        if callable(close):
            close()

    def _redis(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self._dsn, decode_responses=True)
        return self._client

    def _cache_key(self, namespace: str, cache_key: str) -> str:
        return f"{self._key_prefix}:{namespace}:{cache_key}"

    @staticmethod
    def _ttl_seconds(expires_at: datetime | None, *, now: datetime) -> int | None:
        if expires_at is None:
            return None
        seconds = int((expires_at - now).total_seconds())
        return max(seconds, 1)
