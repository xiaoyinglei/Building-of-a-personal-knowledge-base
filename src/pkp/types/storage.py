from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"
    REBUILDING = "rebuilding"


class DocumentPipelineStage(StrEnum):
    INGEST = "ingest"
    PARSE = "parse"
    ROUTE = "route"
    CHUNK = "chunk"
    EXTRACT = "extract"
    PERSIST = "persist"
    INDEX = "index"
    DELETE = "delete"
    REBUILD = "rebuild"


class DocumentStatusRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_id: str
    location: str
    content_hash: str
    status: DocumentProcessingStatus
    stage: DocumentPipelineStage | str
    attempts: int = 0
    error_message: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = Field(default_factory=dict)


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    namespace: str
    cache_key: str
    payload: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
