from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ArtifactType(StrEnum):
    DOCUMENT_SUMMARY = "document_summary"
    SECTION_SUMMARY = "section_summary"
    COMPARISON_PAGE = "comparison_page"
    TOPIC_PAGE = "topic_page"
    TIMELINE = "timeline"
    OPEN_QUESTION_PAGE = "open_question_page"


class ArtifactStatus(StrEnum):
    SUGGESTED = "suggested"
    APPROVED = "approved"
    STALE = "stale"
    CONFLICTED = "conflicted"
    ARCHIVED = "archived"


class KnowledgeArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    artifact_type: ArtifactType
    title: str
    supported_chunk_ids: list[str]
    confidence: float
    status: ArtifactStatus
    last_reviewed_at: datetime
    body_markdown: str
    source_scope: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
