from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryKind(StrEnum):
    USER_PREFERENCE = "user_preference"
    EPISODIC_SUMMARY = "episodic_summary"


class MemoryEvidenceLink(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    citation_anchor: str


class _BaseMemory(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    user_id: str
    evidence: list[MemoryEvidenceLink]
    source_scope: list[str] = Field(default_factory=list)
    reliability: float
    created_at: datetime
    updated_at: datetime

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: list[MemoryEvidenceLink]) -> list[MemoryEvidenceLink]:
        if not value:
            raise ValueError("evidence must not be empty")
        return value

    @field_validator("source_scope")
    @classmethod
    def normalize_source_scope(cls, value: list[str]) -> list[str]:
        return sorted(set(value))

    @model_validator(mode="after")
    def validate_source_scope(self) -> _BaseMemory:
        evidence_doc_ids = {item.doc_id for item in self.evidence}
        if any(doc_id not in evidence_doc_ids for doc_id in self.source_scope):
            raise ValueError("source_scope must be backed by evidence doc_ids")
        return self


class UserMemory(_BaseMemory):
    memory_kind: MemoryKind = MemoryKind.USER_PREFERENCE
    preference_key: str
    preference_summary: str

    @field_validator("preference_summary")
    @classmethod
    def validate_preference_summary(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("preference_summary must not be blank")
        return normalized

    @property
    def recall_hint(self) -> str:
        return f"Preference [{self.preference_key}]: {self.preference_summary}"


class EpisodicMemory(_BaseMemory):
    memory_kind: MemoryKind = MemoryKind.EPISODIC_SUMMARY
    session_id: str
    query: str
    episode_summary: str

    @field_validator("episode_summary")
    @classmethod
    def validate_episode_summary(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("episode_summary must not be blank")
        return normalized

    @property
    def recall_hint(self) -> str:
        return f"Episode [{self.query}]: {self.episode_summary}"


# Compatibility aliases for earlier internal naming.
UserProfile = UserMemory
ResearchEpisode = EpisodicMemory
