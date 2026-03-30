from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AnswerCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str
    file_name: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: str
    chunk_type: str
    citation_anchor: str | None = None
    doc_id: str | None = None
    source_id: str | None = None
    source_type: str | None = None


class AnswerEvidenceLink(BaseModel):
    model_config = ConfigDict(frozen=True)

    link_id: str
    answer_section_id: str
    answer_excerpt: str
    evidence_chunk_id: str
    citation_id: str | None = None
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)


class AnswerSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: str
    title: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer_text: str
    answer_sections: list[AnswerSection] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    evidence_links: list[AnswerEvidenceLink] = Field(default_factory=list)
    groundedness_flag: bool
    insufficient_evidence_flag: bool

