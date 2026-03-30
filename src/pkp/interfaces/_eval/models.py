from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pkp.schema._types.content import ChunkRole, SourceType


class OfflineEvalQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    fixture_id: str
    question: str
    category: str
    expected_terms: list[str]
    min_expected_terms: int = 1
    expected_chunk_role: ChunkRole | None = None
    expected_special_chunk_type: str | None = None
    expect_parent_backfill: bool = False
    scope_to_fixture: bool = True
    top_k: int = 5
    notes: str | None = None


class OfflineEvalFixture(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    fixture_id: str
    filename: str
    source_type: SourceType
    description: str
    path: Path


class ChunkInspectionSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    fixture_id: str
    filename: str
    chunk_id: str
    chunk_role: ChunkRole
    special_chunk_type: str | None = None
    text: str
    citation_anchor: str
    parent_text: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class QualityAuditSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_parent_chunks: int
    total_child_chunks: int
    total_special_chunks: int
    duplicate_searchable_chunks: int
    blank_chunks: int
    too_short_child_chunks: int
    too_long_child_chunks: int
    missing_metadata_chunks: int
    inspection_samples: list[ChunkInspectionSample] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval_kind: str
    rank: int
    chunk_id: str
    doc_id: str
    score: float
    chunk_role: ChunkRole
    special_chunk_type: str | None = None
    citation_anchor: str
    matched_terms: list[str] = Field(default_factory=list)
    matched_term_count: int = 0
    text_preview: str
    parent_chunk_id: str | None = None
    parent_matched_terms: list[str] = Field(default_factory=list)
    parent_matched_term_count: int = 0
    parent_text_preview: str | None = None
    is_expected_hit: bool = False


class OfflineEvalQuestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    fixture_id: str
    question: str
    category: str
    expected_terms: list[str]
    corpus_has_expected_answer: bool
    likely_issue: str
    vector_hit: bool
    fts_hit: bool
    runtime_hit: bool
    parent_backfill_improves: bool = False
    vector_top_k: list[RetrievalHit] = Field(default_factory=list)
    fts_top_k: list[RetrievalHit] = Field(default_factory=list)
    runtime_top_k: list[RetrievalHit] = Field(default_factory=list)


class OfflineEvalSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding_provider: str
    embedding_model: str | None
    embedding_space: str
    total_documents: int
    total_questions: int
    vector_hit_rate: float
    fts_hit_rate: float
    runtime_hit_rate: float
    parent_backfill_question_count: int
    parent_backfill_success_count: int
    table_question_success_count: int
    image_question_success_count: int


class OfflineEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: OfflineEvalSummary
    fixtures: list[OfflineEvalFixture]
    quality_audit: QualityAuditSummary
    question_results: list[OfflineEvalQuestionResult]


class OfflineEvalRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    report_json_path: Path
    report_markdown_path: Path
    report: OfflineEvalReport
