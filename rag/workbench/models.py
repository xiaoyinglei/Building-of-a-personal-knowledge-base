from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WorkbenchModelProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    label: str
    provider_kind: str
    location: str
    chat_model: str | None = None
    embedding_model: str | None = None
    rerank_model: str | None = None
    supports_chat: bool = False
    supports_embedding: bool = False
    supports_rerank: bool = False
    compatible_with_index: bool = False
    compatibility_error: str | None = None


class WorkbenchFileEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    rel_path: str
    abs_path: str
    node_type: str
    source_type: str | None = None
    sync_state: str
    status: str | None = None
    stage: str | None = None
    error_message: str | None = None
    doc_id: str | None = None
    source_id: str | None = None
    chunk_count: int = 0
    ingest_version: int | None = None
    size_bytes: int | None = None
    modified_at: str | None = None
    children: list[WorkbenchFileEntry] = Field(default_factory=list)


class WorkbenchIndexSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    sources: int = 0
    documents: int = 0
    active_documents: int = 0
    chunks: int = 0
    vectors: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    statuses: dict[str, int] = Field(default_factory=dict)
    runtime_contract: dict[str, str | int | bool | None] = Field(default_factory=dict)


class WorkbenchState(BaseModel):
    model_config = ConfigDict(frozen=True)

    storage_root: str
    workspace_root: str
    backend_summary: list[str] = Field(default_factory=list)
    active_profile_id: str | None = None
    model_profiles: list[WorkbenchModelProfile] = Field(default_factory=list)
    index_summary: WorkbenchIndexSummary = Field(default_factory=WorkbenchIndexSummary)
    files_version: str = ""
    files: list[WorkbenchFileEntry] = Field(default_factory=list)
    sync_messages: list[str] = Field(default_factory=list)


class WorkbenchEvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    chunk_id: str
    doc_id: str
    source_id: str | None = None
    citation_anchor: str
    text: str
    score: float
    evidence_kind: str
    chunk_role: str | None = None
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    file_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_type: str | None = None
    source_type: str | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    retrieval_family: str | None = None
    token_count: int
    selected_token_count: int
    truncated: bool = False


class WorkbenchQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    mode: str
    profile_id: str | None = None
    answer_text: str
    insufficient_evidence: bool = False
    generation_provider: str | None = None
    generation_model: str | None = None
    rerank_provider: str | None = None
    mode_executor: str | None = None
    token_budget: int = 0
    token_count: int = 0
    truncated_count: int = 0
    diagnostics: dict[str, object] = Field(default_factory=dict)
    routing_decision: dict[str, object] = Field(default_factory=dict)
    query_understanding: dict[str, object] | None = None
    evidence: list[WorkbenchEvidenceItem] = Field(default_factory=list)


class WorkbenchOperationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    message: str
    error: str | None = None
    state: WorkbenchState | None = None
    query_result: WorkbenchQueryResult | None = None


WorkbenchFileEntry.model_rebuild()
