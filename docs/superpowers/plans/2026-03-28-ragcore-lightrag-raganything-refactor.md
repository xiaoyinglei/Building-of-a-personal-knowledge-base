# RAGCore LightRAG/RAG-Anything Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the project around a pure-library `RAGCore` that follows the LightRAG main pipeline, absorbs RAG-Anything multimodal processing ideas, and reuses the current project's strongest parsing, chunking, retrieval, rerank, and SQLite building blocks.

**Architecture:** The new core keeps a thin public API (`RAGCore`, `QueryOptions`, `StorageConfig`) and moves all real work into explicit pipelines and algorithms. Ingest follows `parse -> route -> chunk -> extract -> persist`; query follows `search -> truncation -> fusion/merge -> prompt build -> generation`; storage is split into KV, Vector, Graph, DocStatus, and Cache groups with provenance-preserving links.

**Tech Stack:** Python, Pydantic, SQLite, Docling, existing provider repos, pytest, Ruff, mypy

---

## Implementation Notes

- Reuse current strengths instead of rewriting blindly:
  - parse layer under `src/rag/repo/parse/**`
  - model providers under `src/rag/repo/models/**`
  - SQLite-backed storage under `src/rag/repo/storage/**`, `src/rag/repo/search/**`, `src/rag/repo/graph/**`
  - structural/special chunk knowledge from `src/rag/service/document_processing_service.py`
  - fusion/rerank ideas from `src/rag/service/retrieval_service.py`
- Remove these from the core control plane:
  - `src/rag/runtime/**`
  - `src/rag/ui/**`
  - artifact/session/telemetry/workbench-driven orchestration
- Hard design rules:
  - No FastAPI or CLI imports inside the new core path.
  - No route/runtime-specific types inside the new core path.
  - Every graph entity/relation must preserve source chunk provenance.
  - Token chunking stays as a first-class algorithm, even when structural chunking is enabled.

## File Map

- Create: `src/rag/core/rag_core.py`
- Create: `src/rag/core/options.py`
- Create: `src/rag/core/storage_config.py`
- Create: `src/rag/core/query_modes.py`
- Create: `src/rag/core/pipelines/ingest_pipeline.py`
- Create: `src/rag/core/pipelines/query_pipeline.py`
- Create: `src/rag/core/pipelines/delete_pipeline.py`
- Create: `src/rag/core/pipelines/rebuild_pipeline.py`
- Create: `src/rag/algorithms/chunking/token_chunker.py`
- Create: `src/rag/algorithms/chunking/structured_chunker.py`
- Create: `src/rag/algorithms/chunking/multimodal_chunk_router.py`
- Create: `src/rag/algorithms/extract/entity_relation_extractor.py`
- Create: `src/rag/algorithms/extract/entity_relation_merger.py`
- Create: `src/rag/algorithms/retrieval/mode_planner.py`
- Create: `src/rag/algorithms/retrieval/branch_retrievers.py`
- Create: `src/rag/algorithms/retrieval/fusion.py`
- Create: `src/rag/algorithms/retrieval/rerank.py`
- Create: `src/rag/algorithms/context_build/truncation.py`
- Create: `src/rag/algorithms/context_build/merge.py`
- Create: `src/rag/algorithms/context_build/prompt_builder.py`
- Create: `src/rag/algorithms/generation/answer_generator.py`
- Create: `src/rag/stores/document_store.py`
- Create: `src/rag/stores/chunk_store.py`
- Create: `src/rag/stores/vector_store.py`
- Create: `src/rag/stores/graph_store.py`
- Create: `src/rag/stores/status_store.py`
- Create: `src/rag/stores/cache_store.py`
- Create: `tests/core/test_public_api.py`
- Create: `tests/core/test_ingest_pipeline.py`
- Create: `tests/core/test_query_modes.py`
- Create: `tests/core/test_context_pipeline.py`
- Create: `tests/core/test_delete_rebuild.py`
- Modify: `src/rag/service/document_processing_service.py`
- Modify: `src/rag/service/retrieval_service.py`
- Modify: `src/rag/repo/graph/sqlite_graph_repo.py`
- Modify: `src/rag/repo/search/sqlite_vector_repo.py`
- Modify: `src/rag/repo/storage/sqlite_metadata_repo.py`
- Modify: `src/rag/bootstrap.py`
- Modify: `README.md`

### Task 1: Freeze the New Public Core API

**Files:**
- Create: `src/rag/core/rag_core.py`
- Create: `src/rag/core/options.py`
- Create: `src/rag/core/storage_config.py`
- Test: `tests/core/test_public_api.py`

- [ ] **Step 1: Write the failing public API tests**

```python
from rag.core.options import QueryOptions
from rag.core.rag_core import RAGCore
from rag.core.storage_config import StorageConfig


def test_ragcore_exposes_insert_query_delete_rebuild():
    core = RAGCore(storage=StorageConfig.in_memory())
    assert hasattr(core, "insert")
    assert hasattr(core, "query")
    assert hasattr(core, "delete")
    assert hasattr(core, "rebuild")


def test_query_options_defaults_to_mix_mode():
    options = QueryOptions()
    assert options.mode == "mix"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_public_api.py -q`
Expected: FAIL with import/module errors because the new public core does not exist yet.

- [ ] **Step 3: Implement minimal core API types**

```python
@dataclass(frozen=True)
class QueryOptions:
    mode: Literal["naive", "local", "global", "hybrid", "mix"] = "mix"


class RAGCore:
    def __init__(self, *, storage: StorageConfig) -> None: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_public_api.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/core tests/core/test_public_api.py
git commit -m "feat: add public rag core api skeleton"
```

### Task 2: Extract Chunking Into Independent Algorithms

**Files:**
- Create: `src/rag/algorithms/chunking/token_chunker.py`
- Create: `src/rag/algorithms/chunking/structured_chunker.py`
- Create: `src/rag/algorithms/chunking/multimodal_chunk_router.py`
- Modify: `src/rag/service/document_processing_service.py`
- Test: `tests/service/test_document_processing_pipeline.py`
- Test: `tests/core/test_ingest_pipeline.py`

- [ ] **Step 1: Write failing tests for token + structural + special chunk outputs**

```python
def test_token_chunker_produces_stable_child_chunks():
    chunks = chunk_by_tokens("a " * 500, chunk_token_size=64, chunk_overlap_token_size=8)
    assert len(chunks) > 1


def test_structured_chunker_keeps_parent_and_child_roles():
    result = structured_chunk(parsed_document_fixture)
    assert result.parent_chunks
    assert result.child_chunks


def test_multimodal_router_keeps_special_chunks():
    routed = route_multimodal_chunks(parsed_multimodal_fixture)
    assert any(chunk.special_chunk_type == "table" for chunk in routed.special_chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_ingest_pipeline.py tests/service/test_document_processing_pipeline.py -q`
Expected: FAIL because chunking is still trapped inside `DocumentProcessingService`.

- [ ] **Step 3: Move algorithmic code out of orchestration**

```python
def chunk_by_tokens(...): ...
def structured_chunk(...): ...
def route_multimodal_chunks(...): ...
```

Use the existing structural logic from `DocumentProcessingService`, keep token chunking as a hard-preserved primitive, and formalize the three-layer result:
- parent chunks
- token child chunks
- special multimodal chunks

- [ ] **Step 4: Update `DocumentProcessingService` to call the new algorithms**

Run: `uv run pytest tests/service/test_document_processing_pipeline.py tests/core/test_ingest_pipeline.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/algorithms/chunking src/rag/service/document_processing_service.py tests/core/test_ingest_pipeline.py tests/service/test_document_processing_pipeline.py
git commit -m "refactor: extract chunking algorithms from document processing"
```

### Task 3: Introduce LightRAG-Style Storage Groups

**Files:**
- Create: `src/rag/stores/document_store.py`
- Create: `src/rag/stores/chunk_store.py`
- Create: `src/rag/stores/vector_store.py`
- Create: `src/rag/stores/graph_store.py`
- Create: `src/rag/stores/status_store.py`
- Create: `src/rag/stores/cache_store.py`
- Modify: `src/rag/repo/storage/sqlite_metadata_repo.py`
- Modify: `src/rag/repo/search/sqlite_vector_repo.py`
- Modify: `src/rag/repo/graph/sqlite_graph_repo.py`
- Test: `tests/core/test_ingest_pipeline.py`
- Test: `tests/repo/test_sqlite_graph_repo.py`
- Test: `tests/repo/test_sqlite_vector_repo.py`

- [ ] **Step 1: Write failing store contract tests**

```python
def test_storage_groups_include_document_chunk_status_graph_cache():
    storage = StorageConfig.in_memory().build()
    assert storage.documents is not None
    assert storage.chunks is not None
    assert storage.status is not None
    assert storage.graph is not None
    assert storage.cache is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_ingest_pipeline.py tests/repo/test_sqlite_graph_repo.py tests/repo/test_sqlite_vector_repo.py -q`
Expected: FAIL because grouped store contracts do not exist yet.

- [ ] **Step 3: Build storage facades over existing SQLite repos**

Required storage split:
- KV: documents, chunks, cache
- Vector: entity vectors, relation vectors, chunk vectors
- Graph: entity/relation graph
- DocStatus: document processing state

Also extend graph/vector persistence to keep:
- entity-to-source chunks
- relation-to-source chunks
- document processing status

- [ ] **Step 4: Run tests to verify store contracts pass**

Run: `uv run pytest tests/core/test_ingest_pipeline.py tests/repo/test_sqlite_graph_repo.py tests/repo/test_sqlite_vector_repo.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/stores src/rag/repo/storage/sqlite_metadata_repo.py src/rag/repo/search/sqlite_vector_repo.py src/rag/repo/graph/sqlite_graph_repo.py tests/core/test_ingest_pipeline.py tests/repo/test_sqlite_graph_repo.py tests/repo/test_sqlite_vector_repo.py
git commit -m "feat: add lightrag-style grouped storage contracts"
```

### Task 4: Build the Ingest Pipeline Around Parse -> Chunk -> Extract -> Persist

**Files:**
- Create: `src/rag/core/pipelines/ingest_pipeline.py`
- Create: `src/rag/algorithms/extract/entity_relation_extractor.py`
- Create: `src/rag/algorithms/extract/entity_relation_merger.py`
- Modify: `src/rag/service/ingest_service.py`
- Test: `tests/core/test_ingest_pipeline.py`
- Test: `tests/integration/test_markdown_query_flow.py`
- Test: `tests/integration/test_pdf_ingest_query_flow.py`

- [ ] **Step 1: Write failing ingest pipeline tests**

```python
def test_insert_persists_document_chunks_entities_relations_and_status():
    result = core.insert(markdown_source_fixture)
    assert result.document_id
    assert result.chunk_count > 0
    assert result.entity_count >= 0
    assert result.relation_count >= 0
    assert result.status == "processed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_ingest_pipeline.py tests/integration/test_markdown_query_flow.py tests/integration/test_pdf_ingest_query_flow.py -q`
Expected: FAIL because there is no unified ingest pipeline under the new core.

- [ ] **Step 3: Implement the unified ingest flow**

Flow:
- parse source
- route content into text/special chunks
- create parent + token child + special chunk records
- extract entities/relations from token child chunks
- merge entity/relation descriptions with provenance
- persist to grouped stores
- persist document status

- [ ] **Step 4: Run tests to verify it passes**

Run: `uv run pytest tests/core/test_ingest_pipeline.py tests/integration/test_markdown_query_flow.py tests/integration/test_pdf_ingest_query_flow.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/core/pipelines/ingest_pipeline.py src/rag/algorithms/extract src/rag/service/ingest_service.py tests/core/test_ingest_pipeline.py tests/integration/test_markdown_query_flow.py tests/integration/test_pdf_ingest_query_flow.py
git commit -m "feat: add unified ingest pipeline for rag core"
```

### Task 5: Replace Fast/Deep Retrieval Control With LightRAG Query Modes

**Files:**
- Create: `src/rag/core/query_modes.py`
- Create: `src/rag/core/pipelines/query_pipeline.py`
- Create: `src/rag/algorithms/retrieval/mode_planner.py`
- Create: `src/rag/algorithms/retrieval/branch_retrievers.py`
- Create: `src/rag/algorithms/retrieval/fusion.py`
- Create: `src/rag/algorithms/retrieval/rerank.py`
- Modify: `src/rag/service/retrieval_service.py`
- Test: `tests/core/test_query_modes.py`
- Test: `tests/service/test_retrieval_service.py`

- [ ] **Step 1: Write failing query mode tests**

```python
def test_naive_mode_uses_chunk_vectors_only(): ...
def test_local_mode_uses_entities_then_related_chunks(): ...
def test_global_mode_uses_relations_then_related_chunks(): ...
def test_hybrid_mode_combines_local_and_global(): ...
def test_mix_mode_combines_graph_and_chunk_vector_paths(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_query_modes.py tests/service/test_retrieval_service.py -q`
Expected: FAIL because the new query mode planner does not exist.

- [ ] **Step 3: Implement branch retrieval and mode planner**

Required behavior:
- `naive`: chunk vector retrieval only
- `local`: entity retrieval -> related chunks
- `global`: relation retrieval -> related chunks
- `hybrid`: local + global
- `mix`: local + global + naive chunk vector

Keep the current project's useful branches as branch internals where applicable:
- keyword/FTS
- dense vector
- structure/metadata
- special multimodal

- [ ] **Step 4: Upgrade rerank into a unified post-fusion layer**

Rerank features must include:
- text relevance
- supporting branches
- structural hit
- chunk role
- special multimodal type
- graph provenance

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_query_modes.py tests/service/test_retrieval_service.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/rag/core/query_modes.py src/rag/core/pipelines/query_pipeline.py src/rag/algorithms/retrieval src/rag/service/retrieval_service.py tests/core/test_query_modes.py tests/service/test_retrieval_service.py
git commit -m "feat: implement lightrag-style query modes and unified rerank"
```

### Task 6: Separate Context Building Into Search -> Truncation -> Fusion -> Prompt Build

**Files:**
- Create: `src/rag/algorithms/context_build/truncation.py`
- Create: `src/rag/algorithms/context_build/merge.py`
- Create: `src/rag/algorithms/context_build/prompt_builder.py`
- Create: `src/rag/algorithms/generation/answer_generator.py`
- Test: `tests/core/test_context_pipeline.py`
- Test: `tests/service/test_answer_generation_service.py`

- [ ] **Step 1: Write failing context pipeline tests**

```python
def test_context_pipeline_truncates_before_prompt_build():
    result = build_context(search_results_fixture, options=QueryOptions(mode="mix"))
    assert result.prompt
    assert result.metadata["truncated"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_context_pipeline.py tests/service/test_answer_generation_service.py -q`
Expected: FAIL because context build is not separated into standalone algorithms.

- [ ] **Step 3: Implement the explicit context pipeline**

Required stages:
- search results in
- token/importance truncation
- fusion/merge with provenance preservation
- prompt build
- answer generation

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_context_pipeline.py tests/service/test_answer_generation_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/algorithms/context_build src/rag/algorithms/generation tests/core/test_context_pipeline.py tests/service/test_answer_generation_service.py
git commit -m "refactor: separate context building and answer generation pipeline"
```

### Task 7: Add Delete/Rebuild/DocStatus as First-Class Core Flows

**Files:**
- Create: `src/rag/core/pipelines/delete_pipeline.py`
- Create: `src/rag/core/pipelines/rebuild_pipeline.py`
- Modify: `src/rag/core/rag_core.py`
- Test: `tests/core/test_delete_rebuild.py`

- [ ] **Step 1: Write failing delete/rebuild tests**

```python
def test_delete_removes_doc_and_rebuilds_shared_graph_state(): ...
def test_rebuild_recomputes_vectors_and_graph_from_existing_docs(): ...
def test_doc_status_tracks_pending_processed_failed(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_delete_rebuild.py -q`
Expected: FAIL because delete/rebuild/status flows are not implemented in the new core.

- [ ] **Step 3: Implement delete/rebuild flows with provenance-aware cleanup**

Required behavior:
- delete chunks for the document
- remove orphaned entities/relations
- preserve shared entities/relations
- rebuild surviving descriptions/vectors/graph links
- update doc status throughout

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_delete_rebuild.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/core/pipelines/delete_pipeline.py src/rag/core/pipelines/rebuild_pipeline.py src/rag/core/rag_core.py tests/core/test_delete_rebuild.py
git commit -m "feat: add delete rebuild and doc status flows to rag core"
```

### Task 8: Demote Runtime/UI to Optional Adapters

**Files:**
- Modify: `src/rag/bootstrap.py`
- Modify: `README.md`
- Optional cleanup: `src/rag/runtime/**`, `src/rag/ui/**`
- Test: `tests/integration/test_project_bootstrap.py`

- [ ] **Step 1: Write failing bootstrap tests for library-first entry**

```python
def test_bootstrap_can_build_ragcore_without_fastapi_or_cli():
    core = build_rag_core(settings_fixture)
    assert core is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_project_bootstrap.py -q`
Expected: FAIL because bootstrap still centers the runtime container.

- [ ] **Step 3: Make bootstrap library-first**

Required behavior:
- bootstrap builds `RAGCore`
- runtime/UI become wrappers around `RAGCore`, not the opposite
- docs present the library API as the primary interface

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_project_bootstrap.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/bootstrap.py README.md tests/integration/test_project_bootstrap.py
git commit -m "refactor: make rag core the primary bootstrap target"
```

### Task 9: Full Verification Sweep

**Files:**
- Verify all files touched above

- [ ] **Step 1: Run focused core tests**

Run: `uv run pytest tests/core -q`
Expected: PASS

- [ ] **Step 2: Run service/repo/integration regression tests**

Run: `uv run pytest tests/service tests/repo tests/integration -q`
Expected: PASS or explicit known failures recorded during migration

- [ ] **Step 3: Run static checks**

Run: `uv run ruff check .`
Expected: PASS

Run: `uv run mypy src`
Expected: PASS

- [ ] **Step 4: Update plan checkboxes and docs**

Run: mark completed steps in this file and refresh README examples to use `RAGCore`.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "chore: verify rag core refactor baseline"
```

## Execution Order

1. Task 1: public API shell
2. Task 2: chunking extraction
3. Task 3: storage groups
4. Task 4: ingest pipeline
5. Task 5: query modes + rerank
6. Task 6: context build + generation
7. Task 7: delete/rebuild/status
8. Task 8: demote runtime/UI
9. Task 9: full verification

## Phase Boundary Rules

- Phase 1 complete after Tasks 1-3: core shell exists and storage/chunking are independent.
- Phase 2 complete after Tasks 4-6: insert/query path works end-to-end.
- Phase 3 complete after Tasks 7-9: delete/rebuild/library-first bootstrap work and regression suite is stable.

## Notes on Multimodal Scope

- First-class special chunks for `image`, `table`, and `equation` stay in scope from Task 2 onward.
- First-class multimodal retrieval stays as an extension-capable branch during Tasks 5-6.
- Query-time VLM-enhanced generation is optional until the text + graph + special chunk core is stable.
