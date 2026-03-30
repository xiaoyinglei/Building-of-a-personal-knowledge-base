# RAG Core Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `rag/` to cover the highest-value missing LightRAG and RAG-Anything core capabilities without reintroducing a second framework or bloating the package layout.

**Architecture:** Extend the current `rag` core in place. `bypass`, batch ingest, direct content-list ingestion, custom KG operations, Office-family parsing, and stronger multimodal orchestration all plug into the existing `engine`, `ingest`, `query`, `document`, and `storage` modules. No runtime/api/workbench layer returns.

**Tech Stack:** Python 3.12, Typer, Pydantic, SQLite, Docling, openpyxl, python-pptx, pytest, mypy, Ruff

---

### Task 1: Add Failing Tests For `bypass` Query Mode

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/core/test_query_modes.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_retrieval_service.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/core/test_public_api.py`

- [ ] **Step 1: Write failing tests for `QueryMode.BYPASS`**
- [ ] **Step 2: Verify failures**
Run: `uv run pytest -q tests/core/test_query_modes.py tests/service/test_retrieval_service.py tests/core/test_public_api.py`
Expected: failure because `bypass` is not defined or not handled.
- [ ] **Step 3: Implement `bypass` mode in query surface**
- [ ] **Step 4: Re-run focused tests**
Run: `uv run pytest -q tests/core/test_query_modes.py tests/service/test_retrieval_service.py tests/core/test_public_api.py`
Expected: pass.

### Task 2: Implement `bypass` Inside Existing Retrieval Orchestration

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/query/query.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/query/_retrieval/mode_planner.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/query/retrieve.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/query/_retrieval/fusion.py`

- [ ] **Step 1: Keep orchestration in one pipeline**
- [ ] **Step 2: Make `bypass` disable graph expansion, web retrieval, and planner-added specialty branches**
- [ ] **Step 3: Preserve evidence-backed rerank and answer generation**
- [ ] **Step 4: Run focused retrieval tests**
Run: `uv run pytest -q tests/core/test_query_modes.py tests/service/test_retrieval_service.py`
Expected: pass with no regression in existing modes.

### Task 3: Add Failing Tests For Batch Ingest

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/core/test_public_api.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/core/test_ingest_pipeline.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/ui/test_cli.py`

- [ ] **Step 1: Add failing tests for `insert_many()` and deterministic per-item results**
- [ ] **Step 2: Verify failures**
Run: `uv run pytest -q tests/core/test_public_api.py tests/core/test_ingest_pipeline.py tests/ui/test_cli.py`
Expected: failure because batch insert does not exist.
- [ ] **Step 3: Implement batch ingest without adding a second ingest framework**
- [ ] **Step 4: Re-run focused tests**
Run: `uv run pytest -q tests/core/test_public_api.py tests/core/test_ingest_pipeline.py tests/ui/test_cli.py`
Expected: pass.

### Task 4: Implement Direct Content List Insertion

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/engine.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/ingest/ingest.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/core/test_public_api.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_ingest_service_plain_text.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_ingest_service_web.py`

- [ ] **Step 1: Add failing tests for mixed content-list insertion**
- [ ] **Step 2: Normalize raw text/html/markdown/file-path/bytes items into ordinary ingest requests**
- [ ] **Step 3: Reuse existing deduplication, chunking, graph extraction, and indexing**
- [ ] **Step 4: Run focused ingest tests**
Run: `uv run pytest -q tests/core/test_public_api.py tests/service/test_ingest_service_plain_text.py tests/service/test_ingest_service_web.py`
Expected: pass.

### Task 5: Add Failing Tests For Custom KG Operations

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/repo/test_sqlite_graph_repo.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/core/test_public_api.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_retrieval_service.py`

- [ ] **Step 1: Add failing tests for node/edge upsert, evidence binding, deletion, and batch custom-KG insertion**
- [ ] **Step 2: Verify failures**
Run: `uv run pytest -q tests/repo/test_sqlite_graph_repo.py tests/core/test_public_api.py tests/service/test_retrieval_service.py`
Expected: failure because public graph ops are incomplete.
- [ ] **Step 3: Implement graph CRUD in storage and expose it through `RAG`**
- [ ] **Step 4: Re-run focused tests**
Run: `uv run pytest -q tests/repo/test_sqlite_graph_repo.py tests/core/test_public_api.py tests/service/test_retrieval_service.py`
Expected: pass.

### Task 6: Add Office-Family Coverage For `pptx` And `xlsx`

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/schema/_types/content.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/document/_parse/docling_parser_repo.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/ingest/ingest.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_document_processing_pipeline.py`

- [ ] **Step 1: Replace the current unsupported-file tests with failing `pptx` and `xlsx` ingest tests**
- [ ] **Step 2: Extend `SourceType` and Docling parser inference**
- [ ] **Step 3: Normalize `pptx` and `xlsx` Docling output into the current parsed-document contract**
- [ ] **Step 4: Re-run focused document-processing tests**
Run: `uv run pytest -q tests/service/test_document_processing_pipeline.py`
Expected: pass with `pptx` and `xlsx` covered.

### Task 7: Strengthen Multimodal Query Orchestration

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/query/_retrieval/mode_planner.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/query/retrieve.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/query/graph.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_retrieval_service.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/core/test_context_pipeline.py`

- [ ] **Step 1: Add failing tests for table/figure/formula/ocr-focused retrieval**
- [ ] **Step 2: Improve modality-aware branch selection and evidence competition**
- [ ] **Step 3: Keep one fusion/rerank/evidence path**
- [ ] **Step 4: Re-run focused multimodal retrieval tests**
Run: `uv run pytest -q tests/service/test_retrieval_service.py tests/core/test_context_pipeline.py`
Expected: pass.

### Task 8: Add Optional VLM Enrichment Without A Second Pipeline

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/document/_parse/docling_parser_repo.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/utils/_contracts.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_document_processing_pipeline.py`

- [ ] **Step 1: Add failing tests for optional multimodal enrichment when a capable provider exists**
- [ ] **Step 2: Integrate optional VLM descriptions into the current parsed-element flow**
- [ ] **Step 3: Ensure clean fallback when VLM is unavailable**
- [ ] **Step 4: Re-run focused parser tests**
Run: `uv run pytest -q tests/service/test_document_processing_pipeline.py`
Expected: pass.

### Task 9: Minimal CLI Growth Only Where It Helps

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/rag/cli.py`
- Modify: `/Users/leixiaoying/LLM/RAG学习/tests/ui/test_cli.py`

- [ ] **Step 1: Decide which new surfaces deserve CLI exposure**
- [ ] **Step 2: Add only minimal commands or flags that materially help inspect/use the core**
- [ ] **Step 3: Re-run CLI tests**
Run: `uv run pytest -q tests/ui/test_cli.py`
Expected: pass.

### Task 10: Full Verification

**Files:**
- Modify: only files touched above

- [ ] **Step 1: Run Ruff**
Run: `uv run ruff check .`
Expected: `All checks passed!`
- [ ] **Step 2: Run mypy**
Run: `uv run mypy . --hide-error-context --no-error-summary`
Expected: no output.
- [ ] **Step 3: Run pytest**
Run: `uv run pytest -q`
Expected: full pass.
