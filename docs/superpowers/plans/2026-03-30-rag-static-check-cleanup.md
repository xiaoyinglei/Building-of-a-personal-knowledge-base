# RAG Static Check Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the new `rag` core pass full `ruff`, full `mypy`, and keep the full test suite green.

**Architecture:** Treat this as a constrained cleanup, not a redesign. First remove mechanical lint noise with automated tools, then fix type errors in clustered modules (`ingest`, `query`, `storage`, selected tests) using minimal source edits and repeated verification.

**Tech Stack:** Python 3.12, Ruff, mypy, pytest, Typer, Pydantic

---

### Task 1: Capture Baseline And Apply Safe Automated Ruff Fixes

**Files:**
- Modify: `rag/**/*.py`
- Modify: `tests/**/*.py`
- Verify: `pyproject.toml`

- [ ] **Step 1: Record the current failing commands**

Run: `uv run ruff check rag tests pyproject.toml --output-format=concise && uv run mypy rag tests`
Expected: Ruff and mypy both fail with the known baseline.

- [ ] **Step 2: Apply auto-fixable Ruff changes**

Run: `uv run ruff check rag tests pyproject.toml --fix`
Expected: Import ordering and other fixable issues are rewritten in place.

- [ ] **Step 3: Apply formatter pass**

Run: `uv run ruff format rag tests`
Expected: Long lines and formatting-compatible style issues are reduced.

- [ ] **Step 4: Re-run Ruff**

Run: `uv run ruff check rag tests pyproject.toml --output-format=concise`
Expected: Remaining Ruff failures are only non-auto-fix issues that require manual edits.

### Task 2: Fix Core Ingest And Extraction Typing

**Files:**
- Modify: `rag/ingest/extract.py`
- Modify: `rag/ingest/ingest.py`
- Modify: `rag/storage/_search/in_memory_vector_repo.py`
- Test: `tests/core/test_ingest_pipeline.py`
- Test: `tests/service/test_document_processing_pipeline.py`

- [ ] **Step 1: Fix local variable/type-annotation mismatches in extraction**

Target: wrong inferred variable reuse, missing annotations, and relation/entity mixups in `rag/ingest/extract.py`.

- [ ] **Step 2: Tighten sequence parameter types in ingest helpers**

Target: replace broad `list[object]` style signatures with concrete `Sequence[MergedEntity]`, `Sequence[MergedRelation]`, and provider protocol types in `rag/ingest/ingest.py`.

- [ ] **Step 3: Fix the in-memory vector repo key typing**

Target: align `_VectorRecord` key shape with the actual `(item_id, embedding_space, item_kind)` indexing pattern.

- [ ] **Step 4: Run focused verification**

Run: `uv run mypy rag/ingest/extract.py rag/ingest/ingest.py rag/storage/_search/in_memory_vector_repo.py`
Expected: These files are type-clean.

### Task 3: Fix Query And Engine Typing

**Files:**
- Modify: `rag/query/context.py`
- Modify: `rag/query/graph.py`
- Modify: `rag/query/_retrieval/branch_retrievers.py`
- Modify: `rag/query/retrieve.py`
- Modify: `rag/engine.py`
- Test: `tests/core/test_context_pipeline.py`
- Test: `tests/core/test_query_modes.py`
- Test: `tests/service/test_retrieval_service.py`

- [ ] **Step 1: Add missing imports and concrete container element types**

Target: `Sequence` import gaps, `setdefault()` value typing, and other collection invariance problems.

- [ ] **Step 2: Remove stale casts and fix nullable plan/provider handling**

Target: `RetrievalPlanBuilder | None`, redundant casts, provider attempt typing, and `engine.py` callable signatures.

- [ ] **Step 3: Fix any remaining query-module Ruff failures**

Target: manual line wraps or unused imports left after formatter/autofix.

- [ ] **Step 4: Run focused verification**

Run: `uv run mypy rag/query rag/engine.py`
Expected: Query stack and engine are type-clean.

### Task 4: Fix Test Typing And Compatibility Friction

**Files:**
- Modify: `tests/service/test_policy_resolution_service.py`
- Modify: `tests/service/test_ingest_service_plain_text.py`
- Modify: `tests/service/test_ingest_service_pdf.py`
- Modify: `tests/service/test_ingest_service_web.py`
- Modify: `tests/service/test_document_processing_pipeline.py`

- [ ] **Step 1: Replace mutable set literals with `frozenset(...)` where required**

Target: `AccessPolicy` constructor expectations in tests.

- [ ] **Step 2: Handle untyped third-party imports explicitly in tests**

Target: `fitz` imports and any test-only missing-typing issues.

- [ ] **Step 3: Fix minor test-side type mismatches**

Target: nullable assertions, fake protocol conformance, and callable typing in fixtures.

- [ ] **Step 4: Run focused verification**

Run: `uv run mypy tests/service`
Expected: Service tests are type-clean.

### Task 5: Final Full Verification

**Files:**
- Verify: `rag/`
- Verify: `tests/`
- Verify: `pyproject.toml`

- [ ] **Step 1: Run full Ruff**

Run: `uv run ruff check rag tests pyproject.toml`
Expected: PASS

- [ ] **Step 2: Run full mypy**

Run: `uv run mypy rag tests`
Expected: PASS

- [ ] **Step 3: Run full pytest**

Run: `uv run pytest -q`
Expected: PASS
