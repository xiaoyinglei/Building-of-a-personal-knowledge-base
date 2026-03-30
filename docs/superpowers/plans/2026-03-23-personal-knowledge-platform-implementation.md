# Personal Knowledge Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reliability-first personal knowledge platform with strict `Types -> Config -> Repo -> Service -> Runtime -> UI` layering, cloud-first execution, local fallback, hybrid retrieval, bounded deep research, and semi-automatic artifact preservation.

**Architecture:** The implementation uses a Python monorepo managed by `uv`, with a strict layered package layout under `src/rag/`. Local bootstrap uses SQLite metadata plus FTS5, filesystem object storage, a pluggable vector repository, and provider adapters for cloud and local models. Fast Path and Deep Path runtimes share domain services but remain separate execution modes with different routing budgets and safeguards.

**Tech Stack:** Python 3.12, `uv`, FastAPI, Pydantic v2, SQLAlchemy, SQLite/FTS5, HTTPX, pytest, ruff, mypy, import-linter, PyMuPDF, markdown-it-py, BeautifulSoup/trafilatura, Pillow, optional OCR and model provider adapters.

---

## Planned File Structure

### Repository root

- Create: `/Users/leixiaoying/LLM/RAG学习/.gitignore`
- Create: `/Users/leixiaoying/LLM/RAG学习/README.md`
- Create: `/Users/leixiaoying/LLM/RAG学习/pyproject.toml`
- Create: `/Users/leixiaoying/LLM/RAG学习/importlinter.ini`
- Create: `/Users/leixiaoying/LLM/RAG学习/.python-version`
- Create: `/Users/leixiaoying/LLM/RAG学习/.env.example`
- Create: `/Users/leixiaoying/LLM/RAG学习/.github/workflows/ci.yml`

### Source tree

- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/types/`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/config/`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/runtime/`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/ui/`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/bootstrap.py`

### Test tree

- Create: `/Users/leixiaoying/LLM/RAG学习/tests/types/`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/config/`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/repo/`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/service/`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/runtime/`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/ui/`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/integration/`

### Sample data and docs

- Create: `/Users/leixiaoying/LLM/RAG学习/data/samples/`
- Create: `/Users/leixiaoying/LLM/RAG学习/docs/architecture/`

## Spec-to-Task Traceability

- Layer dependency rule and CI enforcement: Tasks 1, 2, 6, 8
- Canonical `AccessPolicy` and routing enums: Tasks 2, 4, 5, 6
- Graph contracts and evidence-linked graph rules: Tasks 2, 3, 5
- Supported ingest sources (`PDF`, `Markdown`, `images`, `web pages`, `plain text`): Tasks 4, 7
- Fast Path and Deep Path with bounded iteration and recursion: Tasks 5, 6, 7
- Web-search gate and external evidence separation: Tasks 3, 5, 6, 7
- Failure and degradation ladder: Tasks 3, 6, 7
- Artifact preservation, approval, re-indexing, and reuse: Tasks 5, 6, 7, 8
- Topic-page and artifact-oriented durable knowledge accumulation: Tasks 5, 6, 7, 8

## Task 1: Bootstrap Repository, uv Project, and Guardrails

**Files:**
- Create: `/Users/leixiaoying/LLM/RAG学习/.gitignore`
- Create: `/Users/leixiaoying/LLM/RAG学习/README.md`
- Create: `/Users/leixiaoying/LLM/RAG学习/pyproject.toml`
- Create: `/Users/leixiaoying/LLM/RAG学习/importlinter.ini`
- Create: `/Users/leixiaoying/LLM/RAG学习/.python-version`
- Create: `/Users/leixiaoying/LLM/RAG学习/.env.example`
- Create: `/Users/leixiaoying/LLM/RAG学习/.github/workflows/ci.yml`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_project_bootstrap.py`

- [ ] **Step 1: Initialize git repository and `uv` project metadata**

Run:

```bash
git init
uv init --python 3.12 .
mkdir -p src/rag tests
```

Expected:
- `.git/` exists
- `pyproject.toml` exists

- [ ] **Step 2: Write the failing project bootstrap test**

Create:

```python
from pathlib import Path


def test_bootstrap_files_exist() -> None:
    required = [
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "importlinter.ini",
        ".env.example",
    ]
    for relative in required:
        assert Path(relative).exists(), relative
```

- [ ] **Step 3: Add test and lint dependencies, then run bootstrap test to verify failure**

Run:

```bash
uv add --dev pytest pytest-asyncio ruff mypy import-linter
uv run pytest tests/integration/test_project_bootstrap.py -v
```

Expected:
- FAIL because one or more files are missing

- [ ] **Step 4: Implement root project files and guardrails**

Add:
- `.gitignore` covering `.venv`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.coverage`, `__pycache__`, `.superpowers`, `data/runtime`, local DB files
- `.python-version` pinned to `3.12`
- `pyproject.toml` with runtime, dev, lint, type-check, and test dependencies
- `importlinter.ini` with the layer contract
- `.github/workflows/ci.yml` running `uv sync --all-extras`, `pytest`, `ruff check`, `mypy src`, and `lint-imports`
- `scripts/check_repo_only_imports.py` enforcing that provider SDK imports appear only under `src/rag/repo/**`
- `README.md` with local bootstrap and execution overview
- `.env.example` with provider, storage, and runtime settings

- [ ] **Step 5: Run bootstrap verification**

Run:

```bash
uv sync --all-extras
uv run pytest tests/integration/test_project_bootstrap.py -v
uv run lint-imports
uv run python -m scripts.check_repo_only_imports
```

Expected:
- bootstrap test PASS
- import-linter reports contract is satisfied or no import graph violations yet

- [ ] **Step 6: Commit**

Run:

```bash
git add .gitignore README.md pyproject.toml importlinter.ini .python-version .env.example .github/workflows/ci.yml scripts/check_repo_only_imports.py tests/integration/test_project_bootstrap.py
git commit -m "chore: bootstrap uv project and layering guardrails"
```

## Task 2: Define Types and Config Contracts

**Files:**
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/types/__init__.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/types/access.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/types/content.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/types/query.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/types/artifact.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/types/envelope.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/config/__init__.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/config/settings.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/config/policies.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/types/test_access_policy.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/types/test_content_contracts.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/types/test_query_routing_enums.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/config/test_settings.py`

- [ ] **Step 1: Write failing type and config tests**

Cover:
- `AccessPolicy` narrowing behavior
- canonical required field sets for `Source`, `Document`, `Segment`, `Chunk`, and `KnowledgeArtifact`
- `TaskType` and `ComplexityLevel` enums
- request-level execution policy fields: `effective_access_policy`, `execution_location_preference`, `latency_budget`, `cost_budget`, `fallback_allowed`, `source_scope`
- canonical `GraphNode` and `GraphEdge` contracts with `evidence_chunk_ids`
- artifact statuses
- Pydantic settings parsing from `.env`

- [ ] **Step 2: Run failing tests**

Run:

```bash
uv run pytest tests/types tests/config -v
```

Expected:
- FAIL because types and config modules do not exist yet

- [ ] **Step 3: Implement pure type contracts**

Implement:
- `AccessPolicy`
- `Source`, `Document`, `Segment`, `Chunk` with all required spec fields
- `GraphNode`, `GraphEdge`
- `KnowledgeArtifact` with all required spec fields
- `TaskType`, `ComplexityLevel`, `ExecutionLocationPreference`
- `ExecutionPolicy` carrying `effective_access_policy`, `execution_location_preference`, `latency_budget`, `cost_budget`, `fallback_allowed`, and `source_scope`
- result envelopes for query responses and preservation suggestions

- [ ] **Step 4: Implement config layer**

Implement:
- application settings
- routing thresholds
- policy defaults
- provider configuration shapes
- normative defaults from the spec:
  - Fast Path sufficiency thresholds
  - Deep Path sufficiency thresholds
  - `max retrieval rounds = 4`
  - `max recursive depth = 2`
  - `default wall-clock budget = 180s`
  - `default synthesis retry count = 1`

- [ ] **Step 5: Run tests and import contract**

Run:

```bash
uv run pytest tests/types tests/config -v
uv run lint-imports
```

Expected:
- PASS

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rag/types src/rag/config tests/types tests/config
git commit -m "feat: add core type and config contracts"
```

## Task 3: Build Repo Layer Foundations

**Files:**
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/__init__.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/interfaces.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/storage/file_object_store.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/storage/sqlite_metadata_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/search/sqlite_fts_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/search/in_memory_vector_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/graph/sqlite_graph_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/search/web_search_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/vision/ocr_vision_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/models/openai_provider_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/models/ollama_provider_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/models/fallback_embedding_repo.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/repo/test_sqlite_metadata_repo.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/repo/test_sqlite_fts_repo.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/repo/test_in_memory_vector_repo.py`

- [ ] **Step 1: Write failing repo tests**

Cover:
- persisting sources, documents, segments, chunks, artifacts
- FTS indexing and text search
- vector insert and similarity search

- [ ] **Step 2: Run failing repo tests**

Run:

```bash
uv run pytest tests/repo -v
```

Expected:
- FAIL because repo implementations do not exist yet

- [ ] **Step 3: Implement repository interfaces and local adapters**

Implement:
- filesystem object storage
- SQLite metadata/artifact store
- SQLite FTS5 search repo
- simple vector repo for local bootstrap
- graph repo for candidate edges and approved graph edges
- web-search repo abstraction with deterministic fake adapter for tests and pluggable external adapter for runtime use
- `OcrVisionRepo` interface with deterministic local test adapter and pluggable runtime adapter

- [ ] **Step 4: Implement provider adapters**

Implement:
- OpenAI-compatible cloud provider adapter
- Ollama-compatible local provider adapter
- deterministic local embedding fallback for tests

- [ ] **Step 5: Run repo tests**

Run:

```bash
uv run pytest tests/repo -v
uv run lint-imports
uv run python -m scripts.check_repo_only_imports
```

Expected:
- PASS

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rag/repo tests/repo
git commit -m "feat: add repository adapters for storage search and models"
```

## Task 4: Implement Parsing and Ingest Services

**Files:**
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/parse/pdf_parser_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/parse/markdown_parser_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/parse/plain_text_parser_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/parse/image_parser_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/parse/web_parser_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/ingest_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/chunking_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/policy_resolution_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/toc_service.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_ingest_service_markdown.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_ingest_service_pdf.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_ingest_service_plain_text.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_ingest_service_image.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_ingest_service_web.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_policy_resolution_service.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_reingest_deduplication.py`

- [ ] **Step 1: Write failing ingest service tests**

Cover:
- Markdown heading tree becomes segments with TOC paths
- PDF text pages become anchored segments and chunks
- plain-text ingest produces normalized segments, stable anchors, and inherited policy
- image ingest stores both `visible_text` and `visual_semantics`
- web-page ingest stores extracted headings and article text
- `AccessPolicy` narrows correctly from source to chunk
- duplicate ingest is detected by content hash and does not create duplicate active documents
- incremental re-ingest preserves stable anchors where content structure is unchanged

- [ ] **Step 2: Run failing ingest tests**

Run:

```bash
uv run pytest tests/service/test_ingest_service_markdown.py tests/service/test_ingest_service_pdf.py tests/service/test_ingest_service_plain_text.py tests/service/test_ingest_service_image.py tests/service/test_ingest_service_web.py tests/service/test_policy_resolution_service.py tests/service/test_reingest_deduplication.py -v
```

Expected:
- FAIL

- [ ] **Step 3: Implement parse repositories**

Implement:
- Markdown parser preserving heading hierarchy
- PDF parser extracting pages and text blocks
- plain-text parser producing a synthetic root heading and stable anchors
- image parser returning OCR text plus deterministic v1 `visual_semantics` derived from image metadata, OCR regions, and simple layout heuristics
- web parser extracting article text and headings

- [ ] **Step 4: Implement ingest services**

Implement:
- source normalization
- TOC recovery or inference
- structure-first chunking
- deduplication via source hash and version tracking
- incremental re-ingest with stable anchors for unchanged structures
- policy resolution and inheritance
- source/document/chunk persistence
- indexing into metadata, FTS, vector, and graph candidate stores as part of ingest completion

- [ ] **Step 5: Run ingest tests**

Run:

```bash
uv run pytest tests/service/test_ingest_service_markdown.py tests/service/test_ingest_service_pdf.py tests/service/test_ingest_service_plain_text.py tests/service/test_ingest_service_image.py tests/service/test_ingest_service_web.py tests/service/test_policy_resolution_service.py tests/service/test_reingest_deduplication.py -v
```

Expected:
- PASS

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rag/repo/parse src/rag/service tests/service
git commit -m "feat: add ingest parsing toc and policy services"
```

## Task 5: Implement Retrieval, Fusion, and Evidence Services

**Files:**
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/retrieval_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/evidence_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/graph_expansion_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/artifact_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/routing_service.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_retrieval_service.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_routing_service.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_artifact_service.py`

- [ ] **Step 1: Write failing retrieval and routing tests**

Cover:
- task type and complexity routing
- hybrid retrieval with RRF fusion
- reranking is mandatory in both Fast Path and Deep Path
- rerank gating
- self-check gates
- selected chunks constrain embedding, reranking, answer-generation location, and Fast/Deep eligibility according to `effective_access_policy`
- selected chunks constrain Deep Path summarization and artifact-generation location according to `effective_access_policy`
- `source_scope` constrains retrieval candidates and downstream runtime search expansion
- web-search gate allows external retrieval only when policy permits it
- external evidence is labeled separately from internal evidence in the assembled response envelope
- graph expansion only after non-graph evidence exists
- preservation suggestions for reusable outputs
- artifact lifecycle marks existing artifacts `stale` or `conflicted` instead of silently overwriting them

- [ ] **Step 2: Run failing retrieval tests**

Run:

```bash
uv run pytest tests/service/test_retrieval_service.py tests/service/test_routing_service.py tests/service/test_artifact_service.py -v
```

Expected:
- FAIL

- [ ] **Step 3: Implement routing and retrieval services**

Implement:
- routing normalization
- sequential, conditional, and branch modular retrieval
- RRF fusion
- mandatory rerank callout in both Fast Path and Deep Path retrieval flows
- self-check gate evaluation
- query-time `AccessPolicy` compatibility checks for embedding, reranking, answer-generation location, summarization location, artifact-generation location, and runtime eligibility
- `source_scope` enforcement in retrieval candidate selection and runtime search expansion

- [ ] **Step 4: Implement artifact and graph expansion services**

Implement:
- preservation suggestion scoring
- concrete artifact construction for `document_summary`, `section_summary`, `topic_page`, `comparison_page`, `timeline`, and `open_question_page`
- stable topic-page structure required by the spec
- anti-pollution artifact lifecycle rules for `stale` and `conflicted` states
- graph expansion with evidence prerequisite

- [ ] **Step 5: Run retrieval tests**

Run:

```bash
uv run pytest tests/service/test_retrieval_service.py tests/service/test_routing_service.py tests/service/test_artifact_service.py -v
```

Expected:
- PASS

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rag/service tests/service
git commit -m "feat: add retrieval routing evidence and artifact services"
```

## Task 6: Implement Fast Path and Deep Path Runtimes

**Files:**
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/runtime/__init__.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/runtime/container.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/bootstrap.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/runtime/ingest_runtime.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/runtime/fast_query_runtime.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/runtime/deep_research_runtime.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/runtime/session_runtime.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/runtime/artifact_promotion_runtime.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/runtime/test_ingest_runtime.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/runtime/test_fast_query_runtime.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/runtime/test_deep_research_runtime.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/runtime/test_degradation_ladder.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/runtime/test_access_policy_runtime_matrix.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/runtime/test_session_runtime.py`

- [ ] **Step 1: Write failing runtime tests**

Cover:
- `IngestRuntime` orchestrates ingest requests and persists searchable results only through runtime facades
- Fast Path executes one retrieval cycle and returns evidence-backed output
- Fast Path escalates to Deep Path when evidence is insufficient, conflicts exist, or claim-citation alignment fails
- Deep Path performs bounded iterative retrieval
- Deep Path decomposes a research query into sub-questions
- Deep Path builds an evidence matrix grouped by source and claim
- Deep Path stops at configured round and time limits
- Deep Path enforces token budget in addition to depth, round, and wall-clock limits
- runtime respects `AccessPolicy` across `cloud_allowed`, `local_preferred`, and `local_required` inputs
- runtime respects `AccessPolicy` when choosing synthesis and artifact-generation location
- `SessionRuntime` owns session-scoped working memory for sub-questions and evidence matrices
- claim-citation support check is mandatory before returning answers
- artifact promotion requires approval state transition
- cloud-provider failure triggers backup cloud, then local fallback, then retrieval-only evidence packet
- approved artifact promotion re-indexes the artifact for future retrieval

- [ ] **Step 2: Run failing runtime tests**

Run:

```bash
uv run pytest tests/runtime -v
```

Expected:
- FAIL

- [ ] **Step 3: Implement runtime container and orchestration**

Implement:
- protocol-based runtime container definitions
- `src/rag/runtime/container.py` as the Runtime-owned composition root, accepting injected factories and service instances without importing concrete Repo adapters directly
- `src/rag/bootstrap.py` as a thin launcher that instantiates concrete repos/services and delegates to the Runtime composition root
- `IngestRuntime` for runtime-owned ingest orchestration used by UI facades
- Fast Path runtime
- Deep Path runtime with bounded iteration and recursion
- `SessionRuntime` for session-scoped working memory and orchestration state
- sub-question decomposition and evidence-matrix construction
- token-budget enforcement for recursive research
- Fast Path to Deep Path escalation logic
- mandatory claim-citation support check before returning results
- artifact promotion runtime
- failure and degradation ladder across backup cloud, local fallback, and retrieval-only packet
- artifact re-indexing after approval

- [ ] **Step 4: Run runtime tests**

Run:

```bash
uv run pytest tests/runtime -v
uv run lint-imports
```

Expected:
- PASS

- [ ] **Step 5: Commit**

Run:

```bash
git add src/rag/runtime tests/runtime
git commit -m "feat: add fast and deep query runtimes"
```

## Task 7: Implement UI Layer and End-to-End Vertical Slice

**Files:**
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/ui/__init__.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/ui/api/app.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/ui/api/routes/health.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/ui/api/routes/ingest.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/ui/api/routes/query.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/ui/api/routes/artifacts.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/ui/cli.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/ui/test_api_health.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/ui/test_api_ingest_query.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/ui/test_cli.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_markdown_query_flow.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_plain_text_conflict_flow.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_artifact_promotion_flow.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_pdf_ingest_query_flow.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_image_ingest_flow.py`
- Test: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_web_ingest_flow.py`

- [ ] **Step 1: Write failing UI and integration tests**

Cover:
- API boots successfully
- ingest endpoint stores Markdown sample
- query endpoint returns `Conclusion`, `Evidence`, `Differences or conflicts`, `Uncertainty`, and preservation suggestion
- CLI can ingest and query
- plain-text conflicting sources surface explicit conflict output instead of forced unification
- user can approve a preservation suggestion through API or CLI and the approved artifact becomes retrievable later
- PDF, image, and web-page ingest work through runtime and UI facades end to end

- [ ] **Step 2: Run failing UI tests**

Run:

```bash
uv run pytest tests/ui tests/integration/test_markdown_query_flow.py tests/integration/test_plain_text_conflict_flow.py tests/integration/test_artifact_promotion_flow.py tests/integration/test_pdf_ingest_query_flow.py tests/integration/test_image_ingest_flow.py tests/integration/test_web_ingest_flow.py -v
```

Expected:
- FAIL

- [ ] **Step 3: Implement UI facades**

Implement:
- FastAPI application
- ingest/query/artifact routes
- CLI entrypoints for bootstrap, ingest, fast query, deep query

- [ ] **Step 4: Add sample Markdown and plain-text data plus end-to-end fixtures**

Create:
- `/Users/leixiaoying/LLM/RAG学习/data/samples/agent-rag-overview.md`
- `/Users/leixiaoying/LLM/RAG学习/data/samples/conflict-a.txt`
- `/Users/leixiaoying/LLM/RAG学习/data/samples/conflict-b.txt`
- `/Users/leixiaoying/LLM/RAG学习/data/samples/sample-report.pdf`
- `/Users/leixiaoying/LLM/RAG学习/data/samples/sample-ui.png`
- `/Users/leixiaoying/LLM/RAG学习/data/samples/sample-article.html`

- [ ] **Step 5: Run UI and integration tests**

Run:

```bash
uv run pytest tests/ui tests/integration/test_markdown_query_flow.py tests/integration/test_plain_text_conflict_flow.py tests/integration/test_artifact_promotion_flow.py tests/integration/test_pdf_ingest_query_flow.py tests/integration/test_image_ingest_flow.py tests/integration/test_web_ingest_flow.py -v
```

Expected:
- PASS

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rag/ui data/samples tests/ui tests/integration/test_markdown_query_flow.py tests/integration/test_plain_text_conflict_flow.py tests/integration/test_artifact_promotion_flow.py tests/integration/test_pdf_ingest_query_flow.py tests/integration/test_image_ingest_flow.py tests/integration/test_web_ingest_flow.py
git commit -m "feat: add api cli and end-to-end query flow"
```

## Task 8: Reliability Instrumentation and Evaluation Harness

**Files:**
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/types/telemetry.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/repo/telemetry/local_event_repo.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/src/rag/service/telemetry_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/service/test_telemetry_service.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_reliability_events.py`
- Create: `/Users/leixiaoying/LLM/RAG学习/tests/integration/test_evaluation_metrics.py`

- [ ] **Step 1: Write failing telemetry tests**

Cover:
- retrieval branch usage events
- RRF fusion events
- rerank effectiveness events
- graph expansion usage events
- Fast Path to Deep Path escalation events
- claim-citation validation failures
- local fallback frequency events
- preservation suggestion and approval events

- [ ] **Step 2: Run failing telemetry tests**

Run:

```bash
uv run pytest tests/service/test_telemetry_service.py tests/integration/test_reliability_events.py -v
```

Expected:
- FAIL

- [ ] **Step 3: Implement local telemetry and evaluation primitives**

Implement:
- typed telemetry event contracts
- local event repository
- telemetry service hooks used by retrieval and runtime layers
- evaluation helpers that compute citation precision, evidence sufficiency rate, conflict detection coverage, simple-query latency, deep-query completion quality, and preservation usefulness from test runs

- [ ] **Step 4: Run telemetry tests**

Run:

```bash
uv run pytest tests/service/test_telemetry_service.py tests/integration/test_reliability_events.py tests/integration/test_evaluation_metrics.py -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

Run:

```bash
git add src/rag/types/telemetry.py src/rag/repo/telemetry src/rag/service/telemetry_service.py tests/service/test_telemetry_service.py tests/integration/test_reliability_events.py tests/integration/test_evaluation_metrics.py
git commit -m "feat: add reliability telemetry and evaluation hooks"
```

## Task 9: Documentation, Local Verification, and Optional Remote Publishing

**Files:**
- Modify: `/Users/leixiaoying/LLM/RAG学习/README.md`
- Create: `/Users/leixiaoying/LLM/RAG学习/docs/architecture/layering.md`
- Create: `/Users/leixiaoying/LLM/RAG学习/docs/architecture/runtime-facades.md`
- Create: `/Users/leixiaoying/LLM/RAG学习/docs/architecture/memory-scope.md`

- [ ] **Step 1: Document bootstrap, layering, and runtime facades**

Add:
- local bootstrap instructions
- provider setup instructions
- layering contract explanation
- API and CLI examples
- memory scope decision: `Working Memory` ephemeral in runtime, `User Memory` from configuration and user profile storage, `Semantic Memory` from approved artifacts, `Operational Memory` from telemetry, `Episodic Memory` deferred beyond v1 except for artifact-backed research summaries

- [ ] **Step 2: Verify full project**

Run:

```bash
uv run pytest -v
uv run ruff check .
uv run mypy src
uv run lint-imports
```

Expected:
- all checks PASS

- [ ] **Step 3: Create GitHub repository and publish**

This step is implementation-external and depends on GitHub authentication being available in the environment. It must not block code completion or local verification.

Run one of:

```bash
gh repo create personal-knowledge-platform-design --private --source . --remote origin --push
```

or, if repository already exists:

```bash
git remote add origin <existing-remote>
git push -u origin main
```

- [ ] **Step 4: Commit documentation update and push only if remote/auth are available**

Run:

```bash
git add README.md docs/architecture
git commit -m "docs: add bootstrap architecture and runtime guides"
git push || true
```

- [ ] **Step 5: Run final traceability checklist**

Confirm explicitly:
- all required source types ingest successfully: `PDF`, `Markdown`, `images`, `web pages`, `plain text`
- Deep Path includes sub-question decomposition, bounded iteration, bounded recursion, and evidence-matrix construction
- telemetry includes retrieval branch usage, RRF fusion, rerank effectiveness, graph expansion usage, Fast Path to Deep Path escalation, claim-citation failures, local fallback frequency, and preservation approvals
- approved artifacts are re-indexed and retrievable

## Parallelization Strategy

Only after Task 5 is complete and repo plus service interface shapes are frozen, split work into disjoint write domains:

- Worker A: `src/rag/repo/**`, `tests/repo/**`
- Worker B: `src/rag/service/ingest_service.py`, `src/rag/service/chunking_service.py`, `src/rag/service/policy_resolution_service.py`, `src/rag/service/toc_service.py`, matching tests
- Worker C: `src/rag/service/retrieval_service.py`, `src/rag/service/evidence_service.py`, `src/rag/service/graph_expansion_service.py`, `src/rag/service/artifact_service.py`, `src/rag/service/routing_service.py`, matching tests
- Worker D: `src/rag/runtime/**`, `src/rag/ui/**`, `tests/runtime/**`, `tests/ui/**`, `tests/integration/**`

Controller responsibilities:

- freeze repo/service/runtime handoff contracts before dispatching parallel workers
- maintain the layering contract
- review every worker patch for spec compliance first
- run integration verification after merging each domain
- push to GitHub after stable milestones

## Notes for execution

- Keep all imports layer-compliant at all times.
- Prefer deterministic local adapters in tests; avoid live network in test suites.
- Do not let GraphRAG enter the basic hot path.
- Do not let artifact promotion skip approval state.
- Preserve `AccessPolicy` inheritance and narrowing semantics in every ingest path.
