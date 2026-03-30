# RAG Core Parity Design

Status: Approved direction, implementation in progress
Date: 2026-03-30
Author: Codex

## 1. Summary

This document defines the next-stage target for `rag/`:

- stop optimizing for a personal-knowledge-platform surface
- harden `rag/` as a compact but real core engine
- close the highest-value capability gaps against LightRAG and RAG-Anything
- avoid rebuilding a second framework around the core

The selected direction is:

- keep the current `rag/` package as the only product surface
- absorb missing capabilities into existing modules
- prefer extending existing files over adding new ones
- ship real library-grade features, not demo wrappers

## 2. Goals

### 2.1 Primary goals

- Reach practical LightRAG core parity on query surface, ingest surface, and custom KG operations.
- Reach practical RAG-Anything core parity on multimodal ingest surface and cross-modal retrieval orchestration.
- Preserve a small, readable codebase centered on `rag/engine.py`, `rag/ingest/`, `rag/query/`, `rag/document/`, and `rag/storage/`.
- Keep `ruff`, `mypy`, and `pytest` passing while these capabilities are added.

### 2.2 Required capability targets

#### LightRAG parity targets

- Add `bypass` query mode.
- Add batch insert support through the public library API.
- Add custom KG operations through the public library API.
- Support custom graph injection without forcing re-ingest of source documents.

#### RAG-Anything parity targets

- Add direct content list insertion through the public library API.
- Expand Office-family parsing beyond `docx` to include at least `pptx` and `xlsx`.
- Strengthen multimodal retrieval so text, structure, metadata, graph, and modality-specific evidence participate in one orchestration path.
- Allow optional VLM-backed multimodal enrichment when a capable provider is available, without making VLM mandatory.

## 3. Non-goals

- Reintroducing `runtime`, `ui`, `api`, `deep research`, or any second orchestration layer.
- Rebuilding the old `Types -> Config -> Repo -> Service -> Runtime -> UI` architecture.
- Adding a broad plugin system, job queue, or remote service layer in this phase.
- Chasing every peripheral feature from upstream projects such as observability integrations before core parity is reached.

## 4. Hard constraints

### 4.1 Structure constraint

The implementation must stay inside the current core package layout:

- `rag/engine.py`
- `rag/cli.py`
- `rag/ingest/`
- `rag/query/`
- `rag/document/`
- `rag/storage/`
- existing schema modules

New source files are disallowed unless an existing file boundary becomes technically unworkable. The default path is to extend current files.

### 4.2 Public surface constraint

The primary surface remains the library API. CLI growth must stay minimal and should only expose operations that materially help inspect or use the core engine.

### 4.3 Quality constraint

Every phase must preserve:

- `uv run ruff check .`
- `uv run mypy .`
- `uv run pytest -q`

## 5. Current gaps

### 5.1 Query gaps

Current query modes are only:

- `naive`
- `local`
- `global`
- `hybrid`
- `mix`

There is no `bypass` mode, and the public query API does not expose a minimal direct-answer path.

### 5.2 Ingest gaps

Current ingest supports:

- plain text
- markdown
- pdf
- image
- docx
- web html/url

It does not support:

- batch insert as a first-class public API
- direct content list insertion
- `pptx`
- `xlsx`

### 5.3 Graph gaps

The internal graph layer already supports node and edge persistence, alias lookup, candidate-edge promotion, and evidence binding, but these are not exposed as a complete custom KG surface through `RAG`.

### 5.4 Multimodal gaps

Current multimodal support is useful but still shallow:

- special chunks exist
- multimodal nodes and relations exist
- graph expansion exists

Missing pieces are:

- broader document-family coverage
- direct multimodal content injection
- stronger query-time modality routing
- optional VLM enrichment that feeds the same evidence pipeline

## 6. Selected architecture

The selected design is a single compact RAG core with four upgraded surfaces:

1. Query surface
2. Ingest surface
3. Knowledge-graph surface
4. Multimodal orchestration surface

The critical decision is to add these capabilities by strengthening the existing core pipeline, not by introducing a second orchestration tier.

## 7. Query surface design

### 7.1 Add `bypass` mode

`bypass` is a first-class `QueryMode`.

Its behavior is:

- skip graph expansion
- skip web retrieval
- skip planner-added specialty branches
- use a minimal direct retrieval path intended for low-latency answering
- preserve rerank, evidence assembly, citations, and grounded answer generation

`bypass` is not a raw LLM shortcut. It still remains evidence-backed.

### 7.2 Keep orchestration unified

The existing `RetrievalPlanBuilder`, `BranchRetrieverRegistry`, and `QueryPipeline` remain the orchestration core.

The implementation should:

- extend `QueryMode`
- teach `RetrievalPlanBuilder` how `bypass` maps to branches and disabled features
- avoid adding a separate query runtime or alternate pipeline class

### 7.3 Upgrade multimodal routing inside the existing planner

The query planner should become more modality-aware without becoming a full deep-research planner.

It should:

- elevate modality-specific branches when query understanding indicates tables, figures, captions, OCR regions, formulas, or structure
- prefer source-type and section-aware evidence when the query is localized
- let multimodal evidence compete in the same fusion and rerank path

## 8. Ingest surface design

### 8.1 Batch ingest

Batch ingest is added as a public library feature on `RAG` and reused internally from `IngestPipeline`.

Required behavior:

- accept a sequence of ingest requests
- preserve per-item success or failure results
- support fail-fast and continue-on-error modes
- reuse the current normalization, deduplication, processing, and indexing pipeline

The implementation should avoid parallel worker frameworks in this phase. Sequential batch execution with deterministic result envelopes is sufficient for core parity.

### 8.2 Direct content list insertion

Direct content list insertion is a normalization feature, not a second ingest framework.

The public API should accept a heterogeneous sequence of content items such as:

- raw plain text
- markdown
- html
- browser clip html
- file paths
- raw bytes with declared source type

These inputs are normalized into ordinary ingest requests and pushed through the same ingest pipeline.

This keeps:

- deduplication
- chunking
- graph extraction
- vector indexing
- metadata persistence

under one code path.

### 8.3 Office-family expansion

`DoclingParserRepo` becomes the single Office-family parser entry inside the current architecture.

Required extensions:

- add `pptx`
- add `xlsx`

The parser should map these sources into the same `ParsedDocument` contract already used by the ingest pipeline.

The design explicitly prefers extending `DoclingParserRepo` over creating separate parser trees.

## 9. Custom KG design

### 9.1 Public graph operations

Expose a clean custom KG surface through `RAG` backed by the existing `GraphStore` and `SQLiteGraphRepo`.

Required operations:

- upsert node
- upsert edge
- bind node evidence
- bind edge evidence
- list nodes
- list edges
- get node
- get edge
- delete node
- delete edge
- insert custom KG payloads in batch

### 9.2 Evidence rule

Custom graph insertion must preserve the project’s evidence-first rule:

- edges must keep evidence chunk ids
- nodes may be created without source ingest only if metadata marks them as custom KG nodes
- custom KG records without source-document evidence must remain identifiable in metadata

### 9.3 No separate graph service layer

The current `GraphStore` already owns most persistence semantics. The design is to complete that surface, not wrap it in a new subsystem.

## 10. Multimodal orchestration design

### 10.1 Unified evidence competition

Multimodal retrieval stays inside the current retrieval pipeline:

- branch retrieval
- fusion
- rerank
- evidence assembly
- answer generation

There must not be a second multimodal query engine.

### 10.2 Modality-aware candidate shaping

The existing special-chunk and graph machinery should be strengthened so that:

- table-heavy queries lift table and formula evidence
- figure-heavy queries lift figure, caption, image-summary, and OCR-region evidence
- structure-heavy queries lift section and metadata evidence
- graph expansion can contribute modality-linked chunks, not just textual neighbors

### 10.3 Optional VLM enrichment

If an attached provider can do vision or multimodal description, the parser and retrieval stack may use it to enrich:

- figures
- tables
- slides
- screenshots
- spreadsheet regions

This enrichment must be optional and degrade cleanly to the current OCR/text-first flow.

## 11. File-level implementation boundaries

### 11.1 Files to extend

- `rag/query/query.py`
  - add `bypass`
  - update query option typing

- `rag/query/_retrieval/mode_planner.py`
  - define branch policy for `bypass`
  - strengthen modality-aware branch selection

- `rag/query/retrieve.py`
  - preserve one orchestration path while supporting `bypass`
  - wire stronger multimodal branch behavior

- `rag/engine.py`
  - add batch ingest API
  - add direct content list insertion API
  - add custom KG public methods

- `rag/ingest/ingest.py`
  - normalize direct content items into ingest requests
  - implement batch result envelopes
  - route new source types through existing parsing flow

- `rag/document/_parse/docling_parser_repo.py`
  - add `pptx` and `xlsx`
  - normalize their Docling output into the existing parsed-document contract
  - optionally enrich multimodal items when a capable provider is available

- `rag/schema/_types/content.py`
  - extend `SourceType` where required for new file families

- `rag/storage/graph_store.py`
  - complete public graph CRUD semantics

- `rag/storage/_graph/sqlite_graph_repo.py`
  - add node and edge deletion/update helpers needed by custom KG ops

- `rag/cli.py`
  - keep CLI minimal
  - only expose the subset of new operations that are useful without bloating the surface

### 11.2 Files not to reintroduce

- no new runtime package
- no api package
- no workbench package
- no deep-research package

## 12. Delivery phases

### Phase 1: LightRAG core parity

- add `bypass`
- add batch ingest
- add custom KG ops
- add tests for public API and retrieval behavior

### Phase 2: RAG-Anything ingest parity

- add direct content list insertion
- add `pptx`
- add `xlsx`
- add tests for Office and mixed content ingestion

### Phase 3: Multimodal orchestration hardening

- improve modality-aware planning and fusion
- add optional VLM enrichment path
- add end-to-end multimodal retrieval tests

## 13. Testing strategy

Each phase must add or update:

- public API tests in `tests/core/`
- ingest and parser tests in `tests/service/`
- retrieval tests in `tests/service/`
- repository tests for graph operations in `tests/repo/`
- CLI tests only where a CLI surface is intentionally added

The test bar is:

- no demo-only tests
- no snapshot-only confidence theater
- behavior-focused tests that prove parity-critical features actually work

## 14. Acceptance criteria

The work is complete when all of the following are true:

- `rag` exposes `bypass`, batch ingest, direct content list insertion, and custom KG operations through the library API
- `pptx` and `xlsx` ingest through the same core pipeline as existing source types
- multimodal retrieval remains one unified orchestration path with stronger modality awareness
- the codebase remains centered on the current `rag/` structure without reintroducing old layers
- `uv run ruff check .`, `uv run mypy .`, and `uv run pytest -q` all pass
