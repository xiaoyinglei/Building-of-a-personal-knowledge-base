# 2026-04-04 RAG Workbench Design

## Goal

Build a local browser workbench for the current `rag/` project so the user can:

- inspect ingested sources and documents from an existing storage root
- ingest, rebuild, and delete documents from the UI
- run queries against the real RAG pipeline
- inspect evidence, routing, token budgets, providers, and retrieval diagnostics
- switch between configured chat/embedding model profiles from the UI

The workbench is a local operator console for a single user, not a public web app.

## Constraints

- Keep the architecture light. Do not introduce Node, React, or a separate frontend toolchain.
- Reuse the existing `rag` library directly instead of building a second RAG interface layer.
- Keep the browser UI focused on real debugging and operation, not a demo chat shell.
- Do not hide degraded states. If the system falls back to retrieval-only or has no chat/rerank capability, the UI must show that explicitly.

## Non-Goals

- authentication, multi-user sessions, remote deployment
- a general-purpose REST platform for external clients
- graph visualization in the first version
- streaming generation in the first version
- background job orchestration

## Recommended Architecture

Use a single-process local workbench made of:

1. A thin local HTTP server inside `rag/`
2. A small JSON API for UI actions and data loading
3. A static browser client written in plain HTML/CSS/JS
4. A service layer that adapts the existing `RAG` engine and storage bundle into UI-facing DTOs

This keeps the implementation local-first, dependency-light, and aligned with the existing package.

## File Layout

Create a bounded `rag/workbench/` package instead of mixing HTTP and UI code into `engine.py` or `cli.py`.

Planned structure:

- `rag/workbench/server.py`
  - HTTP server entrypoint
  - static asset serving
  - JSON endpoint dispatch
- `rag/workbench/service.py`
  - workbench orchestration over `RAG`, `StorageConfig`, and `StorageBundle`
  - document directory assembly
  - model catalog assembly
  - ingest/rebuild/delete/query operations
- `rag/workbench/models.py`
  - UI-facing DTOs
  - stable response contracts for document tree, query run, model catalog, index summary
- `rag/workbench/static/index.html`
- `rag/workbench/static/app.css`
- `rag/workbench/static/app.js`

Update `rag/cli.py` to add a `workbench` command that launches the local server.

## Why This Approach

This is the cleanest fit for the current repo:

- the project already has a solid Python core and no frontend stack
- the UI needs rich inspection panels, which are awkward in server-rendered templates alone
- the UI is local-only, so a minimal in-process HTTP server is enough
- the service layer keeps transport concerns away from `engine.py`

## Core Data Flow

### Startup

`rag workbench --storage-root .rag`

1. Load `.env`
2. Build a model catalog from environment-backed provider profiles
3. Create a `WorkbenchService`
4. Serve static UI assets and JSON endpoints
5. Browser loads initial state from `/api/state`

### Query

1. User selects a model profile and query mode in the UI
2. Browser posts query request to `/api/query`
3. Workbench service builds a `RAG` instance using:
   - selected storage root
   - selected provider binding
   - current storage config
4. Real `RAG.query(...)` executes
5. Service transforms `RAGQueryResult` into a UI DTO
6. UI updates:
   - right pane chat transcript
   - center pane evidence cards
   - routing / budget / provider diagnostics

### Ingest / Rebuild / Delete

1. User acts from the left pane
2. Browser posts to `/api/ingest`, `/api/rebuild`, or `/api/delete`
3. Workbench service calls the real `RAG` methods
4. UI refreshes the document directory and index summary

## Model Catalog Design

The workbench needs explicit, inspectable model profiles instead of a fake dropdown.

Each profile will describe:

- profile id
- provider kind (`openai-compatible`, `ollama`, `local`)
- label for UI display
- chat model name
- embedding model name
- rerank capability
- capability health
- whether it is selectable for query

### Initial Profile Sources

The first version should auto-discover profiles from environment variables only.

Supported profile discovery:

- OpenAI-compatible profile
  - supports OpenAI and Gemini-through-Google-gateway style configs
  - accepts current legacy keys already used in the repo examples:
    - `PKP_OPENAI__API_KEY`
    - `PKP_OPENAI__BASE_URL`
    - `PKP_OPENAI__MODEL`
    - `PKP_OPENAI__EMBEDDING_MODEL`
  - also accepts the new `RAG_*` aliases if present
- Ollama profile
  - base URL + chat model + embedding model from env
- Reranker status from:
  - `RAG_RERANK_MODEL`
  - `RAG_RERANK_MODEL_PATH`

The UI model arrows switch between discovered profiles, not arbitrary free text.

## UI Layout

### Left Pane: Index And Document Directory

Purpose:

- browse indexed content
- operate on ingest lifecycle
- understand what exists in the current storage root

Must show:

- current storage root
- backend summary
- index summary counts:
  - documents
  - chunks
  - vectors
  - graph nodes
  - graph edges
- source/document list
- selected document details

Document list items should show:

- file/source title
- source type
- active/inactive state
- chunk count
- ingest time if available

Actions:

- ingest new source
- rebuild selected document/source
- delete selected document/source
- refresh directory

For first version, the ingest form should support:

- file path / URL location
- source type
- optional title
- optional content text for pasted/plain text sources

### Center Pane: Evidence And Diagnostics

Purpose:

- inspect retrieval quality
- inspect why the answer was produced
- debug routing, token budgets, and provider selection

Must show:

- evidence list with:
  - score
  - citation anchor
  - section path
  - retrieval channels
  - retrieval family
  - source type
  - page range
  - token count / selected token count
- selected evidence detail panel
- routing summary:
  - mode
  - mode executor
  - intent
  - confidence
  - structure / metadata / graph flags
- diagnostics:
  - branch hits
  - branch limits
  - rerank provider
  - provider attempts
  - fused count
  - graph expanded
- context budget summary:
  - max context tokens
  - selected tokens
  - truncated count

The center pane is the main debugging surface and must never be collapsed into the chat pane.

### Right Pane: Model Switcher And Chat

Purpose:

- ask questions
- compare replies across model profiles
- keep the dialogue readable

Must show:

- current profile selector with left/right arrows
- current mode selector
- user turns
- LLM answer only
- generation provider/model status

This pane must not show raw evidence cards. It should remain readable and compact.

## API Surface

The workbench server should expose a minimal private API:

- `GET /api/state`
  - storage root
  - backend summary
  - model catalog
  - active profile
  - index summary
- `GET /api/documents`
  - source/document directory
- `GET /api/documents/{doc_id}`
  - document detail
  - chunk list summary
  - structure summary from chunk section paths
- `POST /api/query`
  - query text
  - mode
  - profile id
  - optional source scope
- `POST /api/ingest`
- `POST /api/rebuild`
- `POST /api/delete`

This API is local/private. It is not designed as a durable external contract.

## Service Layer Responsibilities

`WorkbenchService` is the key boundary.

It should:

- build `RAG` instances from selected profile + storage config
- discover model profiles
- build index summary and document tree from `StorageBundle`
- map `RAGQueryResult` to UI-facing response models
- translate exceptions into structured UI errors

It should not:

- reimplement retrieval logic
- bypass `RAG`
- own business rules that belong in query/ingest/storage layers

## Storage Inspection

The document directory should be built from the existing stores:

- metadata repo:
  - sources
  - documents
  - chunks
  - document status
- vector repo:
  - overall vector counts
- graph repo:
  - overall node / edge counts

Per-document chunk counts should come from metadata.

For first version, global graph/vector counts are sufficient. Per-document vector and graph counts are optional.

## Error Handling

The UI must surface these conditions explicitly:

- runtime contract mismatch between current embedding/tokenizer contract and stored index
- no chat-capable model configured
- no reranker configured
- provider request failure
- ingest parse failure
- rebuild/delete target not found

Error UX rules:

- show the actual message in a visible status area
- do not hide failures behind generic “operation failed”
- keep the last successful evidence/result visible until a new successful query replaces it

## DTO Design

The workbench should not expose raw internal objects directly to the browser.

Create stable UI DTOs for:

- model profile
- index summary
- source/document tree item
- document detail
- workbench query result
- evidence detail item
- operation result banner

These DTOs should normalize field names so the browser does not depend on internal schema churn.

## Testing Strategy

Test the workbench as a real feature, not as static snapshots.

Backend tests:

- model catalog discovery from env
- document directory assembly from a real storage root
- query endpoint returns evidence, diagnostics, and model info from a real `RAGQueryResult`
- ingest/rebuild/delete operations update the directory view
- runtime contract mismatch is surfaced as a structured error

UI tests for first version should stay lightweight:

- serve `index.html`
- smoke-test API roundtrips through the server
- verify the browser-facing JSON shape the UI depends on

Do not spend the first version budget on brittle DOM snapshot testing.

## Implementation Notes

- Keep the browser client framework-free.
- Use fetch-based API calls and DOM rendering.
- Keep CSS in a dedicated file with the approved light theme.
- Preserve the three-column layout on desktop and collapse gracefully on smaller widths.
- Avoid introducing a generalized frontend build pipeline.

## Future Extensions

These are intentionally deferred, but the design should leave room for them:

- graph visualization
- streaming generation
- multi-query comparison
- saved query sessions
- richer model profile management UI
- remote deployment mode

## Old Logic To Avoid

Do not turn this into:

- a thin wrapper around the CLI subprocesses
- a fake chat demo that hides retrieval state
- a second retrieval implementation outside `rag`
- a React/Node side project unrelated to the Python core

The workbench exists to expose the current RAG system clearly and operate it directly.
