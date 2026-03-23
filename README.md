# Personal Knowledge Platform

A reliability-first personal knowledge platform with strict layered architecture:

`Types -> Config -> Repo -> Service -> Runtime -> UI`

## Features

- ingest PDF, Markdown, plain text, images, and web pages
- hybrid retrieval using full-text, vector, and section-aware signals
- separate Fast Path and Deep Path runtimes
- explicit evidence packets, conflict exposure, and uncertainty reporting
- semi-automatic preservation of durable knowledge artifacts
- local fallback and policy-aware execution routing

## Bootstrap

```bash
uv sync --all-extras
cp .env.example .env
```

The runtime is local-first by default:

- metadata and artifacts live under `data/runtime`
- source files are copied into `data/runtime/objects`
- SQLite metadata, FTS, graph, and telemetry files are created on demand
- cloud synthesis uses `PKP_OPENAI__API_KEY` when available
- local fallback synthesis uses Ollama at `PKP_OLLAMA__BASE_URL`

## Provider Setup

Minimal environment variables:

```bash
export PKP_OPENAI__API_KEY=your-key
export PKP_OLLAMA__BASE_URL=http://localhost:11434
```

Common runtime controls in [`.env.example`](/Users/leixiaoying/LLM/RAG学习/.env.example):

- `PKP_RUNTIME__EXECUTION_LOCATION_PREFERENCE=cloud_first|local_first|local_only`
- `PKP_RUNTIME__FALLBACK_ALLOWED=true|false`
- `PKP_RUNTIME__DEFAULT_WALL_CLOCK_BUDGET_SECONDS=180`
- `PKP_RUNTIME__MAX_TOKEN_BUDGET=`

## Local Development

```bash
uv sync --all-extras
uv run pytest -v
uv run ruff check .
uv run mypy src
uv run lint-imports
uv run python -m scripts.check_repo_only_imports
```

## Layout

- `src/pkp/types`: pure contracts and enums
- `src/pkp/config`: settings and routing defaults
- `src/pkp/repo`: storage, parsing, search, graph, and model adapters
- `src/pkp/service`: domain rules for ingest, retrieval, evidence, and artifacts
- `src/pkp/runtime`: Fast Path, Deep Path, ingest, and promotion orchestration
- `src/pkp/ui`: FastAPI and CLI entry points

## Running

CLI:

```bash
uv run python -m pkp.ui.cli health
uv run python -m pkp.ui.cli ingest --source-type markdown --location README.md
uv run python -m pkp.ui.cli query --mode fast --query "What does this project do?"
uv run python -m pkp.ui.cli query --mode deep --query "Compare Fast Path and Deep Path"
uv run python -m pkp.ui.cli approve-artifact --artifact-id artifact-123
```

API:

```bash
uv run uvicorn pkp.ui.api.app:create_app --factory --reload
```

Example requests:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type":"markdown","location":"README.md"}'

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare Fast Path and Deep Path","mode":"deep"}'

curl -X POST http://127.0.0.1:8000/artifacts/approve \
  -H "Content-Type: application/json" \
  -d '{"artifact_id":"artifact-123"}'
```

## Reliability Signals

The local telemetry stream records:

- retrieval branch usage, RRF fusion, and rerank effectiveness
- Fast Path to Deep Path escalations and claim-citation failures
- graph expansion, local fallback usage, and preservation suggestions
- artifact approvals

Evaluation helpers aggregate:

- citation precision
- evidence sufficiency rate
- conflict detection quality
- simple-query latency
- deep-query completion quality
- preservation usefulness
