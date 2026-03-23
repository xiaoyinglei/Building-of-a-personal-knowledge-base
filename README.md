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

```bash
uv run python -m pkp.ui.cli health
uv run uvicorn pkp.ui.api.app:create_app --factory --reload
```

