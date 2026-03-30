# Layering Contract

The implementation follows this hard dependency rule:

`Types -> Config -> Repo -> Service -> Runtime -> UI`

## Rules

- `Types` defines contracts only.
- `Config` depends only on `Types`.
- `Repo` owns external IO, storage, parsing, search, and provider adapters.
- `Service` owns domain rules, validation, retrieval logic, and artifact logic.
- `Runtime` orchestrates Fast Path, Deep Path, ingest, and promotion flows.
- `UI` only calls runtime facades and never reaches into services or repos directly.

## Practical Boundary Map

- `Types`: Pydantic models, enums, and response envelopes
- `Config`: settings parsing, routing thresholds, access-policy defaults
- `Repo`: SQLite metadata/FTS/graph stores, file object store, parsers, provider SDK adapters
- `Service`: ingest rules, evidence evaluation, retrieval fusion, artifact lifecycle, telemetry helpers
- `Runtime`: `IngestRuntime`, `FastQueryRuntime`, `DeepResearchRuntime`, `ArtifactPromotionRuntime`
- `UI`: CLI commands and FastAPI routes only

## Non-Negotiable Constraints

- provider SDK imports stay under `src/rag/repo/**`
- UI must not import storage repos or service-layer internals directly
- runtime owns orchestration, retries, fallback ordering, and session state
- services stay framework-agnostic and do not depend on FastAPI or Typer objects

## Enforcement

- `import-linter` checks the package order.
- `scripts/check_repo_only_imports.py` ensures provider/parser SDK imports remain under `src/rag/repo`.
- CI runs `pytest`, `ruff check`, `mypy src`, `lint-imports`, and the repo-only import guard.
