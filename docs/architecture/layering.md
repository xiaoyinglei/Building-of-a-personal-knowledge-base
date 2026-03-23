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

## Enforcement

- `import-linter` checks the package order.
- `scripts/check_repo_only_imports.py` ensures provider/parser SDK imports remain under `src/pkp/repo`.

