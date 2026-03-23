# Runtime Facades

The UI layer is allowed to touch only the runtime container and its facades:

- `container.ingest_runtime.ingest_source(...)`
- `container.fast_query_runtime.run(...)`
- `container.deep_research_runtime.run(...)`
- `container.artifact_promotion_runtime.approve(...)`

## What The UI Must Not Do

- read SQLite metadata or graph tables directly
- call model providers directly
- reimplement execution-policy assembly
- persist artifacts or telemetry without going through runtime/service boundaries

## What Runtime Hides

Runtime facades hide:

- settings-backed bootstrap and dependency wiring
- retrieval orchestration and evidence self-checks
- provider ordering, cloud-to-local fallback, and synthesis degradation
- artifact approval plus artifact re-indexing
- session-scoped deep research memory

This keeps FastAPI routes and CLI commands thin and replaceable.
