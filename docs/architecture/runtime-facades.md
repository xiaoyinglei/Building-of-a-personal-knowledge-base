# Runtime Facades

The UI layer is allowed to call only runtime facades:

- `ingest_source`
- `run_fast_query`
- `run_deep_research`
- `approve_artifact`

These facades hide storage, retrieval, provider selection, and fallback orchestration from the UI.

