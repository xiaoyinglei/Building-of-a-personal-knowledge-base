# Memory Scope

## Working Memory

Session-local sub-questions and evidence matrices live in `SessionRuntime` and are not durable by default.

## User Memory

User-specific defaults live in configuration and environment settings, such as execution-location preference, fallback policy, and runtime budgets. v1 does not yet include a richer profile store.

## Semantic Memory

Approved knowledge artifacts become durable retrieval assets.

## Operational Memory

Operational memory is the telemetry stream. It tracks:

- retrieval branch usage
- RRF fusion and rerank effectiveness
- graph expansion usage
- Fast Path to Deep Path escalations
- claim-citation failures
- cloud-to-local fallback frequency
- preservation suggestions and approvals

## Episodic Memory

Session summaries remain deferred beyond the v1 baseline unless promoted into approved artifacts or other durable research outputs.
