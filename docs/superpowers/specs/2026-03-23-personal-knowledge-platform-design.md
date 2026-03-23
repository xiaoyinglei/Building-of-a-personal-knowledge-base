# Personal Knowledge Platform Design

Status: Draft for user review
Date: 2026-03-23
Author: Codex

## 1. Summary

This document specifies a personal knowledge platform optimized for:

- strong retrieval and answer reliability over raw autonomy
- cloud-first execution for complex tasks with local fallback
- mixed-source knowledge ingest across PDF, Markdown, images, and web content
- deep research workflows for cross-document comparison and iterative search
- gradual accumulation of reusable topic pages, comparison pages, and knowledge artifacts

The selected architecture is a reliability-first variant of a dual-speed knowledge platform:

- `Fast Path` for low-latency, evidence-backed retrieval and direct answers
- `Deep Path` for bounded multi-step research, iterative retrieval, cross-document synthesis, and uncertainty reporting

Graph and GraphRAG capabilities are included as retrieval enhancers and knowledge-asset builders, but they are explicitly forbidden from replacing primary source evidence in final answers.

## 2. Goals

### 2.1 Primary goals

- Answer simple questions quickly with stable citations.
- Answer complex multi-document questions with better decomposition, search, comparison, and synthesis than a basic RAG chatbot.
- Support cloud-first execution while allowing some directories or labels to remain local-only.
- Build a reusable topic knowledge base over time through summaries, comparison pages, topic pages, and open-question pages.
- Keep the system operational under cloud failure by degrading to local retrieval and local model execution where possible.

### 2.2 Success criteria

After 3 months of use, the platform should:

- contain a growing set of topic pages and comparison pages that are useful for future research tasks
- provide answers with traceable evidence and explicit uncertainty when evidence is weak or conflicting
- route simple questions through a fast path without unnecessary planning overhead
- route complex questions through a deep path with bounded iterative retrieval and better synthesis quality

## 3. Non-goals

- A fully general autonomous agent that can take arbitrary external actions without strict tool boundaries
- A graph-only question answering system
- A system that always prefers local execution over answer quality
- A multi-tenant enterprise platform in v1
- A memory system that stores unrestricted chain-of-thought or unverified conclusions

## 4. Design principles

### 4.1 Reliability-first

- Evidence quality is more important than answer fluency.
- Retrieval quality is more important than model cleverness.
- Reranking is a first-class reliability component, not an optional enhancement.
- The system is allowed to say "insufficient evidence" instead of forcing an answer.

### 4.2 Evidence before synthesis

- Final answers must be grounded in source chunks.
- Derived knowledge artifacts must link back to evidence chunks.
- Graph relationships without evidence must not enter the formal graph layer.

### 4.3 Dual-speed execution

- Simple tasks must not pay the latency cost of deep research orchestration.
- Complex tasks must be allowed bounded iterative search and synthesis.
- Fast Path and Deep Path are separate runtime modes with different budgets and safeguards.

### 4.4 Cloud-first, local-safe

- Cloud execution is preferred for high-value reasoning and synthesis.
- Local execution guarantees minimum functionality, privacy enforcement, and graceful degradation.
- Sensitive content is constrained at ingest time, not only at query time.

### 4.5 Knowledge accumulation over chat logs

- Answers are temporary.
- Approved topic pages, comparison pages, summaries, and open-question pages are durable assets.
- The system should help the user decide what to preserve, but should not auto-promote tentative conclusions into long-term knowledge.

## 5. Hard architectural constraint: layer dependency rule

The implementation must follow this exact layer stack:

`Types -> Config -> Repo -> Service -> Runtime -> UI`

Interpretation:

- `Types` is the lowest layer and has no dependencies.
- Each higher layer may depend only on explicitly allowed lower layers.
- No upward dependency is allowed.
- No circular dependency is allowed.
- No side-channel access is allowed.
- This is a hard rule enforced in CI.

### 5.1 Allowed direct dependencies

| Layer | Allowed direct dependencies |
|---|---|
| `Types` | none |
| `Config` | `Types` |
| `Repo` | `Types`, `Config` |
| `Service` | `Types`, `Repo` |
| `Runtime` | `Types`, `Config`, `Service` |
| `UI` | `Types`, `Runtime` |

No other direct dependency is permitted.

### 5.2 Consequences

- `UI` must never read databases, vector stores, model SDKs, or search engines directly.
- `Runtime` is the only layer allowed to compose workflows such as Fast Path, Deep Path, ingest jobs, and research sessions.
- `Service` contains domain logic only and must not import provider SDKs or framework request objects.
- `Repo` is the only layer allowed to talk to external systems such as file systems, databases, model providers, OCR engines, or web search providers.
- `Config` defines policy and routing declarations, but must not execute business logic.
- `Types` contains pure data contracts, enums, protocol shapes, and result envelopes only.

### 5.3 Enforcement

The reference implementation should enforce this with:

- separate packages or modules per layer
- import-boundary lint rules
- dependency graph checks in CI
- a composition root in `Runtime`
- no provider SDK imports outside `Repo`

If Python or other helper runtimes are introduced later for parsing or OCR, they must remain behind `Repo` interfaces and must not break the logical dependency rule above.

## 6. Selected architecture

Three architecture directions were considered:

- A monolithic intelligent RAG system
- A dual-speed knowledge platform
- A graph-first research system

The selected direction is the dual-speed knowledge platform with selective graph capabilities:

- use the dual-speed platform as the primary architecture
- adopt graph and GraphRAG techniques only where they increase recall, relationship discovery, and topic-page value
- keep graph features out of the hot path for basic reliability-critical answering

## 7. System overview

### 7.1 Core modules

The system is organized around six logical modules:

1. `Ingest Pipeline`
   - source intake
   - file parsing
   - OCR and visual extraction
   - TOC recovery
   - chunking
   - deduplication
   - incremental re-ingest

2. `Index Layer`
   - metadata index
   - full-text index
   - vector index
   - graph candidate store

3. `Model Gateway`
   - cloud and local model abstraction
   - embedding, rerank, chat, vision, extraction interfaces
   - execution policy routing

4. `Retrieval Orchestrator`
   - query classification
   - modular RAG graph execution
   - hybrid retrieval
   - RRF fusion
   - reranking
   - self-check gates

5. `Research Runtime`
   - multi-step decomposition
   - bounded iterative search
   - bounded recursive search
   - evidence matrix construction
   - uncertainty reporting

6. `Knowledge Layer`
   - summaries
   - topic pages
   - comparison pages
   - open-question pages
   - entity and concept notes

### 7.2 High-level flow

`source intake -> parse -> normalize -> segment -> chunk -> index -> retrieve -> synthesize -> validate -> answer -> suggest preservation -> approved artifact -> re-index`

### 7.3 Normative ownership and interface boundaries

To make the layer rule actionable during planning, the primary ownership and interface boundaries are fixed here.

#### Repo-owned interfaces

- `SourceRepo`
- `ObjectStoreRepo`
- `MetadataRepo`
- `FullTextSearchRepo`
- `VectorSearchRepo`
- `GraphRepo`
- `ModelProviderRepo`
- `OcrVisionRepo`
- `WebSearchRepo`

These interfaces encapsulate all external IO and provider SDK usage.

#### Service-owned interfaces

- `IngestService`
- `RetrievalService`
- `EvidenceService`
- `ArtifactService`
- `PolicyResolutionService`
- `GraphExpansionService`

These interfaces contain domain rules and validation logic, but no provider SDK imports.

#### Runtime-owned interfaces

- `IngestRuntime`
- `FastQueryRuntime`
- `DeepResearchRuntime`
- `ArtifactPromotionRuntime`
- `SessionRuntime`

These interfaces own orchestration, budget handling, retries, and lifecycle control.

#### UI-facing facades

`UI` is allowed to call only Runtime facades such as:

- `ingestSource`
- `runFastQuery`
- `runDeepResearch`
- `reviewPreservationSuggestion`
- `approveArtifactPromotion`

`UI` must not call `Service` or `Repo` directly.

## 8. Canonical knowledge model

All ingested sources must be normalized into a common knowledge model.

### 8.1 Source

Represents the original intake source.

Examples:

- local file
- watched directory entry
- web URL
- browser clip
- pasted text

Required fields:

- `source_id`
- `source_type`
- `location`
- `owner`
- `content_hash`
- `effective_access_policy`
- `ingest_version`

### 8.2 Document

Represents one independently interpretable document.

Required fields:

- `doc_id`
- `source_id`
- `doc_type`
- `title`
- `authors`
- `created_at`
- `language`
- `effective_access_policy`

### 8.3 Segment

Represents a structural unit such as a section, heading block, page region, or OCR region.

Required fields:

- `segment_id`
- `doc_id`
- `parent_segment_id`
- `toc_path`
- `heading_level`
- `page_range`
- `order_index`

### 8.4 Chunk

Represents the smallest retrievable evidence unit.

Required fields:

- `chunk_id`
- `segment_id`
- `doc_id`
- `text`
- `token_count`
- `citation_anchor`
- `citation_span`
- `effective_access_policy`
- `extraction_quality`
- `embedding_ref`

Chunks are evidence objects, not anonymous text slices.

### 8.5 Knowledge artifact

Represents a derived knowledge object, never a source of record.

Examples:

- `document_summary`
- `section_summary`
- `comparison_page`
- `topic_page`
- `timeline`
- `open_question_page`

Required fields:

- `artifact_id`
- `artifact_type`
- `title`
- `supported_chunk_ids`
- `confidence`
- `status`
- `last_reviewed_at`

`status` must be one of:

- `suggested`
- `approved`
- `stale`
- `conflicted`
- `archived`

### 8.6 Graph node and edge

Graph objects must always preserve evidence links.

Required fields:

- node: `node_id`, `node_type`, `label`
- edge: `edge_id`, `from_node_id`, `to_node_id`, `relation_type`, `confidence`, `evidence_chunk_ids`

Graph relationships without evidence remain candidates and are not promoted into the formal graph layer.

### 8.7 Canonical access policy schema

Privacy and execution policy must use one canonical schema: `AccessPolicy`.

`AccessPolicy` contains:

- `residency`: `cloud_allowed` | `local_preferred` | `local_required`
- `external_retrieval`: `allow` | `deny`
- `allowed_runtimes`: set of `fast` and `deep`
- `allowed_locations`: set of `cloud` and `local`
- `sensitivity_tags`: string set

Rules:

- `AccessPolicy` is attached to `Source`, inherited by `Document`, narrowed by `Segment` if needed, and finalized at `Chunk`.
- Narrowing is allowed; widening is forbidden.
- Derived booleans such as `local_only` or `cloud_allowed` may exist as computed views, but must not be treated as separate sources of truth.
- Request-time execution policy must be compatible with the effective `AccessPolicy` of every selected chunk.

## 9. Ingest and normalization pipeline

### 9.1 Supported source types

The system must support:

- PDF
- Markdown
- images
- web pages
- plain text

### 9.2 Ingest stages

1. intake
2. type detection
3. parse or OCR
4. structure recovery
5. normalization
6. segmentation
7. chunking
8. enrichment
9. indexing

### 9.3 Structure recovery

TOC awareness is mandatory.

The platform must preserve or infer:

- document TOC
- heading hierarchy
- page or region anchors
- section ordering

Why this is mandatory:

- retrieval can target sections rather than only flat text
- summarization can operate hierarchically
- comparison across documents can align sections by topic and depth

### 9.4 Chunking rule

Chunking must be structure-first, window-second.

That means:

- first split by structural boundaries when possible
- then apply token window fallback only to oversized structures
- keep citation anchors stable across re-ingest where possible

### 9.5 Images and scanned documents

Images must generate two outputs:

- `visible_text` via OCR
- `visual_semantics` via image description or layout understanding

Visual semantics are supporting signals unless the user is explicitly asking about the image itself.

### 9.6 Privacy enforcement at ingest

Privacy is decided during ingest, not deferred to answer time.

Each source, document, and chunk must carry or inherit a canonical `AccessPolicy`.

That policy governs:

- embedding eligibility
- reranking eligibility
- answer-generation location
- summarization location
- web-search eligibility
- whether Fast Path, Deep Path, or both are allowed

## 10. Storage and indexing

The design is technology-agnostic at the logical level, but the reference implementation should support:

- object or file storage for source artifacts
- relational storage for metadata and artifacts
- full-text retrieval
- vector retrieval
- graph candidate storage

### 10.1 Recommended v1 logical stores

- source artifact store
- metadata store
- full-text store
- vector store
- artifact store
- graph store or graph tables

### 10.2 Hot-path principle

The hot path for basic answering depends on:

- metadata
- full-text retrieval
- vector retrieval
- reranking

Graph retrieval is optional in the hot path and should remain disabled by default for simple questions.

## 11. Retrieval architecture

### 11.1 Retrieval orchestrator as reliability center

The `Retrieval Orchestrator` is the center of answer reliability.

Responsibilities:

- classify tasks
- select Fast Path or Deep Path
- execute modular retrieval patterns
- fuse candidate sets
- rerank evidence
- trigger self-check gates
- build the evidence packet passed to generation

### 11.2 Routing enums and default heuristics

The platform must normalize every query into these enums before execution:

- `task_type`: `lookup` | `single_doc_qa` | `comparison` | `synthesis` | `timeline` | `research`
- `complexity_level`: `L1_direct` | `L2_scoped` | `L3_comparative` | `L4_research`

Default routing heuristics for planning:

- route to `Fast Path` when `task_type` is `lookup` or `single_doc_qa` and `complexity_level` is `L1_direct` or `L2_scoped`
- route to `Deep Path` when `task_type` is `comparison`, `synthesis`, `timeline`, or `research`
- route to `Deep Path` when the query contains explicit comparative or multi-hop signals such as "compare", "difference", "why", "trend", "conflict", or references to multiple sources

Initial evidence sufficiency heuristics:

- for `lookup` and `single_doc_qa`, the first retrieval round is sufficient only if at least `2` high-ranking chunks from at least `1` relevant section support the answer
- for `comparison`, `synthesis`, `timeline`, and `research`, the first retrieval round is sufficient only if at least `4` high-ranking chunks from at least `2` documents or `2` sections support the task

These are planning-time defaults and may later be tuned, but implementation planning must treat them as the initial contract.

### 11.3 Modular RAG patterns

The retrieval system must support three composition modes:

- sequential mode
- conditional mode
- branch mode

Sequential mode is used for simple questions.

Conditional mode decides:

- whether to upgrade from Fast Path to Deep Path
- whether to run web search
- whether a source must remain local-only
- whether graph expansion is justified

Branch mode allows parallel retrieval from:

- full-text retrieval
- vector retrieval
- TOC or section retrieval
- graph-expanded candidates
- web retrieval when explicitly allowed

### 11.4 RRF fusion

When multiple retrieval branches run, candidate fusion should use RRF before reranking.

RRF is preferred because it:

- improves robustness across heterogeneous retrievers
- avoids over-trusting one ranking source
- works well with full-text plus dense retrieval combinations

### 11.5 Reranking

Reranking is mandatory in both Fast Path and Deep Path.

Reranking must prioritize:

- semantic relevance
- section-level relevance
- citation quality
- source trust and freshness signals
- privacy compatibility

### 11.6 Self-check gates

Self-RAG style ideas are adopted selectively as bounded checks, not as an always-on loop.

Mandatory self-check gates:

- `retrieve_more?`
- `evidence_sufficient?`
- `claim_supported?`

These checks are used to decide whether:

- to stop early
- to retrieve another round
- to downgrade confidence
- to output uncertainty explicitly

The following default gate semantics are normative for planning:

- `retrieve_more? = yes` when evidence coverage is below the thresholds defined in section 11.2
- `evidence_sufficient? = no` when top evidence comes from only one weak chunk, one noisy OCR span, or one unsupported graph relationship
- `claim_supported? = no` when a generated claim cannot be tied to at least one cited source chunk

## 12. Fast Path

The Fast Path is the default runtime for simple and direct questions.

### 12.1 Fast Path flow

1. normalize question
2. apply metadata and privacy filters
3. retrieve using full-text plus vector plus section signals
4. fuse with RRF if multiple branches ran
5. rerank top candidates
6. assemble evidence packet
7. generate answer
8. run claim-citation alignment check
9. return answer with evidence

### 12.2 Fast Path routing signals

Use Fast Path when the question is:

- classified as `lookup` or `single_doc_qa`
- classified as `L1_direct` or `L2_scoped`
- scoped to one document or one topic
- expected to be answerable within one retrieval round and one rerank pass

### 12.3 Fast Path exit conditions

Escalate to Deep Path when:

- evidence coverage is below the thresholds in section 11.2
- the query implies cross-document comparison
- multiple conflicting sources are found
- the answer depends on multi-step reasoning
- the first answer draft fails claim-citation alignment on a material claim

## 13. Deep Path

The Deep Path is a bounded research workflow, not an unrestricted agent loop.

### 13.1 Deep Path flow

1. classify as research task
2. decompose into sub-questions
3. run iterative retrieval
4. optionally run bounded recursive retrieval
5. construct an evidence matrix by source, claim, topic, or time
6. perform synthesis
7. output conclusions, disagreements, and uncertainty
8. propose preservation if the result is valuable

### 13.2 Iterative search

Iterative search is mandatory for complex research tasks.

Pattern:

- retrieve initial evidence
- identify evidence gaps
- rewrite or expand queries
- retrieve again
- stop when evidence is sufficient or the budget is exhausted

Default Deep Path planning limits:

- max retrieval rounds: `4`
- max recursive depth: `2`
- default wall-clock budget: `180` seconds
- default synthesis retry count: `1`

### 13.3 Recursive search

Recursive search is allowed only in bounded form.

Use cases:

- follow references from one concept to related concepts
- expand from one entity to related documents
- refine one unresolved sub-question into narrower sub-questions

Required limits:

- max depth
- max rounds
- max token budget
- max wall-clock budget

### 13.4 Web search gate

Web retrieval is allowed only when:

- internal evidence is insufficient
- the task is not constrained to local-only data
- the execution policy allows external sources

External evidence must remain separately labeled from internal evidence in the final answer.

## 14. Graph and GraphRAG policy

Graph and GraphRAG are explicitly secondary mechanisms.

### 14.1 Allowed uses

- relationship discovery
- candidate expansion after initial evidence exists
- topic-page generation
- concept clustering
- cross-document bridge finding

### 14.2 Forbidden uses

- answering directly from graph structure without source evidence
- replacing primary full-text and vector retrieval in the hot path
- promoting low-confidence edges into final knowledge artifacts

### 14.3 GraphRAG trigger rule

Graph expansion is allowed only after at least one non-graph retrieval round has identified entities, topics, or sections that suggest incomplete coverage.

## 15. Answer generation and validation

### 15.1 Output structure

The default answer structure should contain:

- `Conclusion`
- `Evidence`
- `Differences or conflicts`
- `Uncertainty`
- `Preservation suggestion`

### 15.2 Claim-citation alignment

Every answer must pass a claim-citation support check.

If a generated claim lacks support:

- drop the claim
- restate it as a lower-confidence inference
- or trigger more retrieval if the task budget allows

### 15.3 Conflict handling

Conflicting evidence must be shown explicitly.

The system must prefer:

- exposing disagreement
- naming the scope of each claim
- stating what remains unresolved

over:

- forcing a single unified conclusion without evidence

## 16. Model gateway and execution policy

The model gateway abstracts model providers into capability classes:

- chat and reasoning
- embedding
- reranking
- vision and OCR
- structured extraction and utility prompts

### 16.1 Execution policy fields

Each request must carry or derive:

- `effective_access_policy`
- `task_type`
- `complexity_level`
- `latency_budget`
- `cost_budget`
- `execution_location_preference`
- `fallback_allowed`
- `source_scope`

Where:

- `effective_access_policy` must conform to section 8.7
- `task_type` and `complexity_level` must conform to section 11.2
- `execution_location_preference` is one of `cloud_first`, `local_first`, or `local_only`

### 16.2 Default routing

- complex reasoning and synthesis: cloud-first
- embeddings: local-preferred where feasible
- reranking: local-capable and heavily optimized
- OCR and vision: local-capable for baseline ingest
- local-only sources: full local execution path

### 16.3 Failure and degradation ladder

- `L0`: full cloud-first execution
- `L1`: backup cloud path or reduced context
- `L2`: local fallback answer path
- `L3`: retrieval-only fallback with evidence packet

The system must always prefer a reliable evidence packet over a fabricated fluent answer.

## 17. Knowledge preservation and long-term memory

### 17.1 Three durable layers

The system must separate:

1. source evidence layer
2. derived artifact layer
3. user-approved knowledge layer

### 17.2 Memory types

The system uses five logical memory categories:

- `User Memory`
- `Working Memory`
- `Episodic Memory`
- `Semantic Memory`
- `Operational Memory`

#### User Memory

Stores durable user preferences and constraints.

Examples:

- local-only rules
- preferred answer structure
- topic naming conventions

#### Working Memory

Stores task-local scratch structures.

Examples:

- current sub-questions
- current evidence matrix
- temporary hypotheses

This memory is not durable by default.

#### Episodic Memory

Stores summaries of past high-value research sessions.

#### Semantic Memory

Stores durable knowledge artifacts.

Examples:

- topic pages
- comparison pages
- entity notes

#### Operational Memory

Stores system-level observations that improve retrieval behavior, not user-visible truth claims.

### 17.3 Preservation trigger

The system should propose preservation only when a result is likely reusable.

Typical triggers:

- multi-document comparison
- bounded deep research with meaningful output
- clear recurring topic emergence
- durable conflict map or timeline generation

### 17.4 Preservation mode

Preservation is semi-automatic:

- the system suggests what to preserve
- the user approves promotion into long-term knowledge
- approved artifacts are re-indexed for future retrieval

### 17.5 Anti-pollution rules

- do not store unrestricted chain-of-thought
- do not auto-promote speculative conclusions
- do not let artifacts overwrite source evidence
- mark outdated or conflicted artifacts instead of silently replacing them

## 18. Topic pages and reusable knowledge assets

Each topic page should use a stable structure:

- topic definition
- key conclusions
- key evidence
- boundaries and failure cases
- disagreements
- related documents
- related concepts or entities
- open questions
- last reviewed timestamp
- confidence or coverage indicator

This structure is required so that topic pages remain usable for both humans and retrieval.

## 19. Deployment evolution

### 19.1 Phase 1: local-first bootstrap

Use the local machine as the only runtime host.

Characteristics:

- minimal process count
- local fallback fully available
- cloud used for high-value reasoning
- suitable for M4 Pro class hardware

### 19.2 Phase 2: cloud-primary deployment

Move runtime-heavy components to the cloud while keeping local entry and local-only execution.

Characteristics:

- ingest can become asynchronous
- search and runtime services can scale independently
- local machine becomes a client plus local-safe executor

The design must support this transition without changing domain contracts.

## 20. Observability and evaluation

The platform must measure reliability, not only throughput.

### 20.1 Required telemetry

- retrieval branch usage
- RRF fusion behavior
- rerank effectiveness
- Fast Path to Deep Path escalations
- claim-citation failures
- graph expansion usage
- local fallback frequency
- preservation suggestions and approvals

### 20.2 Evaluation dimensions

- citation precision
- evidence sufficiency rate
- conflict detection quality
- simple-query latency
- deep-query completion quality
- preservation usefulness over time

## 21. Reference implementation boundary map

The following package layout is recommended:

- `packages/types`
- `packages/config`
- `packages/repo`
- `packages/service`
- `packages/runtime`
- `apps/ui`

Recommended responsibility map:

- `packages/types`: contracts, DTOs, envelopes, identifiers
- `packages/config`: policies, route declarations, thresholds
- `packages/repo`: model providers, OCR, storage, search adapters
- `packages/service`: ingest rules, retrieval rules, validation logic, artifact logic
- `packages/runtime`: Fast Path runtime, Deep Path runtime, job orchestration, session orchestration
- `apps/ui`: web, desktop, or CLI entry points

### 21.1 Logical module to layer mapping

To avoid ambiguity during planning and implementation, the logical modules defined earlier map to the layer stack as follows:

| Logical module | Primary layer ownership | Notes |
|---|---|---|
| `Ingest Pipeline` | `Repo + Service + Runtime` | `Repo` handles parsers and storage adapters, `Service` defines ingest rules, `Runtime` runs jobs |
| `Index Layer` | `Repo` | `Service` may define retrieval-facing policies, but index IO remains in `Repo` |
| `Model Gateway` | `Repo + Config + Runtime` | `Repo` owns provider adapters, `Config` owns routing policy, `Runtime` applies policy per request |
| `Retrieval Orchestrator` | `Service` | Domain retrieval logic only; execution sequencing happens in `Runtime` |
| `Research Runtime` | `Runtime` | Owns Deep Path orchestration, bounded iteration, recursion limits, and workflow control |
| `Knowledge Layer` | `Repo + Service` | `Repo` stores artifacts and graph data, `Service` validates and builds durable knowledge objects |

This mapping is normative. Planning and implementation must not relocate these responsibilities across layers in ways that violate the dependency rule.

## 22. Decisions locked by this spec

The following decisions are fixed by this document:

- reliability is more important than autonomy
- Fast Path and Deep Path remain separate runtime modes
- graph features are secondary to source-grounded retrieval
- TOC-aware ingest is mandatory
- chunks are citation-bearing evidence objects
- preservation is semi-automatic, not fully automatic
- the layer dependency rule is hard and enforced

## 23. What planning should decide next

Implementation planning should decide:

- concrete storage technologies
- concrete provider selection for cloud and local execution
- concrete APIs and package interfaces
- rollout order by milestone
- evaluation harness and test plan

Those are planning-time choices and do not alter the design constraints above.
