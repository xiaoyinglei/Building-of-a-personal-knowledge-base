from dataclasses import dataclass, field

from typer.testing import CliRunner

from pkp.runtime.session_runtime import SessionRuntime
from pkp.types import (
    AccessPolicy,
    ExecutionLocation,
    ExecutionLocationPreference,
    ExecutionPolicy,
    ExternalRetrievalPolicy,
    ModelDiagnostics,
    PreservationSuggestion,
    ProviderAttempt,
    QueryDiagnostics,
    QueryResponse,
    Residency,
    RetrievalDiagnostics,
    RuntimeMode,
)
from pkp.ui.cli import app, set_container_factory

runner = CliRunner()


@dataclass
class FakeIngestRuntime:
    calls: list[dict[str, object | None]]
    file_calls: list[dict[str, object | None]] = field(default_factory=list)

    def ingest_source(
        self,
        *,
        source_type: str,
        location: str | None = None,
        content: str | None = None,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> dict[str, int | str]:
        self.calls.append(
            {
                "source_type": source_type,
                "location": location,
                "content": content,
                "title": title,
                "access_policy": access_policy,
            }
        )
        return {
            "source_id": "src-1",
            "chunk_count": 2,
            "source_type": source_type,
            "location": location or f"inline://{source_type}/generated",
        }

    def repair_indexes(self) -> dict[str, int]:
        return {
            "document_count": 1,
            "chunk_count": 2,
            "repaired_vector_count": 2,
        }

    def process_file(
        self,
        *,
        location: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> dict[str, object]:
        self.file_calls.append(
            {
                "location": location,
                "title": title,
                "access_policy": access_policy,
            }
        )
        return {
            "source_id": "src-file",
            "doc_id": "doc-file",
            "processing": {
                "analysis": {"source_type": "docx"},
                "routing": {"selected_strategy": "hierarchical"},
                "stats": {"total_chunks": 6},
            },
        }


@dataclass
class FakeQueryRuntime:
    mode: RuntimeMode
    last_policy: ExecutionPolicy | None = None
    last_session_id: str | None = None

    def run(
        self,
        query: str,
        policy: ExecutionPolicy,
        *,
        session_id: str = "default",
    ) -> QueryResponse:
        self.last_policy = policy
        self.last_session_id = session_id
        return QueryResponse(
            conclusion=f"{self.mode.value}:{query}",
            evidence=[],
            differences_or_conflicts=[],
            uncertainty="low",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=self.mode,
            diagnostics=QueryDiagnostics(
                retrieval=RetrievalDiagnostics(
                    branch_hits={"full_text": 1, "vector": 1},
                    reranked_chunk_ids=["chunk-a"],
                    embedding_provider="ollama",
                    rerank_provider="heuristic",
                ),
                model=ModelDiagnostics(
                    synthesis_provider="local" if self.mode is RuntimeMode.DEEP else None,
                    attempts=[
                        ProviderAttempt(
                            stage="embedding",
                            capability="embed",
                            provider="ollama",
                            location="local",
                            model="embed-test",
                            status="success",
                        )
                    ],
                ),
            ),
        )


@dataclass
class FakeArtifactRuntime:
    def list_artifacts(self) -> list[dict[str, str]]:
        return [{"artifact_id": "artifact-1", "status": "approved"}]

    def get_artifact(self, artifact_id: str) -> dict[str, str]:
        return {"artifact_id": artifact_id, "status": "approved"}

    def approve(self, artifact_id: str) -> dict[str, str]:
        return {"artifact_id": artifact_id, "status": "approved"}


@dataclass
class FakeDiagnosticsRuntime:
    def report(self) -> dict[str, object]:
        return {
            "status": "degraded",
            "providers": [
                {
                    "provider": "openai",
                    "location": "cloud",
                    "capabilities": {
                        "chat": {
                            "configured": True,
                            "available": False,
                            "model": "gpt-test",
                            "error": "404",
                        }
                    },
                }
            ],
            "indices": {"documents": 1, "chunks": 2, "vectors": 2, "missing_vectors": 0},
        }


@dataclass
class FakeContainer:
    ingest_runtime: FakeIngestRuntime
    fast_query_runtime: FakeQueryRuntime
    deep_research_runtime: FakeQueryRuntime
    artifact_promotion_runtime: FakeArtifactRuntime
    session_runtime: SessionRuntime
    diagnostics_runtime: FakeDiagnosticsRuntime = field(default_factory=FakeDiagnosticsRuntime)


def test_cli_supports_health_ingest_query_and_artifact_commands() -> None:
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=FakeIngestRuntime(calls=[]),
            fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
            session_runtime=SessionRuntime(),
            diagnostics_runtime=FakeDiagnosticsRuntime(),
        )
    )

    assert runner.invoke(app, ["health"]).exit_code == 0
    assert "degraded" in runner.invoke(app, ["health"]).stdout
    assert "openai" in runner.invoke(app, ["health", "--json"]).stdout
    assert "src-1" in runner.invoke(app, ["ingest", "--source-type", "markdown", "--location", "data/a.md"]).stdout
    assert "repaired_vector_count" in runner.invoke(app, ["repair-indexes"]).stdout
    assert "fast:hello" in runner.invoke(app, ["query", "--query", "hello", "--mode", "fast"]).stdout
    assert '"embedding_provider": "ollama"' in runner.invoke(
        app,
        ["query", "--query", "hello", "--mode", "fast", "--json"],
    ).stdout
    assert "artifact-1" in runner.invoke(app, ["list-artifacts"]).stdout
    assert "artifact-1" in runner.invoke(app, ["show-artifact", "--artifact-id", "artifact-1"]).stdout
    assert "approved" in runner.invoke(app, ["approve-artifact", "--artifact-id", "artifact-1"]).stdout


def test_cli_query_transmits_execution_controls_into_runtime_policy() -> None:
    fast_runtime = FakeQueryRuntime(mode=RuntimeMode.FAST)
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=FakeIngestRuntime(calls=[]),
            fast_query_runtime=fast_runtime,
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
            session_runtime=SessionRuntime(),
            diagnostics_runtime=FakeDiagnosticsRuntime(),
        )
    )

    result = runner.invoke(
        app,
        [
            "query",
            "--query",
            "hello",
            "--mode",
            "fast",
            "--source-scope",
            "doc-1",
            "--source-scope",
            "doc-2",
            "--latency-budget",
            "17",
            "--token-budget",
            "321",
            "--execution-location-preference",
            "local_first",
            "--no-fallback-allowed",
            "--cost-budget",
            "2.75",
        ],
    )

    assert result.exit_code == 0
    assert fast_runtime.last_policy is not None
    assert fast_runtime.last_policy.source_scope == ["doc-1", "doc-2"]
    assert fast_runtime.last_policy.latency_budget == 17
    assert fast_runtime.last_policy.token_budget == 321
    assert fast_runtime.last_policy.execution_location_preference is ExecutionLocationPreference.LOCAL_FIRST
    assert fast_runtime.last_policy.fallback_allowed is False
    assert fast_runtime.last_policy.cost_budget == 2.75


def test_cli_ingest_supports_inline_pasted_text_and_browser_clip_content() -> None:
    ingest_runtime = FakeIngestRuntime(calls=[])
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=ingest_runtime,
            fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
            session_runtime=SessionRuntime(),
            diagnostics_runtime=FakeDiagnosticsRuntime(),
        )
    )

    pasted_result = runner.invoke(
        app,
        [
            "ingest",
            "--source-type",
            "pasted_text",
            "--content",
            "Inline notes about retrieval reliability.",
            "--title",
            "Retrieval Notes",
        ],
    )
    browser_clip_result = runner.invoke(
        app,
        [
            "ingest",
            "--source-type",
            "browser_clip",
            "--content",
            "<html><body><article><h1>Browser Clip</h1><p>Fast Path prioritizes speed.</p></article></body></html>",
            "--title",
            "Browser Clip",
        ],
    )

    assert pasted_result.exit_code == 0
    assert browser_clip_result.exit_code == 0
    assert ingest_runtime.calls == [
        {
            "source_type": "pasted_text",
            "location": None,
            "content": "Inline notes about retrieval reliability.",
            "title": "Retrieval Notes",
            "access_policy": None,
        },
        {
            "source_type": "browser_clip",
            "location": None,
            "content": (
                "<html><body><article><h1>Browser Clip</h1><p>Fast Path prioritizes speed.</p></article></body></html>"
            ),
            "title": "Browser Clip",
            "access_policy": None,
        },
    ]
    assert '"source_type": "pasted_text"' in pasted_result.stdout
    assert '"source_type": "browser_clip"' in browser_clip_result.stdout


def test_cli_query_transmits_access_policy_into_runtime_policy() -> None:
    fast_runtime = FakeQueryRuntime(mode=RuntimeMode.FAST)
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=FakeIngestRuntime(calls=[]),
            fast_query_runtime=fast_runtime,
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
            session_runtime=SessionRuntime(),
            diagnostics_runtime=FakeDiagnosticsRuntime(),
        )
    )

    result = runner.invoke(
        app,
        [
            "query",
            "--query",
            "hello",
            "--mode",
            "fast",
            "--residency",
            "local_required",
            "--external-retrieval",
            "deny",
            "--allowed-runtime",
            "fast",
            "--allowed-location",
            "local",
            "--sensitivity-tag",
            "private",
        ],
    )

    assert result.exit_code == 0
    assert fast_runtime.last_policy is not None
    assert fast_runtime.last_policy.effective_access_policy == AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes=frozenset({RuntimeMode.FAST}),
        allowed_locations=frozenset({ExecutionLocation.LOCAL}),
        sensitivity_tags=frozenset({"private"}),
    )


def test_cli_ingest_transmits_access_policy() -> None:
    ingest_runtime = FakeIngestRuntime(calls=[])
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=ingest_runtime,
            fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
            session_runtime=SessionRuntime(),
            diagnostics_runtime=FakeDiagnosticsRuntime(),
        )
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "--source-type",
            "pasted_text",
            "--content",
            "Sensitive note",
            "--residency",
            "local_required",
            "--external-retrieval",
            "deny",
            "--allowed-runtime",
            "fast",
            "--allowed-location",
            "local",
            "--sensitivity-tag",
            "private",
        ],
    )

    assert result.exit_code == 0
    assert ingest_runtime.calls == [
        {
            "source_type": "pasted_text",
            "location": None,
            "content": "Sensitive note",
            "title": None,
            "access_policy": AccessPolicy(
                residency=Residency.LOCAL_REQUIRED,
                external_retrieval=ExternalRetrievalPolicy.DENY,
                allowed_runtimes=frozenset({RuntimeMode.FAST}),
                allowed_locations=frozenset({ExecutionLocation.LOCAL}),
                sensitivity_tags=frozenset({"private"}),
            ),
        }
    ]


def test_cli_exposes_process_file_entry() -> None:
    ingest_runtime = FakeIngestRuntime(calls=[])
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=ingest_runtime,
            fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
            session_runtime=SessionRuntime(),
            diagnostics_runtime=FakeDiagnosticsRuntime(),
        )
    )

    result = runner.invoke(app, ["process-file", "--location", "data/samples/example.docx"])

    assert result.exit_code == 0
    assert '"source_id": "src-file"' in result.stdout
    assert ingest_runtime.file_calls == [
        {
            "location": "data/samples/example.docx",
            "title": None,
            "access_policy": None,
        }
    ]


def test_cli_query_transmits_session_id_into_deep_runtime() -> None:
    deep_runtime = FakeQueryRuntime(mode=RuntimeMode.DEEP)
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=FakeIngestRuntime(calls=[]),
            fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
            deep_research_runtime=deep_runtime,
            artifact_promotion_runtime=FakeArtifactRuntime(),
            session_runtime=SessionRuntime(),
        )
    )

    result = runner.invoke(
        app,
        [
            "query",
            "--query",
            "compare docs",
            "--mode",
            "deep",
            "--session-id",
            "research-1",
        ],
    )

    assert result.exit_code == 0
    assert deep_runtime.last_session_id == "research-1"


def test_cli_show_session_returns_session_snapshot() -> None:
    session_runtime = SessionRuntime()
    session_runtime.store_sub_questions("research-1", ["What changed?", "Why?"])
    session_runtime.store_evidence_matrix("research-1", [{"claim": "A", "sources": ["doc-1"]}])
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=FakeIngestRuntime(calls=[]),
            fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
            session_runtime=session_runtime,
        )
    )

    result = runner.invoke(app, ["show-session", "--session-id", "research-1"])

    assert result.exit_code == 0
    assert '"sub_questions": ["What changed?", "Why?"]' in result.stdout
    assert '"evidence_matrix": [{"claim": "A", "sources": ["doc-1"]}]' in result.stdout
