from dataclasses import dataclass

from typer.testing import CliRunner

from pkp.types import (
    ExecutionLocationPreference,
    ExecutionPolicy,
    PreservationSuggestion,
    QueryResponse,
    RuntimeMode,
)
from pkp.ui.cli import app, set_container_factory

runner = CliRunner()


@dataclass
class FakeIngestRuntime:
    calls: list[dict[str, str | None]]

    def ingest_source(
        self,
        *,
        source_type: str,
        location: str | None = None,
        content: str | None = None,
        title: str | None = None,
    ) -> dict[str, int | str]:
        self.calls.append(
            {
                "source_type": source_type,
                "location": location,
                "content": content,
                "title": title,
            }
        )
        return {
            "source_id": "src-1",
            "chunk_count": 2,
            "source_type": source_type,
            "location": location or f"inline://{source_type}/generated",
        }


@dataclass
class FakeQueryRuntime:
    mode: RuntimeMode
    last_policy: ExecutionPolicy | None = None

    def run(self, query: str, policy: ExecutionPolicy) -> QueryResponse:
        self.last_policy = policy
        return QueryResponse(
            conclusion=f"{self.mode.value}:{query}",
            evidence=[],
            differences_or_conflicts=[],
            uncertainty="low",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=self.mode,
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
class FakeContainer:
    ingest_runtime: FakeIngestRuntime
    fast_query_runtime: FakeQueryRuntime
    deep_research_runtime: FakeQueryRuntime
    artifact_promotion_runtime: FakeArtifactRuntime


def test_cli_supports_health_ingest_query_and_artifact_commands() -> None:
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=FakeIngestRuntime(calls=[]),
            fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
        )
    )

    assert runner.invoke(app, ["health"]).exit_code == 0
    assert "ok" in runner.invoke(app, ["health"]).stdout
    assert "src-1" in runner.invoke(app, ["ingest", "--source-type", "markdown", "--location", "data/a.md"]).stdout
    assert "fast:hello" in runner.invoke(app, ["query", "--query", "hello", "--mode", "fast"]).stdout
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
        },
        {
            "source_type": "browser_clip",
            "location": None,
            "content": (
                "<html><body><article><h1>Browser Clip</h1><p>Fast Path prioritizes speed.</p></article></body></html>"
            ),
            "title": "Browser Clip",
        },
    ]
    assert '"source_type": "pasted_text"' in pasted_result.stdout
    assert '"source_type": "browser_clip"' in browser_clip_result.stdout
