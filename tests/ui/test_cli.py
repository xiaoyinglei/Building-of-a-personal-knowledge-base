from dataclasses import dataclass

from typer.testing import CliRunner

from pkp.types import PreservationSuggestion, QueryResponse, RuntimeMode
from pkp.ui.cli import app, set_container_factory

runner = CliRunner()


@dataclass
class FakeIngestRuntime:
    def ingest_source(self, *, source_type: str, location: str) -> dict[str, int | str]:
        return {"source_id": "src-1", "chunk_count": 2, "source_type": source_type, "location": location}


@dataclass
class FakeQueryRuntime:
    mode: RuntimeMode

    def run(self, query: str, policy: object) -> QueryResponse:
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
    def approve(self, artifact_id: str) -> dict[str, str]:
        return {"artifact_id": artifact_id, "status": "approved"}


@dataclass
class FakeContainer:
    ingest_runtime: FakeIngestRuntime
    fast_query_runtime: FakeQueryRuntime
    deep_research_runtime: FakeQueryRuntime
    artifact_promotion_runtime: FakeArtifactRuntime


def test_cli_supports_health_ingest_query_and_approve() -> None:
    set_container_factory(
        lambda: FakeContainer(
            ingest_runtime=FakeIngestRuntime(),
            fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
            deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
            artifact_promotion_runtime=FakeArtifactRuntime(),
        )
    )

    assert runner.invoke(app, ["health"]).exit_code == 0
    assert "ok" in runner.invoke(app, ["health"]).stdout
    assert "src-1" in runner.invoke(app, ["ingest", "--source-type", "markdown", "--location", "data/a.md"]).stdout
    assert "fast:hello" in runner.invoke(app, ["query", "--query", "hello", "--mode", "fast"]).stdout
    assert "approved" in runner.invoke(app, ["approve-artifact", "--artifact-id", "artifact-1"]).stdout
