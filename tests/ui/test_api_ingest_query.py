from dataclasses import dataclass

from fastapi.testclient import TestClient

from pkp.types import PreservationSuggestion, QueryResponse, RuntimeMode
from pkp.ui.api.app import create_app


@dataclass
class FakeIngestRuntime:
    def ingest_source(self, *, source_type: str, location: str) -> dict[str, int | str]:
        return {"source_id": "src-1", "chunk_count": 4, "source_type": source_type, "location": location}


@dataclass
class FakeQueryRuntime:
    mode: RuntimeMode

    def run(self, query: str, policy: object) -> QueryResponse:
        return QueryResponse(
            conclusion=f"answer for {query}",
            evidence=[],
            differences_or_conflicts=[],
            uncertainty="low",
            preservation_suggestion=PreservationSuggestion(suggested=True, artifact_type="topic_page"),
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


def test_ingest_query_and_artifact_routes_use_runtime_facades() -> None:
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=FakeIngestRuntime(),
                fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
            )
        )
    )

    ingest_response = client.post("/ingest", json={"source_type": "markdown", "location": "data/sample.md"})
    fast_response = client.post("/query", json={"query": "hello", "mode": "fast"})
    deep_response = client.post("/query", json={"query": "compare docs", "mode": "deep"})
    approve_response = client.post("/artifacts/approve", json={"artifact_id": "artifact-1"})

    assert ingest_response.status_code == 200
    assert ingest_response.json()["source_id"] == "src-1"
    assert fast_response.status_code == 200
    assert fast_response.json()["runtime_mode"] == "fast"
    assert deep_response.status_code == 200
    assert deep_response.json()["runtime_mode"] == "deep"
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "approved"
