from dataclasses import dataclass

from fastapi.testclient import TestClient

from pkp.types import (
    ExecutionLocationPreference,
    ExecutionPolicy,
    PreservationSuggestion,
    QueryResponse,
    RuntimeMode,
)
from pkp.ui.api.app import create_app


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
            "chunk_count": 4,
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
            conclusion=f"answer for {query}",
            evidence=[],
            differences_or_conflicts=[],
            uncertainty="low",
            preservation_suggestion=PreservationSuggestion(suggested=True, artifact_type="topic_page"),
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


def test_ingest_query_and_artifact_routes_use_runtime_facades() -> None:
    ingest_runtime = FakeIngestRuntime(calls=[])
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=ingest_runtime,
                fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
            )
        )
    )

    ingest_response = client.post("/ingest", json={"source_type": "markdown", "location": "data/sample.md"})
    fast_response = client.post("/query", json={"query": "hello", "mode": "fast"})
    deep_response = client.post("/query", json={"query": "compare docs", "mode": "deep"})
    list_response = client.get("/artifacts")
    show_response = client.get("/artifacts/artifact-1")
    approve_response = client.post("/artifacts/approve", json={"artifact_id": "artifact-1"})

    assert ingest_response.status_code == 200
    assert ingest_response.json()["source_id"] == "src-1"
    assert ingest_runtime.calls == [
        {
            "source_type": "markdown",
            "location": "data/sample.md",
            "content": None,
            "title": None,
        }
    ]
    assert fast_response.status_code == 200
    assert fast_response.json()["runtime_mode"] == "fast"
    assert deep_response.status_code == 200
    assert deep_response.json()["runtime_mode"] == "deep"
    assert list_response.status_code == 200
    assert list_response.json()[0]["artifact_id"] == "artifact-1"
    assert show_response.status_code == 200
    assert show_response.json()["artifact_id"] == "artifact-1"
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "approved"


def test_query_route_transmits_execution_controls_into_runtime_policy() -> None:
    fast_runtime = FakeQueryRuntime(mode=RuntimeMode.FAST)
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=FakeIngestRuntime(calls=[]),
                fast_query_runtime=fast_runtime,
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
            )
        )
    )

    response = client.post(
        "/query",
        json={
            "query": "hello",
            "mode": "fast",
            "source_scope": ["doc-1", "doc-2"],
            "latency_budget": 17,
            "token_budget": 321,
            "execution_location_preference": "local_first",
            "fallback_allowed": False,
            "cost_budget": 2.75,
        },
    )

    assert response.status_code == 200
    assert fast_runtime.last_policy is not None
    assert fast_runtime.last_policy.source_scope == ["doc-1", "doc-2"]
    assert fast_runtime.last_policy.latency_budget == 17
    assert fast_runtime.last_policy.token_budget == 321
    assert fast_runtime.last_policy.execution_location_preference is ExecutionLocationPreference.LOCAL_FIRST
    assert fast_runtime.last_policy.fallback_allowed is False
    assert fast_runtime.last_policy.cost_budget == 2.75


def test_ingest_route_accepts_inline_content_and_optional_title() -> None:
    ingest_runtime = FakeIngestRuntime(calls=[])
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=ingest_runtime,
                fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
            )
        )
    )

    response = client.post(
        "/ingest",
        json={
            "source_type": "browser_clip",
            "content": "<html><body><article><h1>Clip</h1></article></body></html>",
            "title": "Saved Clip",
        },
    )

    assert response.status_code == 200
    assert ingest_runtime.calls == [
        {
            "source_type": "browser_clip",
            "location": None,
            "content": "<html><body><article><h1>Clip</h1></article></body></html>",
            "title": "Saved Clip",
        }
    ]
