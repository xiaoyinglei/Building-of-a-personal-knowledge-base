from dataclasses import dataclass

from fastapi.testclient import TestClient

from pkp.runtime.session_runtime import SessionRuntime
from pkp.types import (
    AccessPolicy,
    AnswerCitation,
    AnswerEvidenceLink,
    AnswerSection,
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
from pkp.ui.api.app import create_app


@dataclass
class FakeIngestRuntime:
    calls: list[dict[str, object | None]]

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
            "chunk_count": 4,
            "source_type": source_type,
            "location": location or f"inline://{source_type}/generated",
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
            conclusion=f"answer for {query}",
            evidence=[],
            differences_or_conflicts=[],
            uncertainty="low",
            preservation_suggestion=PreservationSuggestion(suggested=True, artifact_type="topic_page"),
            runtime_mode=self.mode,
            answer_text=f"answer for {query}",
            answer_sections=[
                AnswerSection(
                    section_id="sec-1",
                    title="直接回答",
                    text=f"answer for {query}",
                    citation_ids=["cit-1"],
                    evidence_chunk_ids=["chunk-a"],
                )
            ],
            citations=[
                AnswerCitation(
                    citation_id="cit-1",
                    file_name="report.docx",
                    section_path=["专项工作"],
                    page_start=2,
                    page_end=2,
                    chunk_id="chunk-a",
                    chunk_type="child",
                )
            ],
            evidence_links=[
                AnswerEvidenceLink(
                    link_id="link-1",
                    answer_section_id="sec-1",
                    answer_excerpt=f"answer for {query}",
                    evidence_chunk_id="chunk-a",
                    citation_id="cit-1",
                    support_score=1.0,
                )
            ],
            groundedness_flag=True,
            insufficient_evidence_flag=False,
            diagnostics=QueryDiagnostics(
                retrieval=RetrievalDiagnostics(
                    branch_hits={"full_text": 2, "vector": 1},
                    reranked_chunk_ids=["chunk-a", "chunk-b"],
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
class FakeContainer:
    ingest_runtime: FakeIngestRuntime
    fast_query_runtime: FakeQueryRuntime
    deep_research_runtime: FakeQueryRuntime
    artifact_promotion_runtime: FakeArtifactRuntime
    session_runtime: SessionRuntime


def test_ingest_query_and_artifact_routes_use_runtime_facades() -> None:
    ingest_runtime = FakeIngestRuntime(calls=[])
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=ingest_runtime,
                fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
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
            "access_policy": None,
        }
    ]
    assert fast_response.status_code == 200
    assert fast_response.json()["runtime_mode"] == "fast"
    assert fast_response.json()["answer_text"] == "answer for hello"
    assert fast_response.json()["answer_sections"][0]["section_id"] == "sec-1"
    assert fast_response.json()["citations"][0]["file_name"] == "report.docx"
    assert fast_response.json()["evidence_links"][0]["evidence_chunk_id"] == "chunk-a"
    assert fast_response.json()["groundedness_flag"] is True
    assert fast_response.json()["diagnostics"]["retrieval"]["embedding_provider"] == "ollama"
    assert fast_response.json()["diagnostics"]["retrieval"]["rerank_provider"] == "heuristic"
    assert deep_response.status_code == 200
    assert deep_response.json()["runtime_mode"] == "deep"
    assert deep_response.json()["diagnostics"]["model"]["synthesis_provider"] == "local"
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
                session_runtime=SessionRuntime(),
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


def test_query_route_transmits_access_policy_into_runtime_policy() -> None:
    fast_runtime = FakeQueryRuntime(mode=RuntimeMode.FAST)
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=FakeIngestRuntime(calls=[]),
                fast_query_runtime=fast_runtime,
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
            )
        )
    )

    response = client.post(
        "/query",
        json={
            "query": "sensitive lookup",
            "mode": "fast",
            "access_policy": {
                "residency": "local_required",
                "external_retrieval": "deny",
                "allowed_runtimes": ["fast"],
                "allowed_locations": ["local"],
                "sensitivity_tags": ["private", "regulated"],
            },
        },
    )

    assert response.status_code == 200
    assert fast_runtime.last_policy is not None
    assert fast_runtime.last_policy.effective_access_policy == AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes=frozenset({RuntimeMode.FAST}),
        allowed_locations=frozenset({ExecutionLocation.LOCAL}),
        sensitivity_tags=frozenset({"private", "regulated"}),
    )


def test_ingest_route_accepts_inline_content_and_optional_title() -> None:
    ingest_runtime = FakeIngestRuntime(calls=[])
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=ingest_runtime,
                fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
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
            "access_policy": None,
        }
    ]


def test_ingest_route_transmits_access_policy() -> None:
    ingest_runtime = FakeIngestRuntime(calls=[])
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=ingest_runtime,
                fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
            )
        )
    )

    response = client.post(
        "/ingest",
        json={
            "source_type": "pasted_text",
            "content": "local-only notes",
            "access_policy": {
                "residency": "local_required",
                "external_retrieval": "deny",
                "allowed_runtimes": ["fast"],
                "allowed_locations": ["local"],
                "sensitivity_tags": ["private"],
            },
        },
    )

    assert response.status_code == 200
    assert ingest_runtime.calls == [
        {
            "source_type": "pasted_text",
            "location": None,
            "content": "local-only notes",
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


def test_query_route_transmits_session_id_into_deep_runtime() -> None:
    deep_runtime = FakeQueryRuntime(mode=RuntimeMode.DEEP)
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=FakeIngestRuntime(calls=[]),
                fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
                deep_research_runtime=deep_runtime,
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
            )
        )
    )

    response = client.post(
        "/query",
        json={"query": "compare docs", "mode": "deep", "session_id": "research-1"},
    )

    assert response.status_code == 200
    assert deep_runtime.last_session_id == "research-1"


def test_session_route_returns_session_snapshot() -> None:
    session_runtime = SessionRuntime()
    session_runtime.store_sub_questions("research-1", ["What changed?", "Why?"])
    session_runtime.store_evidence_matrix("research-1", [{"claim": "A", "sources": ["doc-1"]}])
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=FakeIngestRuntime(calls=[]),
                fast_query_runtime=FakeQueryRuntime(mode=RuntimeMode.FAST),
                deep_research_runtime=FakeQueryRuntime(mode=RuntimeMode.DEEP),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=session_runtime,
            )
        )
    )

    response = client.get("/sessions/research-1")

    assert response.status_code == 200
    assert response.json() == {
        "sub_questions": ["What changed?", "Why?"],
        "evidence_matrix": [{"claim": "A", "sources": ["doc-1"]}],
        "memory_hints": [],
        "episode_id": None,
    }
