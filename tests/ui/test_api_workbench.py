from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from pkp.runtime.session_runtime import SessionRuntime
from pkp.types import (
    AccessPolicy,
    Document,
    DocumentType,
    ExecutionLocation,
    ExternalRetrievalPolicy,
    PreservationSuggestion,
    QueryResponse,
    Residency,
    RuntimeMode,
    Source,
    SourceType,
)
from pkp.ui.api.app import create_app


@dataclass
class FakeIngestRuntime:
    calls: list[dict[str, object | None]] = field(default_factory=list)

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
            "source_id": "src-upload",
            "chunk_count": 3,
            "source_type": source_type,
            "location": location or "inline://generated",
        }


@dataclass
class FakeQueryRuntime:
    def run(self, query: str, policy: object, *, session_id: str = "default") -> QueryResponse:
        del query, policy, session_id
        return QueryResponse(
            conclusion="answer",
            evidence=[],
            uncertainty="low",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=RuntimeMode.FAST,
        )


@dataclass
class FakeArtifactRuntime:
    def list_artifacts(self) -> list[dict[str, str]]:
        return []

    def get_artifact(self, artifact_id: str) -> dict[str, str]:
        return {"artifact_id": artifact_id}

    def approve(self, artifact_id: str) -> dict[str, str]:
        return {"artifact_id": artifact_id, "status": "approved"}


@dataclass
class FailingUploadRuntime(FakeIngestRuntime):
    def ingest_source(
        self,
        *,
        source_type: str,
        location: str | None = None,
        content: str | None = None,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> dict[str, int | str]:
        del source_type, location, content, title, access_policy
        raise ValueError("uploaded file is not UTF-8 text")


@dataclass
class FakeMetadataRepo:
    sources: list[Source]
    documents: list[Document]

    def list_sources(self, location: str | None = None) -> list[Source]:
        if location is None:
            return list(self.sources)
        return [source for source in self.sources if source.location == location]

    def list_documents(self, source_id: str | None = None, *, active_only: bool = False) -> list[Document]:
        del active_only
        if source_id is None:
            return list(self.documents)
        return [document for document in self.documents if document.source_id == source_id]


@dataclass
class FakeContainer:
    ingest_runtime: FakeIngestRuntime
    fast_query_runtime: FakeQueryRuntime
    deep_research_runtime: FakeQueryRuntime
    artifact_promotion_runtime: FakeArtifactRuntime
    session_runtime: SessionRuntime
    metadata_repo: object | None = None
    diagnostics_runtime: object | None = None


def test_workbench_query_page_renders_html_shell() -> None:
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=FakeIngestRuntime(),
                fast_query_runtime=FakeQueryRuntime(),
                deep_research_runtime=FakeQueryRuntime(),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
            )
        )
    )

    response = client.get("/workbench/query")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PKP Workbench" in response.text
    assert "Query Workspace" in response.text
    assert "Ingest" in response.text
    assert "Artifacts" in response.text
    assert "参数说明" in response.text
    assert "Cloud First" in response.text
    assert "优先云端执行检索和综合回答" in response.text
    assert "Compare Fast / Deep" in response.text
    assert "Citations / 引用" in response.text
    assert "Evaluation Pad / 测评辅助" in response.text


def test_workbench_upload_route_saves_file_and_invokes_ingest_runtime(tmp_path: Path) -> None:
    ingest_runtime = FakeIngestRuntime()
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=ingest_runtime,
                fast_query_runtime=FakeQueryRuntime(),
                deep_research_runtime=FakeQueryRuntime(),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
            ),
            workbench_upload_dir=tmp_path / "uploads",
        )
    )

    response = client.post(
        "/ingest/upload",
        files={"file": ("notes.md", b"# Retrieval Notes\nhello", "text/markdown")},
        data={"title": "Retrieval Notes"},
    )

    assert response.status_code == 200
    assert ingest_runtime.calls[0]["source_type"] == "markdown"
    assert ingest_runtime.calls[0]["title"] == "Retrieval Notes"
    saved_location = str(ingest_runtime.calls[0]["location"])
    assert saved_location.endswith("notes.md")
    assert Path(saved_location).exists()


def test_workbench_upload_route_returns_bad_request_for_invalid_file_type(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=FailingUploadRuntime(),
                fast_query_runtime=FakeQueryRuntime(),
                deep_research_runtime=FakeQueryRuntime(),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
            ),
            workbench_upload_dir=tmp_path / "uploads",
        )
    )

    response = client.post(
        "/ingest/upload",
        files={"file": ("notes.bin", b"\xff\xfe\x00\x87", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "uploaded file is not UTF-8 text"


def test_sources_route_returns_recent_documents_with_source_metadata() -> None:
    access_policy = AccessPolicy(
        residency=Residency.CLOUD_ALLOWED,
        external_retrieval=ExternalRetrievalPolicy.ALLOW,
        allowed_runtimes=frozenset({RuntimeMode.FAST, RuntimeMode.DEEP}),
        allowed_locations=frozenset({ExecutionLocation.LOCAL, ExecutionLocation.CLOUD}),
        sensitivity_tags=frozenset(),
    )
    source_a = Source(
        source_id="source-a",
        source_type=SourceType.MARKDOWN,
        location="/tmp/alpha.md",
        owner="user",
        content_hash="hash-a",
        effective_access_policy=access_policy,
        ingest_version=1,
    )
    source_b = Source(
        source_id="source-b",
        source_type=SourceType.PDF,
        location="/tmp/beta.pdf",
        owner="user",
        content_hash="hash-b",
        effective_access_policy=access_policy,
        ingest_version=2,
    )
    document_a = Document(
        doc_id="doc-a",
        source_id="source-a",
        doc_type=DocumentType.NOTE,
        title="Alpha Notes",
        authors=[],
        created_at=datetime(2026, 3, 25, tzinfo=UTC),
        language="zh",
        effective_access_policy=access_policy,
        metadata={"location": "/tmp/alpha.md"},
    )
    document_b = Document(
        doc_id="doc-b",
        source_id="source-b",
        doc_type=DocumentType.REPORT,
        title="Beta Report",
        authors=[],
        created_at=datetime(2026, 3, 26, tzinfo=UTC),
        language="en",
        effective_access_policy=access_policy,
        metadata={"location": "/tmp/beta.pdf"},
    )
    client = TestClient(
        create_app(
            container_factory=lambda: FakeContainer(
                ingest_runtime=FakeIngestRuntime(),
                fast_query_runtime=FakeQueryRuntime(),
                deep_research_runtime=FakeQueryRuntime(),
                artifact_promotion_runtime=FakeArtifactRuntime(),
                session_runtime=SessionRuntime(),
                metadata_repo=FakeMetadataRepo(
                    sources=[source_a, source_b],
                    documents=[document_a, document_b],
                ),
            )
        )
    )

    response = client.get("/sources")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["doc_id"] == "doc-b"
    assert payload[0]["source_type"] == "pdf"
    assert payload[0]["title"] == "Beta Report"
    assert payload[1]["doc_id"] == "doc-a"
    assert payload[1]["source_type"] == "markdown"
