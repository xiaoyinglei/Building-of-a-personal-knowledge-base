from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from pkp.bootstrap import build_test_container
from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.api.app import create_app
from pkp.ui.cli import app as cli_app
from pkp.ui.dependencies import clear_container_factory, set_container_factory

runner = CliRunner()


def test_markdown_query_flow_with_inline_markdown_ingest(
    tmp_path: Path,
) -> None:
    container = build_test_container(tmp_path)
    clear_container_factory()
    set_container_factory(lambda: container)

    ingest = runner.invoke(
        cli_app,
        [
            "ingest",
            "--source-type",
            "markdown",
            "--title",
            "Agent RAG Overview",
            "--content",
            "# Reliability First\n\nEvidence quality is more important than fluent synthesis.",
        ],
    )
    response = container.fast_query_runtime.run(
        "What is more important than fluent synthesis?",
        build_execution_policy(
            task_type=TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L1_DIRECT,
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
    )

    assert ingest.exit_code == 0
    assert response.conclusion
    assert response.evidence
    assert any(
        "evidence quality is more important than fluent synthesis" in item.text.lower() for item in response.evidence
    )
    assert response.uncertainty
    assert response.preservation_suggestion is not None


def test_deep_query_session_can_be_read_back_via_api(tmp_path: Path) -> None:
    container = build_test_container(tmp_path)
    clear_container_factory()
    set_container_factory(lambda: container)

    runner.invoke(
        cli_app,
        [
            "ingest",
            "--source-type",
            "markdown",
            "--title",
            "Agent RAG Overview",
            "--content",
            "# Reliability First\n\nEvidence quality is more important than fluent synthesis.",
        ],
    )
    client = TestClient(create_app(container_factory=lambda: container))
    query = client.post(
        "/query",
        json={
            "query": "compare docs",
            "mode": "deep",
            "session_id": "research-1",
        },
    )
    session = client.get("/sessions/research-1")

    assert query.status_code == 200
    assert session.status_code == 200
    payload = session.json()
    assert payload["sub_questions"]
    assert payload["evidence_matrix"]
