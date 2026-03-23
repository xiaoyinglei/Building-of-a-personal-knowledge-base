from pathlib import Path

from fastapi.testclient import TestClient

from pkp.bootstrap import build_test_container
from pkp.ui.api.app import create_app


def test_markdown_query_flow(tmp_path: Path) -> None:
    app = create_app(container_factory=lambda: build_test_container(tmp_path))
    client = TestClient(app)

    ingest = client.post(
        "/ingest",
        json={
            "source_type": "markdown",
            "location": "data/samples/agent-rag-overview.md",
        },
    )
    query = client.post(
        "/query",
        json={
            "query": "What is more important than fluent synthesis?",
            "mode": "fast",
        },
    )

    assert ingest.status_code == 200
    assert query.status_code == 200
    payload = query.json()
    assert "Conclusion" not in payload  # API returns structured fields, not a rendered string blob
    assert payload["conclusion"]
    assert payload["evidence"]
    assert "uncertainty" in payload
    assert "preservation_suggestion" in payload
