from pathlib import Path

from fastapi.testclient import TestClient

from pkp.bootstrap import build_test_container
from pkp.ui.api.app import create_app


def test_plain_text_conflict_flow_surfaces_conflicts(tmp_path: Path) -> None:
    app = create_app(container_factory=lambda: build_test_container(tmp_path))
    client = TestClient(app)

    client.post("/ingest", json={"source_type": "plain_text", "location": "data/samples/conflict-a.txt"})
    client.post("/ingest", json={"source_type": "plain_text", "location": "data/samples/conflict-b.txt"})
    query = client.post(
        "/query",
        json={"query": "Compare the default retrieval path in the two documents", "mode": "deep"},
    )

    assert query.status_code == 200
    payload = query.json()
    assert payload["runtime_mode"] == "deep"
    assert payload["differences_or_conflicts"]
