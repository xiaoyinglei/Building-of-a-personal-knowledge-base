from pathlib import Path

from fastapi.testclient import TestClient

from pkp.interfaces._bootstrap import build_test_container
from pkp.interfaces._ui.api.app import create_app


def test_artifact_promotion_flow_reindexes_approved_artifact(tmp_path: Path) -> None:
    app = create_app(container_factory=lambda: build_test_container(tmp_path))
    client = TestClient(app)

    client.post("/ingest", json={"source_type": "plain_text", "location": "data/samples/conflict-a.txt"})
    client.post("/ingest", json={"source_type": "plain_text", "location": "data/samples/conflict-b.txt"})
    query = client.post(
        "/query",
        json={"query": "Compare the default retrieval path in the two documents", "mode": "deep"},
    )
    suggestion = query.json()["preservation_suggestion"]
    list_before = client.get("/artifacts")
    approve = client.post("/artifacts/approve", json={"artifact_id": suggestion["artifact_id"]})
    show_after = client.get(f"/artifacts/{suggestion['artifact_id']}")
    follow_up = client.post(
        "/query",
        json={"query": suggestion["title"], "mode": "fast"},
    )

    assert suggestion["suggested"] is True
    assert suggestion["artifact_id"]
    assert list_before.status_code == 200
    assert any(item["artifact_id"] == suggestion["artifact_id"] for item in list_before.json())
    assert approve.status_code == 200
    assert approve.json()["status"] == "approved"
    assert show_after.status_code == 200
    assert show_after.json()["status"] == "approved"
    assert follow_up.status_code == 200
    assert follow_up.json()["evidence"]
