from pathlib import Path

from fastapi.testclient import TestClient

from pkp.bootstrap import build_test_container
from pkp.ui.api.app import create_app


def test_web_ingest_flow(tmp_path: Path) -> None:
    app = create_app(container_factory=lambda: build_test_container(tmp_path))
    client = TestClient(app)

    ingest = client.post(
        "/ingest",
        json={"source_type": "web", "location": "https://example.com/article"},
    )
    query = client.post(
        "/query",
        json={"query": "What happens when cloud synthesis fails?", "mode": "fast"},
    )

    assert ingest.status_code == 200
    assert query.status_code == 200
    assert query.json()["evidence"]
