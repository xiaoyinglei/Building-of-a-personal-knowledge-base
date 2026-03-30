from pathlib import Path

from fastapi.testclient import TestClient

from pkp.interfaces._bootstrap import build_test_container
from pkp.interfaces._ui.api.app import create_app


def test_pdf_ingest_query_flow(tmp_path: Path) -> None:
    app = create_app(container_factory=lambda: build_test_container(tmp_path))
    client = TestClient(app)

    ingest = client.post(
        "/ingest",
        json={"source_type": "pdf", "location": "data/samples/sample-report.pdf"},
    )
    query = client.post(
        "/query",
        json={"query": "What should Deep Path do?", "mode": "fast"},
    )

    assert ingest.status_code == 200
    assert query.status_code == 200
    assert query.json()["evidence"]
