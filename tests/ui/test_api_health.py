from fastapi.testclient import TestClient

from pkp.ui.api.app import create_app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(create_app(container_factory=lambda: type("C", (), {"diagnostics_runtime": None})()))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "providers" in payload
    assert "indices" in payload


def test_health_endpoint_returns_provider_and_index_diagnostics_from_runtime() -> None:
    class FakeDiagnosticsRuntime:
        def report(self) -> dict[str, object]:
            return {
                "status": "degraded",
                "providers": [
                    {
                        "provider": "openai",
                        "location": "cloud",
                        "capabilities": {
                            "chat": {
                                "configured": True,
                                "available": False,
                                "model": "gpt-test",
                                "error": "404",
                            }
                        },
                    }
                ],
                "indices": {
                    "documents": 3,
                    "chunks": 12,
                    "vectors": 8,
                    "missing_vectors": 4,
                },
            }

    client = TestClient(
        create_app(
            container_factory=lambda: type("C", (), {"diagnostics_runtime": FakeDiagnosticsRuntime()})()
        )
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "providers": [
            {
                "provider": "openai",
                "location": "cloud",
                "capabilities": {
                    "chat": {
                        "configured": True,
                        "available": False,
                        "model": "gpt-test",
                        "error": "404",
                    }
                },
            }
        ],
        "indices": {
            "documents": 3,
            "chunks": 12,
            "vectors": 8,
            "missing_vectors": 4,
        },
    }
