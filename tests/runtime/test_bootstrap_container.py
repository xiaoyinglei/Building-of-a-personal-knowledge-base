from fastapi.testclient import TestClient

from pkp.bootstrap import build_runtime_container
from pkp.config import AppSettings, build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.api.app import create_app
from pkp.ui.dependencies import clear_container_factory


def _settings(tmp_path) -> AppSettings:
    runtime_root = tmp_path / "runtime"
    return AppSettings.model_validate(
        {
            "runtime": {
                "data_dir": str(runtime_root),
                "db_url": f"sqlite:///{runtime_root / 'pkp.sqlite3'}",
                "object_store_dir": str(runtime_root / "objects"),
                "execution_location_preference": "local_first",
                "fallback_allowed": True,
                "max_retrieval_rounds": 4,
                "max_token_budget": 256,
            }
        }
    )


def test_build_runtime_container_supports_real_ingest_and_query(tmp_path) -> None:
    container = build_runtime_container(_settings(tmp_path))

    ingest_result = container.ingest_runtime.ingest_source(
        source_type="markdown",
        location="data/samples/agent-rag-overview.md",
    )
    response = container.fast_query_runtime.run(
        "What does Fast Path optimize for?",
        build_execution_policy(
            task_type=TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L1_DIRECT,
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
    )

    assert ingest_result["chunk_count"] > 0
    assert response.evidence
    assert container.metadata_repo is not None
    assert container.telemetry_service is not None
    assert container.telemetry_service.count_by_name("retrieval.branch_used") >= 1


def test_create_app_bootstraps_default_container_from_settings(tmp_path, monkeypatch) -> None:
    clear_container_factory()
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("PKP_RUNTIME__DATA_DIR", str(runtime_root))
    monkeypatch.setenv("PKP_RUNTIME__DB_URL", f"sqlite:///{runtime_root / 'pkp.sqlite3'}")
    monkeypatch.setenv("PKP_RUNTIME__OBJECT_STORE_DIR", str(runtime_root / "objects"))
    monkeypatch.setenv("PKP_RUNTIME__EXECUTION_LOCATION_PREFERENCE", "local_first")
    monkeypatch.setenv("PKP_RUNTIME__MAX_TOKEN_BUDGET", "256")
    client = TestClient(create_app())

    ingest_response = client.post(
        "/ingest",
        json={"source_type": "markdown", "location": "data/samples/agent-rag-overview.md"},
    )
    query_response = client.post(
        "/query",
        json={"query": "What does Deep Path do?", "mode": "fast"},
    )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunk_count"] > 0
    assert query_response.status_code == 200
    assert query_response.json()["evidence"]
