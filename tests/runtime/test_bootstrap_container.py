from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import pkp.interfaces._bootstrap as bootstrap_module
from pkp.interfaces._bootstrap import build_runtime_container
from pkp.interfaces._config import AppSettings, build_execution_policy, default_access_policy
from pkp.llm._providers.fallback_embedding_repo import FallbackEmbeddingRepo
from pkp.schema._types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.interfaces._ui.api.app import create_app
from pkp.interfaces._ui.dependencies import clear_container_factory


def _settings(tmp_path: Path) -> AppSettings:
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


def test_build_runtime_container_supports_real_ingest_and_query(tmp_path: Path) -> None:
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

    assert int(ingest_result["chunk_count"]) > 0
    assert response.evidence
    assert container.metadata_repo is not None
    assert container.telemetry_service is not None
    assert container.telemetry_service.count_by_name("retrieval.branch_used") >= 1


def test_create_app_bootstraps_default_container_from_settings(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
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


def test_build_runtime_container_wires_model_provider_settings(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = AppSettings.model_validate(
        {
            "runtime": {
                "data_dir": str(tmp_path / "runtime"),
                "object_store_dir": str(tmp_path / "runtime" / "objects"),
            },
            "openai": {
                "api_key": "provider-key",
                "base_url": "https://openai.test/v1",
                "model": "gpt-test",
                "embedding_model": "embed-test",
            },
            "ollama": {
                "base_url": "http://ollama.test:11434",
                "chat_model": "qwen3.5:9b",
                "embedding_model": "qwen3-embedding:8b",
            },
            "local_bge": {
                "enabled": True,
                "embedding_model": "BAAI/bge-m3",
                "embedding_model_path": "/models/bge-m3",
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "rerank_model_path": "/models/bge-reranker-v2-m3",
            },
        }
    )
    captured: dict[str, object] = {}

    def fake_openai_provider(**kwargs: object) -> object:
        captured["openai_kwargs"] = kwargs
        return SimpleNamespace(name="openai")

    def fake_ollama_provider(**kwargs: object) -> object:
        captured["ollama_kwargs"] = kwargs
        return SimpleNamespace(name="ollama")

    def fake_local_bge_provider(**kwargs: object) -> object:
        captured["local_bge_kwargs"] = kwargs
        return SimpleNamespace(name="local-bge", provider_name="local-bge")

    def fake_rerank_service(**kwargs: object) -> object:
        captured["rerank_kwargs"] = kwargs
        service = SimpleNamespace(name="rerank")
        captured["rerank_instance"] = service
        return service

    def fake_build_container(**kwargs: object) -> object:
        captured["cloud_providers"] = kwargs["cloud_providers"]
        captured["local_providers"] = kwargs["local_providers"]
        captured["rerank_service"] = kwargs["rerank_service"]
        return SimpleNamespace()

    monkeypatch.setattr(bootstrap_module, "OpenAIProviderRepo", fake_openai_provider)
    monkeypatch.setattr(bootstrap_module, "OllamaProviderRepo", fake_ollama_provider)
    monkeypatch.setattr(bootstrap_module, "LocalBgeProviderRepo", fake_local_bge_provider)
    monkeypatch.setattr(bootstrap_module, "HeuristicRerankService", fake_rerank_service)
    monkeypatch.setattr(bootstrap_module, "_build_runtime_container", fake_build_container)

    result = build_runtime_container(settings)

    assert isinstance(result, SimpleNamespace)
    assert captured["openai_kwargs"] == {
        "api_key": "provider-key",
        "base_url": "https://openai.test/v1",
        "model": "gpt-test",
        "embedding_model": "embed-test",
    }
    assert captured["ollama_kwargs"] == {
        "base_url": "http://ollama.test:11434",
        "chat_model": "qwen3.5:9b",
        "embedding_model": None,
    }
    assert captured["local_bge_kwargs"] == {
        "embedding_model": "BAAI/bge-m3",
        "embedding_model_path": "/models/bge-m3",
        "rerank_model": "BAAI/bge-reranker-v2-m3",
        "rerank_model_path": "/models/bge-reranker-v2-m3",
    }
    assert len(captured["cloud_providers"]) == 1
    assert len(captured["local_providers"]) == 2
    assert captured["rerank_service"] is captured["rerank_instance"]
    rerank_kwargs = captured["rerank_kwargs"]
    rerank_config = rerank_kwargs["config"]
    assert rerank_kwargs["provider"] is captured["local_providers"][0]
    assert rerank_config.cross_encoder.model_name == "BAAI/bge-reranker-v2-m3"
    assert rerank_config.cross_encoder.model_path == "/models/bge-reranker-v2-m3"


def test_build_runtime_container_uses_cloud_embedding_space_for_cloud_first_queries(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeProvider:
        def __init__(self, name: str, *, chat_model: str, embedding_model: str) -> None:
            self.provider_name = name
            self.chat_model_name = chat_model
            self.embedding_model_name = embedding_model
            self.is_chat_configured = True
            self.is_embed_configured = True
            self._fallback = FallbackEmbeddingRepo()

        def embed(self, texts: list[str]) -> list[list[float]]:
            return self._fallback.embed(texts)

        def chat(self, prompt: str) -> str:
            return prompt.splitlines()[0]

    monkeypatch.setattr(
        bootstrap_module,
        "OpenAIProviderRepo",
        lambda **_: FakeProvider("openai", chat_model="cloud-chat", embedding_model="cloud-embed"),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "OllamaProviderRepo",
        lambda **_: FakeProvider("ollama", chat_model="local-chat", embedding_model="local-embed"),
    )
    settings = AppSettings.model_validate(
        {
            "runtime": {
                "data_dir": str(tmp_path / "runtime"),
                "object_store_dir": str(tmp_path / "runtime" / "objects"),
                "execution_location_preference": "cloud_first",
            },
            "openai": {
                "api_key": "provider-key",
                "base_url": "https://openai.test/v1",
                "model": "gpt-test",
                "embedding_model": "embed-test",
            },
            "ollama": {
                "base_url": "http://ollama.test:11434",
                "chat_model": "llama-test",
                "embedding_model": "nomic-test",
            },
        }
    )

    container = build_runtime_container(settings)
    container.ingest_runtime.ingest_source(
        source_type="markdown",
        location="data/samples/agent-rag-overview.md",
    )
    response = container.deep_research_runtime.run(
        "What does Fast Path optimize for?",
        build_execution_policy(
            task_type=TaskType.RESEARCH,
            complexity_level=ComplexityLevel.L4_RESEARCH,
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.CLOUD_FIRST,
        ),
        session_id="cloud-first",
    )

    assert response.diagnostics.retrieval.embedding_provider == "openai"
    assert response.diagnostics.model.synthesis_provider == "openai"
