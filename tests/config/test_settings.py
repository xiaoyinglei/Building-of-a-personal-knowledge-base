from pathlib import Path

from pydantic import SecretStr
from pytest import MonkeyPatch

from pkp.interfaces._config.settings import AppSettings
from pkp.schema._types.access import ExecutionLocationPreference


def test_settings_parse_nested_env_values() -> None:
    settings = AppSettings.model_validate(
        {
            "runtime": {
                "data_dir": "var/data",
                "db_url": "sqlite:///var/data/pkp.sqlite3",
                "object_store_dir": "var/data/objects",
                "execution_location_preference": "local_first",
                "fallback_allowed": False,
                "max_retrieval_rounds": 3,
                "max_token_budget": 512,
            },
            "openai": {
                "api_key": SecretStr("test-key"),
                "base_url": "https://example.com/v1",
            },
        }
    )

    assert settings.runtime.data_dir.as_posix() == "var/data"
    assert settings.runtime.execution_location_preference is ExecutionLocationPreference.LOCAL_FIRST
    assert settings.runtime.max_retrieval_rounds == 3
    assert settings.runtime.max_token_budget == 512
    assert settings.openai.api_key.get_secret_value() == "test-key"


def test_settings_load_dotenv_file_from_current_workdir(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PKP_OPENAI__API_KEY", raising=False)
    monkeypatch.delenv("PKP_RUNTIME__MAX_TOKEN_BUDGET", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "PKP_OPENAI__API_KEY=dotenv-key",
                "PKP_RUNTIME__MAX_TOKEN_BUDGET=321",
            ]
        ),
        encoding="utf-8",
    )

    settings = AppSettings()

    assert settings.openai.api_key.get_secret_value() == "dotenv-key"
    assert settings.runtime.max_token_budget == 321


def test_settings_treat_empty_max_token_budget_as_none(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PKP_RUNTIME__MAX_TOKEN_BUDGET", raising=False)
    (tmp_path / ".env").write_text(
        "PKP_RUNTIME__MAX_TOKEN_BUDGET=\n",
        encoding="utf-8",
    )

    settings = AppSettings()

    assert settings.runtime.max_token_budget is None


def test_settings_parse_local_bge_model_settings() -> None:
    settings = AppSettings.model_validate(
        {
            "local_bge": {
                "enabled": True,
                "embedding_model": "BAAI/bge-m3",
                "embedding_model_path": "~/.cache/huggingface/hub/models--BAAI--bge-m3",
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "rerank_model_path": "~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3",
            }
        }
    )

    assert settings.local_bge.enabled is True
    assert settings.local_bge.embedding_model == "BAAI/bge-m3"
    assert settings.local_bge.embedding_model_path is not None
    assert settings.local_bge.embedding_model_path.as_posix().endswith("models--BAAI--bge-m3")
    assert settings.local_bge.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert settings.local_bge.rerank_model_path is not None
    assert settings.local_bge.rerank_model_path.as_posix().endswith(
        "models--BAAI--bge-reranker-v2-m3"
    )
