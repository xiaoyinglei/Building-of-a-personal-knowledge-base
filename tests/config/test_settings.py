from pydantic import SecretStr

from pkp.config.settings import AppSettings
from pkp.types.access import ExecutionLocationPreference


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
    assert settings.openai.api_key.get_secret_value() == "test-key"
