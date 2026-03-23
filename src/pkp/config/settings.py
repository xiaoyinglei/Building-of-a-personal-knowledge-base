from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from pkp.types.access import ExecutionLocationPreference


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_dir: Path = Path("data/runtime")
    db_url: str = "sqlite:///data/runtime/pkp.sqlite3"
    object_store_dir: Path = Path("data/runtime/objects")
    execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.CLOUD_FIRST
    fallback_allowed: bool = True
    fast_min_evidence_chunks: int = 2
    deep_min_evidence_chunks: int = 4
    max_retrieval_rounds: int = 4
    max_recursive_depth: int = 2
    max_token_budget: int | None = None
    default_wall_clock_budget_seconds: int = 180
    default_synthesis_retry_count: int = 1


class OpenAISettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    rerank_model: str = "gpt-4.1-mini"


class OllamaSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PKP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
