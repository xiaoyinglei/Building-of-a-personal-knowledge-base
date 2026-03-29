from __future__ import annotations

from pathlib import Path

from pkp.bootstrap import build_rag_core
from pkp.config import AppSettings
from pkp.engine import RAGCore


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
                "max_token_budget": 256,
            }
        }
    )


def test_bootstrap_can_build_ragcore_without_fastapi_or_cli(tmp_path: Path) -> None:
    core = build_rag_core(_settings(tmp_path))
    try:
        assert isinstance(core, RAGCore)

        inserted = core.insert(
            source_type="plain_text",
            location="memory://bootstrap-core",
            owner="user",
            content_text="Alpha Engine processes ingestion requests for the library-first core.",
        )
        result = core.query("What does Alpha Engine do?")

        assert inserted.chunk_count > 0
        assert result.retrieval.evidence.internal
    finally:
        core.stores.close()
