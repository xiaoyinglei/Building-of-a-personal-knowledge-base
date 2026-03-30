from __future__ import annotations

from pathlib import Path

from pkp.engine import RAGCore
from pkp.query.query import QueryOptions
from pkp.storage import StorageConfig


def test_ragcore_exposes_insert_query_delete_rebuild() -> None:
    core = RAGCore(storage=StorageConfig.in_memory())

    assert hasattr(core, "insert")
    assert hasattr(core, "query")
    assert hasattr(core, "delete")
    assert hasattr(core, "rebuild")


def test_query_options_defaults_to_mix_mode() -> None:
    options = QueryOptions()

    assert options.mode == "mix"


def test_storage_config_accepts_string_root(tmp_path: Path) -> None:
    root = tmp_path / ".ragcore"

    core = RAGCore(storage=StorageConfig(root=str(root)))

    assert core.stores.root == root
