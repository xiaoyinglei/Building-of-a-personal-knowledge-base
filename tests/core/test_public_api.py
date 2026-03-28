from __future__ import annotations

from pkp.core.options import QueryOptions
from pkp.core.rag_core import RAGCore
from pkp.core.storage_config import StorageConfig


def test_ragcore_exposes_insert_query_delete_rebuild() -> None:
    core = RAGCore(storage=StorageConfig.in_memory())

    assert hasattr(core, "insert")
    assert hasattr(core, "query")
    assert hasattr(core, "delete")
    assert hasattr(core, "rebuild")


def test_query_options_defaults_to_mix_mode() -> None:
    options = QueryOptions()

    assert options.mode == "mix"
