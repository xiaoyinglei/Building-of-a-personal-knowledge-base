from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TARGETS = (
    "rag/retrieval/analysis.py",
    "rag/retrieval/orchestrator.py",
    "rag/agent/service.py",
    "rag/runtime.py",
    "rag/cli.py",
)
_REMOVED_WRAPPER_IMPORTS = (
    "rag.interfaces",
    "rag.runtime",
    "rag.ui",
    "rag.api",
)
_ALLOWED_TOP_LEVEL = {
    "__init__.py",
    "agent",
    "answer_benchmarks.py",
    "assembly",
    "benchmark_diagnostics.py",
    "benchmarks.py",
    "cli.py",
    "ingest",
    "providers",
    "retrieval",
    "runtime.py",
    "schema",
    "storage",
    "utils",
    "workbench",
}


def test_core_modules_do_not_depend_on_removed_wrappers() -> None:
    offenders: dict[str, list[str]] = {}
    for relative_path in _TARGETS:
        content = (_ROOT / relative_path).read_text(encoding="utf-8")
        hits = [legacy for legacy in _REMOVED_WRAPPER_IMPORTS if legacy in content]
        if hits:
            offenders[relative_path] = hits

    assert offenders == {}


def test_top_level_package_matches_target_architecture() -> None:
    package_root = _ROOT / "rag"
    present = {path.name for path in package_root.iterdir() if path.name not in {"__pycache__", ".DS_Store"}}
    assert present == _ALLOWED_TOP_LEVEL


def test_removed_interfaces_package_is_absent() -> None:
    assert not (_ROOT / "rag" / "interfaces").exists()


def _class_method_names(relative_path: str, class_name: str) -> set[str]:
    module = ast.parse((_ROOT / relative_path).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            }
    raise AssertionError(f"class {class_name} not found in {relative_path}")


def test_runtime_module_does_not_import_concrete_storage_backends() -> None:
    content = (_ROOT / "rag/runtime.py").read_text(encoding="utf-8")
    assert "PostgresMetadataRepo" not in content
    assert "MilvusVectorRepo" not in content


def test_metadata_protocol_is_trimmed_to_runtime_core_contract() -> None:
    assert _class_method_names("rag/schema/runtime.py", "MetadataRepo") == {
        "save_source",
        "get_source",
        "get_source_by_location_and_hash",
        "find_source_by_content_hash",
        "get_latest_source_for_location",
        "list_sources",
        "save_document",
        "get_document",
        "list_documents",
        "close",
    }


def test_full_text_protocol_matches_runtime_usage_surface() -> None:
    assert _class_method_names("rag/schema/runtime.py", "FullTextSearchRepo") == {
        "index_chunk",
        "search",
        "delete_by_chunk_ids",
        "close",
    }
