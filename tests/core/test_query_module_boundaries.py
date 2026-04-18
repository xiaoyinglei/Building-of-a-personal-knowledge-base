from __future__ import annotations

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
