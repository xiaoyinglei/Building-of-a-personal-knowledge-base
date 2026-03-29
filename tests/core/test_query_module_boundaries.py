from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_TARGETS = (
    "src/pkp/query/retrieve.py",
    "src/pkp/query/graph.py",
    "src/pkp/engine.py",
    "src/pkp/bootstrap.py",
    "src/pkp/eval/offline_eval_service.py",
)
_LEGACY_IMPORTS = (
    "pkp.service.evidence_service",
    "pkp.service.query_understanding_service",
    "pkp.service.routing_service",
)


def test_query_modules_do_not_depend_on_legacy_query_services() -> None:
    offenders: dict[str, list[str]] = {}
    for relative_path in _TARGETS:
        content = (_ROOT / relative_path).read_text(encoding="utf-8")
        hits = [legacy for legacy in _LEGACY_IMPORTS if legacy in content]
        if hits:
            offenders[relative_path] = hits

    assert offenders == {}
