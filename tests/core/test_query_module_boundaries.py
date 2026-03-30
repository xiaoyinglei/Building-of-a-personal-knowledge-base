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
_INGEST_TARGETS = ("src/pkp/ingest/chunk.py",)
_INGEST_LEGACY_IMPORTS = (
    "pkp.service.chunk_postprocessing_service",
    "pkp.service.chunk_routing_service",
    "pkp.service.chunking_service",
    "pkp.service.document_feature_service",
    "pkp.service.document_processing_service",
    "pkp.service.toc_service",
)


def test_query_modules_do_not_depend_on_legacy_query_services() -> None:
    offenders: dict[str, list[str]] = {}
    for relative_path in _TARGETS:
        content = (_ROOT / relative_path).read_text(encoding="utf-8")
        hits = [legacy for legacy in _LEGACY_IMPORTS if legacy in content]
        if hits:
            offenders[relative_path] = hits

    assert offenders == {}


def test_ingest_chunk_module_does_not_depend_on_legacy_chunk_services() -> None:
    offenders: dict[str, list[str]] = {}
    for relative_path in _INGEST_TARGETS:
        content = (_ROOT / relative_path).read_text(encoding="utf-8")
        hits = [legacy for legacy in _INGEST_LEGACY_IMPORTS if legacy in content]
        if hits:
            offenders[relative_path] = hits

    assert offenders == {}
