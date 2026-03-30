from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_TARGETS = (
    "src/pkp/query/retrieve.py",
    "src/pkp/query/graph.py",
    "src/pkp/engine.py",
    "src/pkp/interfaces/_bootstrap.py",
    "src/pkp/interfaces/_eval/offline_eval_service.py",
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
_LLM_TARGETS = (
    "src/pkp/interfaces/_bootstrap.py",
    "src/pkp/interfaces/_eval/offline_eval_service.py",
    "src/pkp/query/context.py",
    "src/pkp/interfaces/_runtime/adapters.py",
    "src/pkp/llm/_generation/answer_generator.py",
    "src/pkp/llm/generation.py",
    "src/pkp/llm/rerank.py",
)
_LLM_LEGACY_IMPORTS = (
    "pkp.service.answer_generation_service",
    "pkp.service.rerank_service",
    "pkp.service.retrieval_service",
)
_INGEST_API_TARGETS = (
    "src/pkp/interfaces/_bootstrap.py",
    "src/pkp/interfaces/_eval/offline_eval_service.py",
    "src/pkp/interfaces/_runtime/adapters.py",
    "src/pkp/interfaces/_runtime/ingest_runtime.py",
)
_INGEST_API_LEGACY_IMPORTS = ("pkp.service.ingest_service",)
_ALLOWED_TOP_LEVEL = {
    "__init__.py",
    "document",
    "engine.py",
    "ingest",
    "interfaces",
    "llm",
    "query",
    "schema",
    "storage",
    "utils",
}


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


def test_llm_and_retrieval_modules_do_not_depend_on_service_wrappers() -> None:
    offenders: dict[str, list[str]] = {}
    for relative_path in _LLM_TARGETS:
        content = (_ROOT / relative_path).read_text(encoding="utf-8")
        hits = [legacy for legacy in _LLM_LEGACY_IMPORTS if legacy in content]
        if hits:
            offenders[relative_path] = hits

    assert offenders == {}


def test_runtime_and_bootstrap_modules_do_not_depend_on_ingest_service_wrapper() -> None:
    offenders: dict[str, list[str]] = {}
    for relative_path in _INGEST_API_TARGETS:
        content = (_ROOT / relative_path).read_text(encoding="utf-8")
        hits = [legacy for legacy in _INGEST_API_LEGACY_IMPORTS if legacy in content]
        if hits:
            offenders[relative_path] = hits

    assert offenders == {}


def test_top_level_package_matches_target_architecture() -> None:
    package_root = _ROOT / "src/pkp"
    present = {path.name for path in package_root.iterdir() if path.name != "__pycache__"}
    assert present == _ALLOWED_TOP_LEVEL
