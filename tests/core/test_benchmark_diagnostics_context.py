from __future__ import annotations

from pathlib import Path

from rag.benchmark_diagnostics import build_diagnostic_context


def test_build_diagnostic_context_normalizes_basic_fields() -> None:
    context = build_diagnostic_context(
        dataset="medical_retrieval",
        run_id="run-1",
        variant="mini",
        profile_id="local_full",
        storage_root=Path("data/benchmarks/medical_retrieval/index/mini"),
        queries_path=Path("data/benchmarks/medical_retrieval/prepared/mini/queries.jsonl"),
        qrels_path=Path("data/benchmarks/medical_retrieval/prepared/mini/qrels.jsonl"),
        retrieval_mode="naive",
        top_k=10,
        chunk_top_k=20,
        retrieval_pool_k=20,
        rerank_enabled=True,
        rerank_pool_k=10,
        enable_parent_backfill=True,
        embedding_provider_kind="ollama",
        embedding_model="qwen3-embedding:4b",
    )

    assert context.retrieval_mode == "naive"
    assert context.top_k == 10
    assert context.chunk_top_k == 20
    assert context.rerank_enabled is True
    assert context.embedding_provider_kind == "ollama"
    assert context.embedding_model == "qwen3-embedding:4b"
