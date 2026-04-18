from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.benchmark_diagnostics import (
    BenchmarkDiagnosticsPostProcessor,
    build_diagnostic_context,
    build_runtime_for_diagnostics,
    load_run_summary,
)
from rag.benchmarks import FIQA_DATASET, MEDICAL_RETRIEVAL_DATASET, default_benchmark_paths, ensure_benchmark_layout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose an existing benchmark retrieval run.")
    parser.add_argument(
        "--dataset",
        default=MEDICAL_RETRIEVAL_DATASET,
        choices=[FIQA_DATASET, MEDICAL_RETRIEVAL_DATASET],
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", default="full", choices=["full", "mini"])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--queries-path", default=None)
    parser.add_argument("--qrels-path", default=None)
    parser.add_argument("--diagnostics-root", default="data/eval/diagnostics")
    parser.add_argument("--mode", default=None, choices=["bypass", "naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--chunk-top-k", type=int, default=None)
    parser.add_argument("--retrieval-pool-k", type=int, default=None)
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank_enabled", action="store_true")
    rerank_group.add_argument("--no-rerank", dest="rerank_enabled", action="store_false")
    parser.set_defaults(rerank_enabled=None)
    parser.add_argument("--rerank-pool-k", type=int, default=None)
    parser.add_argument("--disable-parent-backfill", action="store_true")
    parser.add_argument("--enable-query-understanding-llm", action="store_true")
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--embedding-provider", default=None, choices=["local-bge", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--chat-provider", default=None, choices=["ollama", "openai-compatible", "local-hf"])
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--chat-model-path", default=None)
    parser.add_argument("--chat-backend", default=None, choices=["auto", "mlx", "transformers"])
    args = parser.parse_args(argv)

    paths = ensure_benchmark_layout(default_benchmark_paths(args.dataset))
    run_dir = paths.eval_variant_dir("retrieval", args.variant) / "runs" / args.run_id
    run_summary = load_run_summary(run_dir)

    retrieval_mode = args.mode or str(run_summary.get("retrieval_mode") or "mix")
    top_k = max(int(args.top_k or run_summary.get("top_k") or 10), 1)
    chunk_top_k = max(int(args.chunk_top_k or run_summary.get("chunk_top_k") or top_k), top_k)
    rerank_enabled = (
        args.rerank_enabled if args.rerank_enabled is not None else bool(run_summary.get("rerank_enabled", True))
    )

    context = build_diagnostic_context(
        dataset=args.dataset,
        run_id=args.run_id,
        variant=args.variant,
        profile_id=args.profile,
        storage_root=Path(args.storage_root) if args.storage_root else paths.index_variant_dir(args.variant),
        queries_path=(
            Path(args.queries_path)
            if args.queries_path
            else (paths.prepared_variant_dir(args.variant) / "queries.jsonl")
        ),
        qrels_path=(
            Path(args.qrels_path)
            if args.qrels_path
            else (paths.prepared_variant_dir(args.variant) / "qrels.jsonl")
        ),
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        retrieval_pool_k=args.retrieval_pool_k,
        rerank_enabled=rerank_enabled,
        rerank_pool_k=args.rerank_pool_k,
        enable_parent_backfill=not args.disable_parent_backfill,
        query_limit=args.query_limit,
        enable_query_understanding_llm=args.enable_query_understanding_llm,
        embedding_provider_kind=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_model_path=args.embedding_model_path,
        chat_provider_kind=args.chat_provider,
        chat_model=args.chat_model,
        chat_model_path=args.chat_model_path,
        chat_backend=args.chat_backend,
    )
    runtime = build_runtime_for_diagnostics(context)
    try:
        outputs = BenchmarkDiagnosticsPostProcessor(
            runtime=runtime,
            context=context,
        ).diagnose(
            diagnostics_root=Path(args.diagnostics_root),
        )
    finally:
        runtime.close()
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
