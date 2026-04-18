from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag.benchmarks import (
    FIQA_DATASET,
    MEDICAL_RETRIEVAL_DATASET,
    RetrievalBenchmarkEvaluator,
    benchmark_access_policy,
    benchmark_dataset_spec,
    build_runtime_for_benchmark,
    default_benchmark_paths,
    ensure_benchmark_layout,
)
from rag.schema.runtime import ExecutionLocationPreference


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a public benchmark.")
    parser.add_argument("--dataset", default=FIQA_DATASET, choices=[FIQA_DATASET, MEDICAL_RETRIEVAL_DATASET])
    parser.add_argument("--variant", default="full", choices=["full", "mini"])
    parser.add_argument("--queries-path", default=None)
    parser.add_argument("--qrels-path", default=None)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mode", default="mix", choices=["bypass", "naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-top-k", type=int, default=None)
    parser.add_argument("--retrieval-pool-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--rerank-pool-k", type=int, default=None)
    parser.add_argument("--disable-parent-backfill", action="store_true")
    parser.add_argument("--enable-query-understanding-llm", action="store_true")
    parser.add_argument("--split", default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--embedding-provider", default=None, choices=["local-bge", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--chat-provider", default=None, choices=["ollama", "openai-compatible", "local-hf"])
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--chat-model-path", default=None)
    parser.add_argument("--chat-backend", default=None, choices=["auto", "mlx", "transformers"])
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--rerank-model-path", default=None)
    parser.add_argument("--vector-backend", default="sqlite", choices=["sqlite", "milvus", "pgvector"])
    parser.add_argument("--vector-dsn", default=None)
    parser.add_argument("--vector-namespace", default=None)
    parser.add_argument("--vector-collection-prefix", default=None)
    args = parser.parse_args(argv)

    paths = ensure_benchmark_layout(default_benchmark_paths(args.dataset))
    spec = benchmark_dataset_spec(args.dataset)
    queries_path = (
        paths.prepared_variant_dir(args.variant) / "queries.jsonl"
        if args.queries_path is None
        else Path(args.queries_path)
    )
    qrels_path = (
        paths.prepared_variant_dir(args.variant) / "qrels.jsonl"
        if args.qrels_path is None
        else Path(args.qrels_path)
    )
    eval_dir = (
        paths.eval_variant_dir("retrieval", args.variant)
        if args.eval_dir is None
        else Path(args.eval_dir)
    )
    storage_root = Path(args.storage_root) if args.storage_root else paths.index_variant_dir(args.variant)
    storage_root.mkdir(parents=True, exist_ok=True)
    top_k = max(args.top_k, 1)
    chunk_top_k = max(args.chunk_top_k or max(top_k * 4, 40), top_k)
    retrieval_pool_k = args.retrieval_pool_k
    rerank_enabled = not args.no_rerank

    runtime = build_runtime_for_benchmark(
        storage_root=storage_root,
        profile_id=args.profile,
        require_chat=args.enable_query_understanding_llm,
        require_rerank=rerank_enabled,
        embedding_provider_kind=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_model_path=args.embedding_model_path,
        chat_provider_kind=args.chat_provider,
        chat_model=args.chat_model,
        chat_model_path=args.chat_model_path,
        chat_backend=args.chat_backend,
        rerank_model=args.rerank_model,
        rerank_model_path=args.rerank_model_path,
        vector_backend=args.vector_backend,
        vector_dsn=args.vector_dsn,
        vector_namespace=args.vector_namespace,
        vector_collection_prefix=args.vector_collection_prefix,
    )
    try:
        runtime.retrieval_service.query_understanding_service._enable_llm = args.enable_query_understanding_llm
        summary = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset=args.dataset,
            split=args.split or spec.default_split,
            retrieval_mode=args.mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            retrieval_pool_k=retrieval_pool_k,
            rerank_enabled=rerank_enabled,
            rerank_pool_k=args.rerank_pool_k,
            enable_parent_backfill=not args.disable_parent_backfill,
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
            access_policy=benchmark_access_policy(),
        ).evaluate(
            queries_path=queries_path,
            qrels_path=qrels_path,
            eval_dir=eval_dir,
            query_limit=args.query_limit,
        )
        payload = summary.as_json()
        payload["variant"] = args.variant
        payload["storage_root"] = str(storage_root)
        payload["queries_path"] = str(queries_path)
        payload["qrels_path"] = str(qrels_path)
        payload["eval_dir"] = str(eval_dir)
        payload["retrieval_pool_k"] = retrieval_pool_k
        payload["rerank_pool_k"] = args.rerank_pool_k
        payload["parent_backfill_enabled"] = not args.disable_parent_backfill
        payload["query_limit"] = args.query_limit
        payload["embedding_provider"] = args.embedding_provider
        payload["embedding_model_override"] = args.embedding_model
        payload["chat_provider"] = args.chat_provider
        payload["chat_model_override"] = args.chat_model
        payload["chat_model_path_override"] = args.chat_model_path
        payload["chat_backend_override"] = args.chat_backend
        payload["rerank_model_override"] = args.rerank_model
        payload["vector_backend"] = args.vector_backend
        payload["vector_namespace"] = args.vector_namespace
        payload["vector_collection_prefix"] = args.vector_collection_prefix
        payload["execution_location_preference"] = ExecutionLocationPreference.LOCAL_ONLY.value
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
