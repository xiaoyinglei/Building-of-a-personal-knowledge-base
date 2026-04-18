from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag.answer_benchmarks import AnswerBenchmarkEvaluator, build_chat_judge
from rag.benchmarks import (
    FIQA_DATASET,
    MEDICAL_RETRIEVAL_DATASET,
    build_runtime_for_benchmark,
    default_benchmark_paths,
    ensure_benchmark_layout,
)
from rag.schema.runtime import ExecutionLocationPreference


def _resolve_default_answer_index(
    *,
    dataset: str,
    variant: str,
    storage_root: str | None,
    vector_backend: str | None,
    vector_collection_prefix: str | None,
    embedding_provider: str | None,
    embedding_model: str | None,
    paths,
) -> tuple[Path, str | None, str | None]:
    if storage_root is not None:
        resolved_root = Path(storage_root)
        resolved_backend = vector_backend
        resolved_prefix = vector_collection_prefix
        if resolved_backend is None and "milvus" in resolved_root.name:
            resolved_backend = "milvus"
        if resolved_prefix is None:
            if resolved_root.name == "mini-milvus-bge-v2":
                resolved_prefix = "medical_retrieval_mini_bge_v2"
            elif resolved_root.name == "mini-milvus-qwen8b-v1":
                resolved_prefix = "medical_retrieval_mini_qwen8b_v1"
        return resolved_root, resolved_backend, resolved_prefix

    if dataset == MEDICAL_RETRIEVAL_DATASET and variant == "mini":
        if embedding_provider == "ollama" and embedding_model == "qwen3-embedding:8b":
            return (
                Path("data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1"),
                vector_backend or "milvus",
                vector_collection_prefix or "medical_retrieval_mini_qwen8b_v1",
            )
        return (
            Path("data/benchmarks/medical_retrieval/index/mini-milvus-bge-v2"),
            vector_backend or "milvus",
            vector_collection_prefix or "medical_retrieval_mini_bge_v2",
        )

    return paths.index_variant_dir(variant), vector_backend, vector_collection_prefix


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Evaluate answer quality on a public benchmark using the formal RAG path."
    )
    parser.add_argument("--dataset", default=FIQA_DATASET, choices=[FIQA_DATASET, MEDICAL_RETRIEVAL_DATASET])
    parser.add_argument("--variant", default="mini", choices=["full", "mini"])
    parser.add_argument("--documents-path", default=None)
    parser.add_argument("--queries-path", default=None)
    parser.add_argument("--qrels-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mode", default="naive", choices=["bypass", "naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-top-k", type=int, default=20)
    parser.add_argument("--retrieval-pool-k", type=int, default=20)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--rerank-pool-k", type=int, default=10)
    parser.add_argument("--answer-context-top-k", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--judge-subset-size", type=int, default=250)
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--local-judge-profile", default="local_full")
    parser.add_argument("--review-judge-profile", default="local_retrieval_cloud_chat")
    parser.add_argument("--disable-review", action="store_true")
    parser.add_argument("--review-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--embedding-provider", default=None, choices=["local-bge", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--chat-provider", default=None, choices=["ollama", "openai-compatible", "local-hf"])
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--chat-model-path", default=None)
    parser.add_argument("--chat-backend", default=None, choices=["auto", "mlx", "transformers"])
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--rerank-model-path", default=None)
    parser.add_argument("--vector-backend", default=None, choices=["sqlite", "milvus", "pgvector"])
    parser.add_argument("--vector-dsn", default=None)
    parser.add_argument("--vector-namespace", default=None)
    parser.add_argument("--vector-collection-prefix", default=None)
    args = parser.parse_args(argv)

    paths = ensure_benchmark_layout(default_benchmark_paths(args.dataset), tasks=("retrieval", "ingest"))
    prepared_dir = paths.prepared_variant_dir(args.variant)
    documents_path = prepared_dir / "documents.jsonl" if args.documents_path is None else Path(args.documents_path)
    queries_path = prepared_dir / "queries.jsonl" if args.queries_path is None else Path(args.queries_path)
    qrels_path = prepared_dir / "qrels.jsonl" if args.qrels_path is None else Path(args.qrels_path)
    output_root = (
        Path("data") / "eval" / "answers" / args.dataset / args.variant
        if args.output_root is None
        else Path(args.output_root)
    )
    storage_root, resolved_vector_backend, resolved_collection_prefix = _resolve_default_answer_index(
        dataset=args.dataset,
        variant=args.variant,
        storage_root=args.storage_root,
        vector_backend=args.vector_backend,
        vector_collection_prefix=args.vector_collection_prefix,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        paths=paths,
    )
    storage_root.mkdir(parents=True, exist_ok=True)

    runtime = build_runtime_for_benchmark(
        storage_root=storage_root,
        profile_id=args.profile,
        require_chat=True,
        require_rerank=not args.no_rerank,
        embedding_provider_kind=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_model_path=args.embedding_model_path,
        rerank_model=args.rerank_model,
        rerank_model_path=args.rerank_model_path,
        chat_provider_kind=args.chat_provider,
        chat_model=args.chat_model,
        chat_model_path=args.chat_model_path,
        chat_backend=args.chat_backend,
        vector_backend=resolved_vector_backend,
        vector_dsn=args.vector_dsn,
        vector_namespace=args.vector_namespace,
        vector_collection_prefix=resolved_collection_prefix,
    )
    try:
        local_judge = build_chat_judge(profile_id=args.local_judge_profile, allow_missing=False)
        review_judge = (
            None
            if args.disable_review
            else build_chat_judge(profile_id=args.review_judge_profile, require_cloud=True, allow_missing=True)
        )
        result = AnswerBenchmarkEvaluator(
            runtime=runtime,
            dataset=args.dataset,
            variant=args.variant,
            retrieval_mode=args.mode,
            top_k=max(args.top_k, 1),
            chunk_top_k=max(args.chunk_top_k, max(args.top_k, 1)),
            retrieval_pool_k=args.retrieval_pool_k,
            rerank_enabled=not args.no_rerank,
            rerank_pool_k=args.rerank_pool_k,
            answer_context_top_k=args.answer_context_top_k,
            judge_subset_size=max(args.judge_subset_size, 0),
            judge_seed=args.judge_seed,
            local_judge=local_judge,
            review_judge=review_judge,
            review_confidence_threshold=args.review_confidence_threshold,
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
        ).evaluate(
            queries_path=queries_path,
            qrels_path=qrels_path,
            documents_path=documents_path,
            output_root=output_root,
            query_limit=args.query_limit,
        )
        payload = dict(result["summary"])
        payload.update(
            {
                "run_id": result["run_id"],
                "run_dir": result["run_dir"],
                "documents_path": str(documents_path),
                "queries_path": str(queries_path),
                "qrels_path": str(qrels_path),
                "storage_root": str(storage_root),
                "output_root": str(output_root),
                "vector_backend": resolved_vector_backend,
                "vector_namespace": args.vector_namespace,
                "vector_collection_prefix": resolved_collection_prefix,
                "chat_provider": args.chat_provider,
                "chat_model_override": args.chat_model,
                "chat_model_path_override": args.chat_model_path,
                "chat_backend_override": args.chat_backend,
                "rerank_model_override": args.rerank_model,
                "local_judge_profile": args.local_judge_profile,
                "review_judge_profile": None if args.disable_review else args.review_judge_profile,
            }
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
