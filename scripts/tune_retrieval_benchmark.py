from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from rag.benchmarks import (
    MEDICAL_RETRIEVAL_DATASET,
    RetrievalBenchmarkEvaluator,
    benchmark_access_policy,
    benchmark_dataset_spec,
    build_prepared_query_slice_subset,
    build_runtime_for_benchmark,
    default_benchmark_paths,
    ensure_benchmark_layout,
    profile_benchmark_ingest,
)
from rag.schema.runtime import ExecutionLocationPreference


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str
    top_k: int
    chunk_top_k: int
    retrieval_pool_k: int | None
    rerank_enabled: bool
    rerank_pool_k: int | None
    enable_parent_backfill: bool
    chunk_token_size: int
    chunk_overlap_tokens: int
    storage_root: Path


def _tradeoff_choice(
    rows: list[dict[str, Any]],
    *,
    score_key: str = "NDCG@10",
    latency_key: str = "avg_latency_ms",
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rows to choose from")
    best_score = max(float(row[score_key]) for row in rows)
    threshold = best_score * 0.98
    shortlist = [row for row in rows if float(row[score_key]) >= threshold]
    shortlist.sort(key=lambda row: (float(row[latency_key]), -float(row[score_key])))
    return shortlist[0]


def _evaluate_config(
    *,
    dataset: str,
    variant: str,
    profile_id: str,
    queries_path: Path,
    qrels_path: Path,
    eval_dir: Path,
    query_limit: int | None,
    config: RetrievalConfig,
    embedding_provider: str | None,
    embedding_model: str | None,
    embedding_model_path: str | None,
) -> dict[str, Any]:
    runtime = build_runtime_for_benchmark(
        storage_root=config.storage_root,
        profile_id=profile_id,
        require_chat=False,
        require_rerank=config.rerank_enabled,
        chunk_token_size=config.chunk_token_size,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
        embedding_provider_kind=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
    )
    try:
        runtime.retrieval_service.query_understanding_service._enable_llm = False
        summary = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset=dataset,
            split=benchmark_dataset_spec(dataset).default_split,
            retrieval_mode=config.mode,
            top_k=config.top_k,
            chunk_top_k=config.chunk_top_k,
            retrieval_pool_k=config.retrieval_pool_k,
            rerank_enabled=config.rerank_enabled,
            rerank_pool_k=config.rerank_pool_k,
            enable_parent_backfill=config.enable_parent_backfill,
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
            access_policy=benchmark_access_policy(),
        ).evaluate(
            queries_path=queries_path,
            qrels_path=qrels_path,
            eval_dir=eval_dir,
            query_limit=query_limit,
        )
        payload = summary.as_json()
        payload.update(
            {
                "variant": variant,
                "query_limit": query_limit,
                "storage_root": str(config.storage_root),
                "mode": config.mode,
                "chunk_top_k": config.chunk_top_k,
                "retrieval_pool_k": config.retrieval_pool_k,
                "rerank_enabled": config.rerank_enabled,
                "rerank_pool_k": config.rerank_pool_k,
                "enable_parent_backfill": config.enable_parent_backfill,
                "chunk_token_size": config.chunk_token_size,
                "chunk_overlap_tokens": config.chunk_overlap_tokens,
            }
        )
        return payload
    finally:
        runtime.close()


def _ensure_chunk_index(
    *,
    dataset: str,
    variant: str,
    profile_id: str,
    documents_path: Path,
    storage_root: Path,
    ingest_batch_size: int,
    embedding_batch_size: int,
    embedding_device: str | None,
    embedding_provider: str | None,
    embedding_model: str | None,
    embedding_model_path: str | None,
    chunk_token_size: int,
    chunk_overlap_tokens: int,
) -> None:
    metadata_db = storage_root / "metadata.sqlite3"
    if metadata_db.exists():
        return
    profile_benchmark_ingest(
        dataset=dataset,
        profile_id=profile_id,
        documents_path=documents_path,
        storage_root=storage_root,
        doc_limit=10_000_000,
        ingest_batch_size=ingest_batch_size,
        encode_batch_size=embedding_batch_size,
        ingest_strategy="stream",
        embedding_device=embedding_device,
        embedding_provider_kind=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        chunk_token_size=chunk_token_size,
        chunk_overlap_tokens=chunk_overlap_tokens,
        skip_graph_extraction=True,
        log_embedding_calls=False,
        show_backend_progress=False,
    )


def _write_markdown_record(
    *,
    path: Path,
    dataset: str,
    variant: str,
    baseline: RetrievalConfig,
    sections: list[tuple[str, list[dict[str, Any]], dict[str, Any]]],
    ingest_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# {dataset} {variant} Retrieval Tuning Record",
        "",
        "## Baseline",
        "",
        f"- fixed mode: `{baseline.mode}`",
        f"- fixed top_k: `{baseline.top_k}`",
        f"- fixed chunk_top_k: `{baseline.chunk_top_k}`",
        f"- fixed retrieval_pool_k: `{baseline.retrieval_pool_k}`",
        f"- fixed rerank: `{baseline.rerank_enabled}`",
        f"- fixed rerank_pool_k: `{baseline.rerank_pool_k}`",
        f"- fixed parent_backfill: `{baseline.enable_parent_backfill}`",
        f"- fixed chunk_token_size: `{baseline.chunk_token_size}`",
        f"- fixed chunk_overlap_tokens: `{baseline.chunk_overlap_tokens}`",
        "",
    ]
    if ingest_rows:
        lines.extend(
            [
                "## Ingest Strategy / Throughput",
                "",
                "| strategy | docs | chunk_size | overlap | ingest_batch | encode_batch | "
                "total_elapsed_ms | embedding_elapsed_ms | docs/s | chunks/s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ingest_rows:
            lines.append(
                "| {ingest_strategy} | {doc_count} | {chunk_token_size} | {chunk_overlap_tokens} | "
                "{ingest_batch_size} | {encode_batch_size} | {total_elapsed_ms:.3f} | "
                "{embedding_elapsed_ms:.3f} | {docs_per_second:.3f} | {chunks_per_second:.3f} |".format(**row)
            )
        lines.append("")
    for title, rows, choice in sections:
        lines.extend(
            [
                f"## {title}",
                "",
                "| value | Recall@10 | MRR@10 | NDCG@10 | avg_latency_ms | p95_latency_ms |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {label} | {Recall@10:.6f} | {MRR@10:.6f} | {NDCG@10:.6f} | "
                "{avg_latency_ms:.3f} | {p95_latency_ms:.3f} |".format(
                    **row
                )
            )
        lines.extend(
            [
                "",
                f"- selected: `{choice['label']}`",
                "- rationale: keep `NDCG@10` within 98% of best in this group, "
                "then choose lower `avg_latency_ms`.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled retrieval tuning sweeps on a benchmark variant.")
    parser.add_argument("--dataset", default=MEDICAL_RETRIEVAL_DATASET, choices=[MEDICAL_RETRIEVAL_DATASET])
    parser.add_argument("--variant", default="mini", choices=["full", "mini"])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--query-limit", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode-values", type=_parse_str_list, default=["naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--chunk-top-k-values", type=_parse_int_list, default=[10, 20, 30, 40, 60])
    parser.add_argument("--rerank-pool-values", type=_parse_int_list, default=[10, 20, 30, 40])
    parser.add_argument("--chunk-size-values", type=_parse_int_list, default=[128, 256, 480])
    parser.add_argument("--chunk-overlap-values", type=_parse_int_list, default=[0, 32, 64])
    parser.add_argument("--chunk-target-doc-count", type=int, default=1000)
    parser.add_argument("--embedding-device", default="mps")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-provider", default=None, choices=["local-bge", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--ingest-batch-size", type=int, default=32)
    parser.add_argument("--force-reingest", action="store_true")
    args = parser.parse_args()

    paths = ensure_benchmark_layout(default_benchmark_paths(args.dataset), tasks=("retrieval", "ingest"))
    queries_path = paths.prepared_variant_dir(args.variant) / "queries.jsonl"
    qrels_path = paths.prepared_variant_dir(args.variant) / "qrels.jsonl"
    documents_path = paths.prepared_variant_dir(args.variant) / "documents.jsonl"
    chunk_subset_dir = paths.subset_dir(f"{args.variant}-query{args.query_limit}-chunk")
    chunk_subset = build_prepared_query_slice_subset(
        dataset=args.dataset,
        source_prepared_dir=paths.prepared_variant_dir(args.variant),
        target_prepared_dir=chunk_subset_dir,
        query_limit=max(args.query_limit or 0, 0),
        target_doc_count=max(args.chunk_target_doc_count, 1),
    )
    storage_root = Path(args.storage_root) if args.storage_root else paths.index_variant_dir(args.variant)
    tuning_dir = paths.eval_variant_dir("retrieval", args.variant) / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = tuning_dir / "tuning_results.csv"
    ingest_csv = tuning_dir / "ingest_strategy_results.csv"
    tuning_record = tuning_dir / "tuning_record.md"
    tuning_jsonl = tuning_dir / "tuning_results.jsonl"

    base = RetrievalConfig(
        mode="naive",
        top_k=max(args.top_k, 1),
        chunk_top_k=40,
        retrieval_pool_k=40,
        rerank_enabled=False,
        rerank_pool_k=None,
        enable_parent_backfill=True,
        chunk_token_size=480,
        chunk_overlap_tokens=64,
        storage_root=storage_root,
    )

    ingest_rows: list[dict[str, Any]] = []
    for strategy in ("stream", "preload"):
        run_root = paths.index_root / "profile" / args.variant / f"strategy-{strategy}"
        if run_root.exists():
            shutil.rmtree(run_root)
        result = profile_benchmark_ingest(
            dataset=args.dataset,
            profile_id=args.profile,
            documents_path=documents_path,
            storage_root=run_root,
            doc_limit=1000,
            ingest_batch_size=args.ingest_batch_size,
            encode_batch_size=args.embedding_batch_size,
            ingest_strategy=strategy,
            embedding_device=args.embedding_device,
            embedding_provider_kind=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_model_path=args.embedding_model_path,
            chunk_token_size=base.chunk_token_size,
            chunk_overlap_tokens=base.chunk_overlap_tokens,
            skip_graph_extraction=True,
        )
        row = result.as_json()
        row["chunk_token_size"] = base.chunk_token_size
        row["chunk_overlap_tokens"] = base.chunk_overlap_tokens
        ingest_rows.append(row)

    with ingest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ingest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ingest_rows)

    sections: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []

    # The rest of the sweeps are chained from previous selection.
    # We build them after each section chooses a tradeoff config.
    current = base

    with tuning_csv.open("w", encoding="utf-8", newline="") as csv_handle:
        fieldnames = [
            "experiment",
            "label",
            "dataset",
            "variant",
            "query_limit",
            "mode",
            "top_k",
            "chunk_top_k",
            "retrieval_pool_k",
            "rerank_enabled",
            "rerank_pool_k",
            "enable_parent_backfill",
            "chunk_token_size",
            "chunk_overlap_tokens",
            "Recall@10",
            "MRR@10",
            "NDCG@10",
            "avg_latency_ms",
            "p95_latency_ms",
            "storage_root",
            "run_id",
        ]
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()

        def run_group(title: str, configs: list[RetrievalConfig], labeler: callable) -> dict[str, Any]:
            rows: list[dict[str, Any]] = []
            for config in tqdm(configs, desc=title, unit="run"):
                if args.force_reingest and config.storage_root != storage_root and config.storage_root.exists():
                    shutil.rmtree(config.storage_root)
                if config.storage_root != storage_root:
                    _ensure_chunk_index(
                        dataset=args.dataset,
                        variant=args.variant,
                        profile_id=args.profile,
                        documents_path=chunk_subset.documents_path
                        if config.chunk_token_size != base.chunk_token_size
                        or config.chunk_overlap_tokens != base.chunk_overlap_tokens
                        else documents_path,
                        storage_root=config.storage_root,
                        ingest_batch_size=args.ingest_batch_size,
                        embedding_batch_size=args.embedding_batch_size,
                        embedding_device=args.embedding_device,
                        embedding_provider=args.embedding_provider,
                        embedding_model=args.embedding_model,
                        embedding_model_path=args.embedding_model_path,
                        chunk_token_size=config.chunk_token_size,
                        chunk_overlap_tokens=config.chunk_overlap_tokens,
                    )
                payload = _evaluate_config(
                    dataset=args.dataset,
                    variant=args.variant,
                    profile_id=args.profile,
                    queries_path=queries_path,
                    qrels_path=qrels_path,
                    eval_dir=paths.eval_variant_dir("retrieval", args.variant),
                    query_limit=args.query_limit,
                    config=config,
                    embedding_provider=args.embedding_provider,
                    embedding_model=args.embedding_model,
                    embedding_model_path=args.embedding_model_path,
                )
                payload["experiment"] = title
                payload["label"] = labeler(config)
                rows.append(payload)
                writer.writerow({key: payload.get(key) for key in fieldnames})
                csv_handle.flush()
                with tuning_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            choice = _tradeoff_choice(rows)
            sections.append((title, rows, choice))
            return choice

        choice = run_group("Mode sweep", [replace(base, mode=value) for value in args.mode_values], lambda c: c.mode)
        current = replace(current, mode=str(choice["mode"]))

        choice = run_group(
            "chunk_top_k sweep",
            [
                replace(current, chunk_top_k=value, retrieval_pool_k=value, rerank_enabled=False, rerank_pool_k=None)
                for value in args.chunk_top_k_values
            ],
            lambda c: f"chunk_top_k={c.chunk_top_k}",
        )
        current = replace(
            current,
            chunk_top_k=int(choice["chunk_top_k"]),
            retrieval_pool_k=int(choice["retrieval_pool_k"]),
        )

        choice = run_group(
            "rerank pool sweep",
            [replace(current, rerank_enabled=False, rerank_pool_k=None)]
            + [
                replace(current, rerank_enabled=True, rerank_pool_k=value)
                for value in args.rerank_pool_values
            ],
            lambda c: "rerank=off" if not c.rerank_enabled else f"rerank_pool_k={c.rerank_pool_k}",
        )
        current = replace(
            current,
            rerank_enabled=bool(choice["rerank_enabled"]),
            rerank_pool_k=None if choice["rerank_pool_k"] in (None, "", "None") else int(choice["rerank_pool_k"]),
        )

        choice = run_group(
            "parent backfill sweep",
            [
                replace(current, enable_parent_backfill=True),
                replace(current, enable_parent_backfill=False),
            ],
            lambda c: f"parent_backfill={c.enable_parent_backfill}",
        )
        current = replace(current, enable_parent_backfill=bool(choice["enable_parent_backfill"]))

        chunk_base_root = paths.index_root / "tuning" / args.variant
        choice = run_group(
            "chunk size sweep",
            [
                replace(
                    current,
                    chunk_token_size=value,
                    chunk_overlap_tokens=32,
                    storage_root=(chunk_base_root / f"chunk-size-{value}-overlap-32"),
                )
                for value in args.chunk_size_values
            ],
            lambda c: f"chunk_size={c.chunk_token_size}",
        )
        current = replace(
            current,
            chunk_token_size=int(choice["chunk_token_size"]),
            chunk_overlap_tokens=int(choice["chunk_overlap_tokens"]),
            storage_root=Path(str(choice["storage_root"])),
        )

        overlap_size = 256 if 256 in args.chunk_size_values else current.chunk_token_size
        choice = run_group(
            "overlap sweep",
            [
                replace(
                    current,
                    chunk_token_size=overlap_size,
                    chunk_overlap_tokens=value,
                    storage_root=(chunk_base_root / f"chunk-size-{overlap_size}-overlap-{value}"),
                )
                for value in args.chunk_overlap_values
            ],
            lambda c: f"overlap={c.chunk_overlap_tokens}",
        )
        current = replace(
            current,
            chunk_token_size=int(choice["chunk_token_size"]),
            chunk_overlap_tokens=int(choice["chunk_overlap_tokens"]),
            storage_root=Path(str(choice["storage_root"])),
        )

    _write_markdown_record(
        path=tuning_record,
        dataset=args.dataset,
        variant=args.variant,
        baseline=base,
        sections=sections,
        ingest_rows=ingest_rows,
    )

    recommended = {
        "dataset": args.dataset,
        "variant": args.variant,
        "recommended": asdict(current) | {"storage_root": str(current.storage_root)},
        "tuning_record": str(tuning_record),
        "tuning_csv": str(tuning_csv),
        "ingest_csv": str(ingest_csv),
    }
    print(json.dumps(recommended, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
