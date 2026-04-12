from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.benchmarks import (
    FIQA_DATASET,
    MEDICAL_RETRIEVAL_DATASET,
    benchmark_dataset_spec,
    default_benchmark_paths,
    ensure_benchmark_layout,
    prepare_public_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a public retrieval benchmark into project JSONL format.")
    parser.add_argument("--dataset", default=FIQA_DATASET, choices=[FIQA_DATASET, MEDICAL_RETRIEVAL_DATASET])
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--no-mini", action="store_true")
    parser.add_argument("--mini-query-count", type=int, default=None)
    parser.add_argument("--mini-doc-count", type=int, default=None)
    args = parser.parse_args()

    paths = ensure_benchmark_layout(default_benchmark_paths(args.dataset))
    raw_dir = paths.raw_dir if args.raw_dir is None else Path(args.raw_dir)
    prepared_root = paths.prepared_root if args.prepared_dir is None else Path(args.prepared_dir)
    spec = benchmark_dataset_spec(args.dataset)
    results = prepare_public_benchmark(
        args.dataset,
        raw_dir,
        prepared_root,
        split=args.split or spec.default_split,
        build_mini=not args.no_mini,
        mini_query_count=args.mini_query_count,
        mini_target_doc_count=args.mini_doc_count,
    )
    print(
        json.dumps(
            {
                variant: {
                    "dataset": result.dataset,
                    "split": result.split,
                    "document_count": result.document_count,
                    "query_count": result.query_count,
                    "qrel_count": result.qrel_count,
                    "documents_path": str(result.documents_path),
                    "queries_path": str(result.queries_path),
                    "qrels_path": str(result.qrels_path),
                }
                for variant, result in results.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
