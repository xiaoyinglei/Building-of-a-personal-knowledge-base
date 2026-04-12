from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.benchmarks import (
    FIQA_DATASET,
    MEDICAL_RETRIEVAL_DATASET,
    default_benchmark_paths,
    download_public_benchmark,
    ensure_benchmark_layout,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a public retrieval benchmark dataset.")
    parser.add_argument("--dataset", default=FIQA_DATASET, choices=[FIQA_DATASET, MEDICAL_RETRIEVAL_DATASET])
    parser.add_argument("--raw-dir", default=None, help="Override raw download directory.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract files.")
    args = parser.parse_args()

    paths = ensure_benchmark_layout(default_benchmark_paths(args.dataset))
    raw_dir = paths.raw_dir if args.raw_dir is None else Path(args.raw_dir)
    result = download_public_benchmark(args.dataset, raw_dir, force=args.force)
    print(
        json.dumps(
            {
                "dataset": result.dataset,
                "archive_path": str(result.archive_path),
                "corpus_path": str(result.corpus_path),
                "queries_path": str(result.queries_path),
                "qrels_path": str(result.qrels_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
