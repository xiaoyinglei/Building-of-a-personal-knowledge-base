from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from rag import RAGRuntime
from rag.benchmarks import (
    benchmark_access_policy,
    benchmark_dataset_spec,
    build_runtime_for_benchmark,
    default_benchmark_paths,
    ensure_benchmark_layout,
    group_qrels_by_query,
    iter_jsonl,
    load_qrels,
    load_queries,
    write_json,
)
from rag.retrieval.analysis import narrow_access_policy_for_query
from rag.retrieval.models import QueryOptions, normalize_query_mode
from rag.schema.query import QueryUnderstanding
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference
from rag.utils.text import search_terms

_DEFAULT_BRANCHES: tuple[str, ...] = (
    "vector",
    "full_text",
    "local",
    "global",
)


@dataclass(frozen=True, slots=True)
class FailureAnalysisRecord:
    run_id: str
    dataset: str
    query_id: str
    query_text: str
    gold_doc_ids: list[str]
    predicted_doc_ids: list[str]
    hit_at_1: bool
    hit_at_10: bool
    first_relevant_rank: int | None
    failure_bucket: str
    failure_subtype: str | None
    latency_ms: float
    retrieval_mode: str
    rerank_enabled: bool
    reranked_chunk_ids: list[str] = field(default_factory=list)
    top_chunk_benchmark_doc_id: str | None = None
    unmapped_chunk_ids: list[str] = field(default_factory=list)
    mapping_debug_info: list[dict[str, object]] = field(default_factory=list)
    query_understanding_debug: dict[str, object] = field(default_factory=dict)
    heuristic_labels: list[str] = field(default_factory=list)
    branch_hit_details: dict[str, bool] = field(default_factory=dict)

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "gold_doc_ids": self.gold_doc_ids,
            "predicted_doc_ids": self.predicted_doc_ids,
            "hit_at_1": self.hit_at_1,
            "hit_at_10": self.hit_at_10,
            "first_relevant_rank": self.first_relevant_rank,
            "failure_bucket": self.failure_bucket,
            "failure_subtype": self.failure_subtype,
            "latency_ms": round(self.latency_ms, 3),
            "retrieval_mode": self.retrieval_mode,
            "rerank_enabled": self.rerank_enabled,
            "reranked_chunk_ids": self.reranked_chunk_ids,
            "top_chunk_benchmark_doc_id": self.top_chunk_benchmark_doc_id,
            "unmapped_chunk_ids": self.unmapped_chunk_ids,
            "mapping_debug_info": self.mapping_debug_info,
            "query_understanding_debug": self.query_understanding_debug,
            "heuristic_labels": self.heuristic_labels,
            "branch_hit_details": self.branch_hit_details,
        }


@dataclass(frozen=True, slots=True)
class BranchDiagnosticsRecord:
    run_id: str
    dataset: str
    query_id: str
    query_text: str
    gold_doc_ids: list[str]
    predicted_doc_ids: list[str]
    retrieval_mode: str
    rerank_enabled: bool
    branch_candidate_doc_ids: dict[str, list[str]]
    branch_candidate_benchmark_doc_ids: dict[str, list[str]]
    branch_candidate_chunk_ids: dict[str, list[str]]
    branch_hit_at_10: dict[str, bool]
    branch_overlap_with_vector: dict[str, float]
    branch_added_doc_count_vs_vector: dict[str, int]
    active_branches: list[str]
    branches_hitting_gold: list[str]
    branches_hitting_gold_only: list[str]
    fused_doc_ids: list[str]
    reranked_doc_ids: list[str]
    gold_in_fused_top_k: bool
    gold_in_reranked_top_k: bool
    fusion_lost_gold: bool
    rerank_helped: bool
    rerank_hurt: bool
    unmapped_chunk_ids: list[str] = field(default_factory=list)
    mapping_debug_info: list[dict[str, object]] = field(default_factory=list)

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "gold_doc_ids": self.gold_doc_ids,
            "predicted_doc_ids": self.predicted_doc_ids,
            "retrieval_mode": self.retrieval_mode,
            "rerank_enabled": self.rerank_enabled,
            "branch_candidate_doc_ids": self.branch_candidate_doc_ids,
            "branch_candidate_benchmark_doc_ids": self.branch_candidate_benchmark_doc_ids,
            "branch_candidate_chunk_ids": self.branch_candidate_chunk_ids,
            "branch_hit_at_10": self.branch_hit_at_10,
            "branch_overlap_with_vector": {
                name: round(value, 6) for name, value in self.branch_overlap_with_vector.items()
            },
            "branch_added_doc_count_vs_vector": self.branch_added_doc_count_vs_vector,
            "active_branches": self.active_branches,
            "branches_hitting_gold": self.branches_hitting_gold,
            "branches_hitting_gold_only": self.branches_hitting_gold_only,
            "fused_doc_ids": self.fused_doc_ids,
            "reranked_doc_ids": self.reranked_doc_ids,
            "gold_in_fused_top_k": self.gold_in_fused_top_k,
            "gold_in_reranked_top_k": self.gold_in_reranked_top_k,
            "fusion_lost_gold": self.fusion_lost_gold,
            "rerank_helped": self.rerank_helped,
            "rerank_hurt": self.rerank_hurt,
            "unmapped_chunk_ids": self.unmapped_chunk_ids,
            "mapping_debug_info": self.mapping_debug_info,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticRunContext:
    run_id: str
    dataset: str
    variant: str
    split: str
    profile_id: str
    storage_root: Path
    queries_path: Path
    qrels_path: Path
    retrieval_mode: str
    top_k: int
    chunk_top_k: int
    retrieval_pool_k: int | None
    rerank_enabled: bool
    rerank_pool_k: int | None
    enable_parent_backfill: bool
    execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_ONLY
    query_limit: int | None = None
    enable_query_understanding_llm: bool = False
    embedding_provider_kind: str | None = None
    embedding_model: str | None = None
    embedding_model_path: str | None = None
    chat_provider_kind: str | None = None
    chat_model: str | None = None
    chat_model_path: str | None = None
    chat_backend: str | None = None


@dataclass(frozen=True, slots=True)
class _BranchSnapshot:
    chunk_ids: list[str]
    doc_ids: list[str]
    benchmark_doc_ids: list[str]
    unmapped_chunk_ids: list[str]
    mapping_debug: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _QueryDiagnosticSnapshot:
    query_understanding: QueryUnderstanding
    query_understanding_debug: dict[str, object]
    active_branches: list[str]
    branch_snapshots: dict[str, _BranchSnapshot]
    fused_doc_ids: list[str]
    reranked_doc_ids: list[str]
    reranked_chunk_ids: list[str]
    top_chunk_benchmark_doc_id: str | None
    unmapped_chunk_ids: list[str]
    mapping_debug_info: list[dict[str, object]]


def _first_relevant_rank(predicted_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], top_k: int) -> int | None:
    gold = set(gold_doc_ids)
    for index, doc_id in enumerate(predicted_doc_ids[:top_k], start=1):
        if doc_id in gold:
            return index
    return None


def _looks_like_query_expression_issue(query_text: str) -> bool:
    terms = search_terms(query_text)
    if len(terms) <= 3:
        return True
    if re.search(r"\b[A-Z]{2,}[A-Z0-9-]{0,10}\b", query_text):
        return True
    if any(symbol in query_text for symbol in ("/", "-", "_", "(", ")", ":")):
        return True
    return False


def classify_failure_case(
    *,
    predicted_doc_ids: list[str],
    gold_doc_ids: list[str],
    reranked_chunk_ids: list[str],
    final_unmapped_chunk_ids: list[str],
    active_branch_hits: Mapping[str, bool],
    inactive_branch_hits: Mapping[str, bool],
    query_text: str,
    top_k: int,
) -> dict[str, object]:
    first_rank = _first_relevant_rank(predicted_doc_ids, gold_doc_ids, top_k)
    hit_at_1 = first_rank == 1
    hit_at_10 = first_rank is not None and first_rank <= top_k

    if hit_at_1:
        return {
            "hit_at_1": True,
            "hit_at_10": True,
            "first_relevant_rank": 1,
            "failure_bucket": "top1_hit",
            "failure_subtype": None,
            "heuristic_labels": [],
        }
    if hit_at_10:
        return {
            "hit_at_1": False,
            "hit_at_10": True,
            "first_relevant_rank": first_rank,
            "failure_bucket": "top10_hit_but_low_rank",
            "failure_subtype": None,
            "heuristic_labels": [],
        }
    heuristic_labels: list[str] = []
    if _looks_like_query_expression_issue(query_text):
        heuristic_labels.append("possible_query_expression_issue")
    if any(inactive_branch_hits.values()):
        heuristic_labels.append("possible_branch_coverage_issue")
    if not predicted_doc_ids:
        failure_subtype = (
            "mapping_failure"
            if final_unmapped_chunk_ids or reranked_chunk_ids
            else "empty_or_invalid_prediction"
        )
        return {
            "hit_at_1": False,
            "hit_at_10": False,
            "first_relevant_rank": None,
            "failure_bucket": "empty_or_invalid_prediction",
            "failure_subtype": failure_subtype,
            "heuristic_labels": heuristic_labels,
        }
    if final_unmapped_chunk_ids:
        failure_subtype = "mapping_failure"
    elif any(active_branch_hits.values()):
        failure_subtype = "fusion_loss"
    else:
        failure_subtype = "recall_failure"
    return {
        "hit_at_1": False,
        "hit_at_10": False,
        "first_relevant_rank": None,
        "failure_bucket": "top10_miss",
        "failure_subtype": failure_subtype,
        "heuristic_labels": heuristic_labels,
    }


def summarize_failure_records(records: Sequence[FailureAnalysisRecord]) -> dict[str, object]:
    total = len(records)
    bucket_counts = Counter(record.failure_bucket for record in records)
    subtype_counts = Counter(record.failure_subtype for record in records if record.failure_subtype)
    heuristic_counts = Counter(label for record in records for label in record.heuristic_labels)

    def ratio(count: int) -> float:
        return 0.0 if total == 0 else count / total

    return {
        "total_queries": total,
        "top1_hit_count": bucket_counts.get("top1_hit", 0),
        "top1_hit_ratio": round(ratio(bucket_counts.get("top1_hit", 0)), 6),
        "top10_hit_but_low_rank_count": bucket_counts.get("top10_hit_but_low_rank", 0),
        "top10_hit_but_low_rank_ratio": round(ratio(bucket_counts.get("top10_hit_but_low_rank", 0)), 6),
        "top10_miss_count": bucket_counts.get("top10_miss", 0),
        "top10_miss_ratio": round(ratio(bucket_counts.get("top10_miss", 0)), 6),
        "empty_or_invalid_prediction_count": bucket_counts.get("empty_or_invalid_prediction", 0),
        "empty_or_invalid_prediction_ratio": round(ratio(bucket_counts.get("empty_or_invalid_prediction", 0)), 6),
        "recall_failure_count": subtype_counts.get("recall_failure", 0),
        "recall_failure_ratio": round(ratio(subtype_counts.get("recall_failure", 0)), 6),
        "fusion_loss_count": subtype_counts.get("fusion_loss", 0),
        "fusion_loss_ratio": round(ratio(subtype_counts.get("fusion_loss", 0)), 6),
        "mapping_failure_count": subtype_counts.get("mapping_failure", 0),
        "mapping_failure_ratio": round(ratio(subtype_counts.get("mapping_failure", 0)), 6),
        "possible_query_expression_issue_count": heuristic_counts.get("possible_query_expression_issue", 0),
        "possible_query_expression_issue_ratio": round(
            ratio(heuristic_counts.get("possible_query_expression_issue", 0)), 6
        ),
        "possible_branch_coverage_issue_count": heuristic_counts.get("possible_branch_coverage_issue", 0),
        "possible_branch_coverage_issue_ratio": round(
            ratio(heuristic_counts.get("possible_branch_coverage_issue", 0)), 6
        ),
    }


def summarize_branch_records(records: Sequence[BranchDiagnosticsRecord]) -> dict[str, object]:
    total = len(records)
    branch_names = sorted({name for record in records for name in record.branch_candidate_benchmark_doc_ids})
    branch_stats: dict[str, dict[str, object]] = {}
    for branch in branch_names:
        hit_count = sum(1 for record in records if record.branch_hit_at_10.get(branch, False))
        independent_count = sum(1 for record in records if branch in record.branches_hitting_gold_only)
        overlaps = [
            record.branch_overlap_with_vector[branch]
            for record in records
            if branch in record.branch_overlap_with_vector
        ]
        added_docs = [
            record.branch_added_doc_count_vs_vector[branch]
            for record in records
            if branch in record.branch_added_doc_count_vs_vector
        ]
        branch_stats[branch] = {
            "hit_at_10_count": hit_count,
            "hit_at_10_ratio": round(0.0 if total == 0 else hit_count / total, 6),
            "independent_hit_count": independent_count,
            "independent_hit_ratio": round(0.0 if total == 0 else independent_count / total, 6),
            "avg_overlap_with_vector": round(_mean(overlaps), 6),
            "avg_added_doc_count_vs_vector": round(_mean(added_docs), 6),
        }

    low_value_branches = [
        branch
        for branch, stats in branch_stats.items()
        if branch != "vector"
        and int(stats["independent_hit_count"]) == 0
        and (
            int(stats["hit_at_10_count"]) == 0 or float(stats["avg_overlap_with_vector"]) >= 0.8
        )
    ]
    overlap_summary = {
        pair: round(_mean(values), 6)
        for pair, values in _collect_overlap_pairs(records).items()
    }
    return {
        "total_queries": total,
        "branches": branch_stats,
        "overlap_summary": overlap_summary,
        "fusion_loss_query_count": sum(1 for record in records if record.fusion_lost_gold),
        "rerank_helped_query_count": sum(1 for record in records if record.rerank_helped),
        "rerank_hurt_query_count": sum(1 for record in records if record.rerank_hurt),
        "branches_without_independent_value": low_value_branches,
    }


def analyze_recall_failure_profile(
    *,
    failure_records: Sequence[FailureAnalysisRecord],
    branch_records: Sequence[BranchDiagnosticsRecord],
    documents_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    branch_by_query = {record.query_id: record for record in branch_records}
    recall_records = [record for record in failure_records if record.failure_subtype == "recall_failure"]
    baseline_rows = [
        _query_feature_row(
            query_text=record.query_text,
            gold_doc_ids=record.gold_doc_ids,
            documents_by_id=documents_by_id,
        )
        for record in failure_records
    ]
    target_rows = [
        _query_feature_row(
            query_text=record.query_text,
            gold_doc_ids=record.gold_doc_ids,
            documents_by_id=documents_by_id,
        )
        for record in recall_records
    ]
    branch_coverage = Counter()
    samples: list[dict[str, object]] = []
    for record, feature_row in zip(recall_records, target_rows, strict=False):
        branch_record = branch_by_query.get(record.query_id)
        if branch_record is not None:
            if branch_record.branch_hit_at_10.get("full_text", False):
                branch_coverage["full_text_hit"] += 1
            if branch_record.branch_hit_at_10.get("local", False):
                branch_coverage["local_hit"] += 1
            if branch_record.branch_hit_at_10.get("global", False):
                branch_coverage["global_hit"] += 1
            if any(
                branch_record.branch_hit_at_10.get(name, False)
                for name in branch_record.branch_hit_at_10
                if name != "vector"
            ):
                branch_coverage["any_non_vector_hit"] += 1
        if len(samples) < 20:
            samples.append(
                {
                    "query_id": record.query_id,
                    "query_text": record.query_text,
                    "gold_doc_ids": record.gold_doc_ids,
                    "predicted_doc_ids": record.predicted_doc_ids,
                    "branch_hit_details": record.branch_hit_details,
                    "query_term_overlap_with_gold": round(
                        float(feature_row["query_term_overlap_with_gold"]), 6
                    ),
                    "feature_flags": _feature_flags(feature_row),
                    "heuristic_labels": record.heuristic_labels,
                }
            )
    return {
        "recall_failure_count": len(recall_records),
        "recall_failure_ratio": _safe_ratio(len(recall_records), len(failure_records)),
        "feature_summary": _summarize_query_feature_rows(target_rows),
        "feature_delta_vs_all_queries": _feature_delta_vs_baseline(target_rows, baseline_rows),
        "branch_coverage_summary": {
            "full_text_hit_count": branch_coverage.get("full_text_hit", 0),
            "local_hit_count": branch_coverage.get("local_hit", 0),
            "global_hit_count": branch_coverage.get("global_hit", 0),
            "any_non_vector_hit_count": branch_coverage.get("any_non_vector_hit", 0),
        },
        "top_terms": _top_terms(target_rows),
        "sample_queries": samples,
    }


def analyze_rerank_profile(
    *,
    branch_records: Sequence[BranchDiagnosticsRecord],
    top_k: int,
) -> dict[str, object]:
    groups = {
        "helped": [record for record in branch_records if record.rerank_helped],
        "hurt": [record for record in branch_records if record.rerank_hurt],
        "neutral": [record for record in branch_records if not record.rerank_helped and not record.rerank_hurt],
    }
    payload: dict[str, object] = {}
    for name, records in groups.items():
        feature_rows = [
            _query_feature_row(
                query_text=record.query_text,
                gold_doc_ids=record.gold_doc_ids,
                documents_by_id={},
            )
            for record in records
        ]
        vector_ranks = [
            _first_relevant_rank(
                record.branch_candidate_benchmark_doc_ids.get("vector", []),
                record.gold_doc_ids,
                top_k,
            )
            for record in records
        ]
        fused_ranks = [
            _first_relevant_rank(record.fused_doc_ids, record.gold_doc_ids, top_k)
            for record in records
        ]
        reranked_ranks = [
            _first_relevant_rank(record.reranked_doc_ids, record.gold_doc_ids, top_k)
            for record in records
        ]
        full_text_hits = sum(1 for record in records if record.branch_hit_at_10.get("full_text", False))
        vector_top3 = sum(1 for rank in vector_ranks if rank is not None and rank <= 3)
        samples: list[dict[str, object]] = []
        for record, vector_rank, fused_rank, reranked_rank in zip(
            records,
            vector_ranks,
            fused_ranks,
            reranked_ranks,
            strict=False,
        ):
            if len(samples) >= 20:
                break
            samples.append(
                {
                    "query_id": record.query_id,
                    "query_text": record.query_text,
                    "gold_doc_ids": record.gold_doc_ids,
                    "vector_first_relevant_rank": vector_rank,
                    "fused_first_relevant_rank": fused_rank,
                    "reranked_first_relevant_rank": reranked_rank,
                    "full_text_hit_at_10": record.branch_hit_at_10.get("full_text", False),
                    "full_text_added_doc_count_vs_vector": record.branch_added_doc_count_vs_vector.get(
                        "full_text", 0
                    ),
                }
            )
        payload[name] = {
            "count": len(records),
            "ratio": _safe_ratio(len(records), len(branch_records)),
            **_summarize_query_feature_rows(feature_rows),
            "avg_vector_first_relevant_rank": _mean_present(vector_ranks),
            "avg_fused_first_relevant_rank": _mean_present(fused_ranks),
            "avg_reranked_first_relevant_rank": _mean_present(reranked_ranks),
            "vector_already_top3_ratio": _safe_ratio(vector_top3, len(records)),
            "full_text_hit_ratio": _safe_ratio(full_text_hits, len(records)),
            "avg_full_text_added_doc_count": round(
                _mean(
                    record.branch_added_doc_count_vs_vector.get("full_text", 0)
                    for record in records
                ),
                6,
            ),
            "sample_queries": samples,
        }
    payload["helped_vs_hurt_delta"] = _profile_delta(
        payload.get("helped", {}),
        payload.get("hurt", {}),
        keys=(
            "short_query_ratio",
            "contains_ascii_ratio",
            "contains_digit_ratio",
            "likely_abbreviation_ratio",
            "vector_already_top3_ratio",
            "full_text_hit_ratio",
        ),
    )
    return payload


def analyze_full_text_profile(
    *,
    branch_records: Sequence[BranchDiagnosticsRecord],
    documents_by_id: Mapping[str, Mapping[str, object]],
    top_k: int,
) -> dict[str, object]:
    full_text_hit_vector_miss = [
        record
        for record in branch_records
        if record.branch_hit_at_10.get("full_text", False) and not record.branch_hit_at_10.get("vector", False)
    ]
    full_text_independent = [
        record for record in branch_records if "full_text" in record.branches_hitting_gold_only
    ]
    feature_rows = [
        _query_feature_row(
            query_text=record.query_text,
            gold_doc_ids=record.gold_doc_ids,
            documents_by_id=documents_by_id,
        )
        for record in full_text_hit_vector_miss
    ]
    samples: list[dict[str, object]] = []
    for record, feature_row in zip(full_text_hit_vector_miss, feature_rows, strict=False):
        if len(samples) >= 20:
            break
        samples.append(
            {
                "query_id": record.query_id,
                "query_text": record.query_text,
                "gold_doc_ids": record.gold_doc_ids,
                "vector_hit_at_10": record.branch_hit_at_10.get("vector", False),
                "full_text_hit_at_10": record.branch_hit_at_10.get("full_text", False),
                "query_term_overlap_with_gold": round(
                    float(feature_row["query_term_overlap_with_gold"]), 6
                ),
                "feature_flags": _feature_flags(feature_row),
            }
        )
    return {
        "full_text_independent_hit_count": len(full_text_independent),
        "full_text_independent_hit_ratio": _safe_ratio(len(full_text_independent), len(branch_records)),
        "vector_miss_full_text_hit_count": len(full_text_hit_vector_miss),
        "vector_miss_full_text_hit_ratio": _safe_ratio(len(full_text_hit_vector_miss), len(branch_records)),
        "feature_summary": _summarize_query_feature_rows(feature_rows),
        "top_terms": _top_terms(feature_rows),
        "sample_queries": samples,
        "avg_overlap_with_vector": round(
            _mean(
                record.branch_overlap_with_vector.get("full_text", 0.0)
                for record in full_text_hit_vector_miss
            ),
            6,
        ),
        "avg_full_text_added_doc_count": round(
            _mean(
                record.branch_added_doc_count_vs_vector.get("full_text", 0)
                for record in full_text_hit_vector_miss
            ),
            6,
        ),
    }


def generate_diagnostic_recommendations(
    *,
    failure_summary: Mapping[str, object],
    branch_summary: Mapping[str, object],
    recall_failure_profile: Mapping[str, object] | None = None,
    rerank_profile: Mapping[str, object] | None = None,
    full_text_profile: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    total = max(int(failure_summary.get("total_queries", 0)), 1)
    recall_failure_profile = recall_failure_profile or {}
    rerank_profile = rerank_profile or {}
    full_text_profile = full_text_profile or {}
    if int(failure_summary.get("mapping_failure_count", 0)) > 0:
        recommendations.append(
            {
                "priority": 0,
                "category": "mapping",
                "recommendation": "先修 chunk -> benchmark_doc_id 映射链路，再继续做 embedding 或 rerank 实验。",
                "rationale": {
                    "mapping_failure_count": failure_summary.get("mapping_failure_count", 0),
                    "total_queries": total,
                },
            }
        )
    helped_count = int(branch_summary.get("rerank_helped_query_count", 0))
    hurt_count = int(branch_summary.get("rerank_hurt_query_count", 0))
    if float(failure_summary.get("top10_hit_but_low_rank_ratio", 0.0)) >= 0.15 or helped_count >= max(
        1, total // 20
    ):
        recommendations.append(
            {
                "priority": 1,
                "category": "rerank",
                "recommendation": "优先优化 rerank 候选规模、特征或模型，而不是先换 embedding。",
                "rationale": {
                    "top10_hit_but_low_rank_ratio": failure_summary.get("top10_hit_but_low_rank_ratio", 0.0),
                    "rerank_helped_query_count": helped_count,
                },
            }
        )
    if hurt_count > 0 and hurt_count >= max(10, int(helped_count * 0.5)):
        recommendations.append(
            {
                "priority": 1,
                "category": "rerank_guard",
                "recommendation": (
                    "rerank 伤害样本不小，下一步优先做 rerank helped/hurt guard 实验，"
                    "而不是继续放大 rerank 候选规模。"
                ),
                "rationale": {
                    "rerank_helped_query_count": helped_count,
                    "rerank_hurt_query_count": hurt_count,
                    "hurt_vector_already_top3_ratio": (
                        rerank_profile.get("hurt", {}) or {}
                    ).get("vector_already_top3_ratio", 0.0),
                },
            }
        )
    if float(failure_summary.get("recall_failure_ratio", 0.0)) >= 0.1:
        recall_features = recall_failure_profile.get("feature_summary", {}) or {}
        recommendations.append(
            {
                "priority": 1,
                "category": "recall",
                "recommendation": (
                    "当前主问题是召回失败，下一步优先做 "
                    "query normalization / alias expansion / conditional sparse-full-text 实验，"
                    "而不是继续重写 fusion 或盲目换 embedding。"
                ),
                "rationale": {
                    "recall_failure_ratio": failure_summary.get("recall_failure_ratio", 0.0),
                    "short_query_ratio": recall_features.get("short_query_ratio", 0.0),
                    "contains_ascii_ratio": recall_features.get("contains_ascii_ratio", 0.0),
                    "contains_digit_ratio": recall_features.get("contains_digit_ratio", 0.0),
                },
            }
        )
    if int(full_text_profile.get("vector_miss_full_text_hit_count", 0)) > 0:
        recommendations.append(
            {
                "priority": 1,
                "category": "conditional_full_text",
                "recommendation": (
                    "full-text 只在少数 query 上有独立价值，下一步做条件触发式 "
                    "sparse/full-text 实验，不要恢复常驻 hybrid/mix。"
                ),
                "rationale": {
                    "full_text_independent_hit_count": full_text_profile.get("full_text_independent_hit_count", 0),
                    "vector_miss_full_text_hit_count": full_text_profile.get(
                        "vector_miss_full_text_hit_count", 0
                    ),
                    "contains_ascii_ratio": (full_text_profile.get("feature_summary", {}) or {}).get(
                        "contains_ascii_ratio", 0.0
                    ),
                    "likely_abbreviation_ratio": (full_text_profile.get("feature_summary", {}) or {}).get(
                        "likely_abbreviation_ratio", 0.0
                    ),
                },
            }
        )
    if int(branch_summary.get("fusion_loss_query_count", 0)) > 0 or float(
        failure_summary.get("fusion_loss_ratio", 0.0)
    ) >= 0.1:
        recommendations.append(
            {
                "priority": 1,
                "category": "fusion",
                "recommendation": "先检查 fusion / mode 组合，不要因为 hybrid 或 full-text 没涨分就直接下线分支。",
                "rationale": {
                    "fusion_loss_ratio": failure_summary.get("fusion_loss_ratio", 0.0),
                    "fusion_loss_query_count": branch_summary.get("fusion_loss_query_count", 0),
                },
            }
        )
    else:
        recommendations.append(
            {
                "priority": 2,
                "category": "fusion",
                "recommendation": (
                    "当前 fusion_loss=0，说明 hybrid/mix 没起效不是融合把 gold 压掉；"
                    "不应优先投入 fusion 重写。"
                ),
                "rationale": {
                    "fusion_loss_ratio": failure_summary.get("fusion_loss_ratio", 0.0),
                    "fusion_loss_query_count": branch_summary.get("fusion_loss_query_count", 0),
                },
            }
        )
    low_value_branches = list(branch_summary.get("branches_without_independent_value", []))
    if low_value_branches:
        recommendations.append(
            {
                "priority": 2,
                "category": "branch_pruning",
                "recommendation": "这些分支几乎没有独立价值，应降级或下线，不要再继续在主链路上投入调参时间。",
                "rationale": {
                    "branches_without_independent_value": low_value_branches,
                },
            }
        )
    if float(failure_summary.get("possible_query_expression_issue_ratio", 0.0)) >= 0.15:
        recommendations.append(
            {
                "priority": 2,
                "category": "query_rewrite",
                "recommendation": (
                    "失败 query 中短 query / 缩写 / 表达变体占比较高，"
                    "值得补 query rewrite / normalization。"
                ),
                "rationale": {
                    "possible_query_expression_issue_ratio": failure_summary.get(
                        "possible_query_expression_issue_ratio", 0.0
                    ),
                },
            }
        )
    return recommendations


class BenchmarkDiagnosticsPostProcessor:
    def __init__(
        self,
        *,
        runtime: RAGRuntime,
        context: DiagnosticRunContext,
        access_policy: AccessPolicy | None = None,
    ) -> None:
        self.runtime = runtime
        self.context = context
        self.access_policy = access_policy or benchmark_access_policy()

    def diagnose(self, *, diagnostics_root: Path) -> dict[str, Path]:
        queries = load_queries(self.context.queries_path)
        if self.context.query_limit is not None:
            queries = queries[: max(self.context.query_limit, 0)]
        qrels = load_qrels(self.context.qrels_path)
        gold_by_query = group_qrels_by_query(qrels)
        documents_by_id = _load_documents_by_id(self.context.queries_path.parent / "documents.jsonl")
        run_per_query = self._load_existing_per_query()
        eligible_queries = [query for query in queries if query.query_id in gold_by_query]

        output_dir = diagnostics_root / self.context.dataset / self.context.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "failure_analysis.jsonl"
        branch_path = output_dir / "branch_diagnostics.jsonl"
        failure_summary_path = output_dir / "failure_summary.json"
        branch_summary_path = output_dir / "branch_summary.json"
        recall_failure_profile_path = output_dir / "recall_failure_profile.json"
        rerank_profile_path = output_dir / "rerank_profile.json"
        full_text_profile_path = output_dir / "full_text_profile.json"
        recommendations_path = output_dir / "recommendations.json"

        failure_records: list[FailureAnalysisRecord] = []
        branch_records: list[BranchDiagnosticsRecord] = []
        with failure_path.open("w", encoding="utf-8") as failure_handle, branch_path.open(
            "w", encoding="utf-8"
        ) as branch_handle:
            for query_record in tqdm(
                eligible_queries,
                total=len(eligible_queries),
                desc=f"Diagnosing {self.context.dataset} retrieval",
                unit="query",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ):
                gold = gold_by_query[query_record.query_id]
                snapshot = self._collect_query_snapshot(query_record.query_text)
                existing = run_per_query.get(query_record.query_id, {})
                predicted_doc_ids = list(
                    existing.get("predicted_doc_ids") or snapshot.reranked_doc_ids[: self.context.top_k]
                )
                latency_ms = _coerce_float(existing.get("latency_ms")) or 0.0
                branch_hit_at_10 = {
                    name: _branch_hits_gold(snapshot.branch_snapshots[name].benchmark_doc_ids, gold, self.context.top_k)
                    for name in snapshot.branch_snapshots
                }
                active_branch_hits = {
                    name: branch_hit_at_10[name]
                    for name in snapshot.active_branches
                    if name in branch_hit_at_10
                }
                inactive_branch_hits = {
                    name: hit
                    for name, hit in branch_hit_at_10.items()
                    if name not in snapshot.active_branches
                }
                gold_in_fused_top_k = _branch_hits_gold(snapshot.fused_doc_ids, gold, self.context.top_k)
                gold_in_reranked_top_k = _branch_hits_gold(snapshot.reranked_doc_ids, gold, self.context.top_k)
                rerank_helped, rerank_hurt = _rerank_delta(
                    fused_doc_ids=snapshot.fused_doc_ids,
                    reranked_doc_ids=snapshot.reranked_doc_ids,
                    gold_doc_ids=sorted(gold),
                    top_k=self.context.top_k,
                )
                classification = classify_failure_case(
                    predicted_doc_ids=predicted_doc_ids,
                    gold_doc_ids=sorted(gold),
                    reranked_chunk_ids=snapshot.reranked_chunk_ids,
                    final_unmapped_chunk_ids=snapshot.unmapped_chunk_ids,
                    active_branch_hits=active_branch_hits,
                    inactive_branch_hits=inactive_branch_hits,
                    query_text=query_record.query_text,
                    top_k=self.context.top_k,
                )
                failure_record = FailureAnalysisRecord(
                    run_id=self.context.run_id,
                    dataset=self.context.dataset,
                    query_id=query_record.query_id,
                    query_text=query_record.query_text,
                    gold_doc_ids=sorted(gold),
                    predicted_doc_ids=predicted_doc_ids,
                    hit_at_1=bool(classification["hit_at_1"]),
                    hit_at_10=bool(classification["hit_at_10"]),
                    first_relevant_rank=classification["first_relevant_rank"],
                    failure_bucket=str(classification["failure_bucket"]),
                    failure_subtype=_coerce_optional_str(classification.get("failure_subtype")),
                    latency_ms=latency_ms,
                    retrieval_mode=self.context.retrieval_mode,
                    rerank_enabled=self.context.rerank_enabled,
                    reranked_chunk_ids=snapshot.reranked_chunk_ids,
                    top_chunk_benchmark_doc_id=snapshot.top_chunk_benchmark_doc_id,
                    unmapped_chunk_ids=snapshot.unmapped_chunk_ids,
                    mapping_debug_info=snapshot.mapping_debug_info,
                    query_understanding_debug=snapshot.query_understanding_debug,
                    heuristic_labels=list(classification["heuristic_labels"]),
                    branch_hit_details=branch_hit_at_10,
                )
                branch_record = BranchDiagnosticsRecord(
                    run_id=self.context.run_id,
                    dataset=self.context.dataset,
                    query_id=query_record.query_id,
                    query_text=query_record.query_text,
                    gold_doc_ids=sorted(gold),
                    predicted_doc_ids=predicted_doc_ids,
                    retrieval_mode=self.context.retrieval_mode,
                    rerank_enabled=self.context.rerank_enabled,
                    branch_candidate_doc_ids={
                        name: snapshot.branch_snapshots[name].doc_ids for name in snapshot.branch_snapshots
                    },
                    branch_candidate_benchmark_doc_ids={
                        name: snapshot.branch_snapshots[name].benchmark_doc_ids
                        for name in snapshot.branch_snapshots
                    },
                    branch_candidate_chunk_ids={
                        name: snapshot.branch_snapshots[name].chunk_ids for name in snapshot.branch_snapshots
                    },
                    branch_hit_at_10=branch_hit_at_10,
                    branch_overlap_with_vector=_branch_overlap_with_vector(
                        snapshot.branch_snapshots,
                        self.context.top_k,
                    ),
                    branch_added_doc_count_vs_vector=_branch_added_doc_count(
                        snapshot.branch_snapshots,
                        self.context.top_k,
                    ),
                    active_branches=snapshot.active_branches,
                    branches_hitting_gold=_branches_hitting_gold(snapshot.branch_snapshots, gold, self.context.top_k),
                    branches_hitting_gold_only=_branches_hitting_gold_only(
                        snapshot.branch_snapshots,
                        gold,
                        self.context.top_k,
                    ),
                    fused_doc_ids=snapshot.fused_doc_ids[: self.context.top_k],
                    reranked_doc_ids=snapshot.reranked_doc_ids[: self.context.top_k],
                    gold_in_fused_top_k=gold_in_fused_top_k,
                    gold_in_reranked_top_k=gold_in_reranked_top_k,
                    fusion_lost_gold=(not gold_in_reranked_top_k) and any(active_branch_hits.values()),
                    rerank_helped=rerank_helped,
                    rerank_hurt=rerank_hurt,
                    unmapped_chunk_ids=snapshot.unmapped_chunk_ids,
                    mapping_debug_info=snapshot.mapping_debug_info,
                )
                failure_handle.write(json.dumps(failure_record.as_json(), ensure_ascii=False) + "\n")
                branch_handle.write(json.dumps(branch_record.as_json(), ensure_ascii=False) + "\n")
                failure_records.append(failure_record)
                branch_records.append(branch_record)
        failure_summary = summarize_failure_records(failure_records)
        branch_summary = summarize_branch_records(branch_records)
        recall_failure_profile = analyze_recall_failure_profile(
            failure_records=failure_records,
            branch_records=branch_records,
            documents_by_id=documents_by_id,
        )
        rerank_profile = analyze_rerank_profile(
            branch_records=branch_records,
            top_k=self.context.top_k,
        )
        full_text_profile = analyze_full_text_profile(
            branch_records=branch_records,
            documents_by_id=documents_by_id,
            top_k=self.context.top_k,
        )
        recommendations = generate_diagnostic_recommendations(
            failure_summary=failure_summary,
            branch_summary=branch_summary,
            recall_failure_profile=recall_failure_profile,
            rerank_profile=rerank_profile,
            full_text_profile=full_text_profile,
        )
        write_json(failure_summary_path, failure_summary)
        write_json(branch_summary_path, branch_summary)
        write_json(recall_failure_profile_path, recall_failure_profile)
        write_json(rerank_profile_path, rerank_profile)
        write_json(full_text_profile_path, full_text_profile)
        write_json(
            recommendations_path,
            {
                "recommendations": recommendations,
                "decision_basis": {
                    "failure_summary": failure_summary,
                    "branch_summary": branch_summary,
                    "recall_failure_profile": {
                        "recall_failure_count": recall_failure_profile.get("recall_failure_count", 0),
                        "feature_summary": recall_failure_profile.get("feature_summary", {}),
                    },
                    "rerank_profile": {
                        "helped_count": (rerank_profile.get("helped", {}) or {}).get("count", 0),
                        "hurt_count": (rerank_profile.get("hurt", {}) or {}).get("count", 0),
                    },
                    "full_text_profile": {
                        "full_text_independent_hit_count": full_text_profile.get(
                            "full_text_independent_hit_count", 0
                        ),
                        "vector_miss_full_text_hit_count": full_text_profile.get(
                            "vector_miss_full_text_hit_count", 0
                        ),
                    },
                },
            },
        )
        return {
            "failure_analysis": failure_path,
            "failure_summary": failure_summary_path,
            "branch_diagnostics": branch_path,
            "branch_summary": branch_summary_path,
            "recall_failure_profile": recall_failure_profile_path,
            "rerank_profile": rerank_profile_path,
            "full_text_profile": full_text_profile_path,
            "recommendations": recommendations_path,
        }

    def _load_existing_per_query(self) -> dict[str, dict[str, object]]:
        run_dir = ensure_benchmark_layout(default_benchmark_paths(self.context.dataset)).eval_variant_dir(
            "retrieval", self.context.variant
        ) / "runs" / self.context.run_id
        path = run_dir / "per_query.jsonl"
        if not path.exists():
            return {}
        return {
            _coerce_required_str(record.get("query_id"), field_name="query_id"): record
            for record in iter_jsonl(path)
        }

    def _collect_query_snapshot(self, query_text: str) -> _QueryDiagnosticSnapshot:
        retrieval_service = self.runtime.retrieval_service
        query_options = QueryOptions(
            mode=self.context.retrieval_mode,
            top_k=self.context.top_k,
            chunk_top_k=self.context.chunk_top_k,
            retrieval_pool_k=self.context.retrieval_pool_k,
            enable_rerank=self.context.rerank_enabled,
            rerank_pool_k=self.context.rerank_pool_k,
            enable_parent_backfill=self.context.enable_parent_backfill,
            max_context_tokens=self.runtime.token_contract.max_context_tokens,
        )
        retrieval_service._prepare_retriever_policies(
            access_policy=self.access_policy,
            execution_location_preference=self.context.execution_location_preference,
        )
        query_understanding, effective_access_policy, decision, plan = retrieval_service.plan_query(
            query_text,
            access_policy=self.access_policy,
            source_scope=[],
            execution_location_preference=self.context.execution_location_preference,
            query_options=query_options,
        )
        active_collection = retrieval_service.collect_internal_branches(
            plan=plan,
            source_scope=[],
            access_policy=effective_access_policy,
            runtime_mode=decision.runtime_mode,
            query_understanding=query_understanding,
        )
        active_branches = active_collection.branches
        active_branch_map = {name: list(candidates) for name, candidates in active_branches}
        fused_candidates = retrieval_service.fusion.fuse(
            query=query_text,
            mode=plan.mode,
            branches=active_branches,
        )
        rank_result = retrieval_service.rank_plan_branches(
            query=query_text,
            plan=plan,
            branches=active_branches,
            query_options=query_options,
            rerank_required=decision.rerank_required,
        )
        reranked_candidates = rank_result.candidates
        retrieval_limit = max(
            query_options.retrieval_pool_k or query_options.chunk_top_k or query_options.top_k,
            query_options.top_k,
        )
        branch_limit_map = {path.branch: path.limit for path in plan.retrieval_paths}
        branch_snapshots: dict[str, _BranchSnapshot] = {}
        active_branch_names = [name for name, _candidates in active_branches]
        diagnostic_branches = set(_DEFAULT_BRANCHES) | set(active_branch_names)
        if query_understanding.needs_structure:
            diagnostic_branches.add("section")
        if query_understanding.needs_metadata:
            diagnostic_branches.add("metadata")
        if query_understanding.needs_special:
            diagnostic_branches.add("special")
        for branch in sorted(diagnostic_branches):
            if branch in active_branch_map:
                limited = active_branch_map[branch]
            else:
                limited = retrieval_service.collect_branch_candidates(
                    branch=branch,
                    plan=plan,
                    query_understanding=query_understanding,
                    source_scope=[],
                    access_policy=effective_access_policy,
                    runtime_mode=decision.runtime_mode,
                    limit=branch_limit_map.get(branch, retrieval_limit),
                )
            branch_snapshots[branch] = _snapshot_branch(limited)
        top_chunk_benchmark_doc_id = (
            None
            if not reranked_candidates
            else _normalize_doc_id(getattr(reranked_candidates[0], "benchmark_doc_id", None))
        )
        unmapped_chunk_ids = [
            candidate.chunk_id
            for candidate in reranked_candidates[: self.context.top_k]
            if _normalize_doc_id(getattr(candidate, "benchmark_doc_id", None)) is None
        ]
        mapping_debug_info = [
            _mapping_debug(candidate)
            for candidate in reranked_candidates[: self.context.top_k]
            if _normalize_doc_id(getattr(candidate, "benchmark_doc_id", None)) is None
        ]
        query_debug = retrieval_service.query_understanding_service.diagnostics_payload()
        return _QueryDiagnosticSnapshot(
            query_understanding=query_understanding,
            query_understanding_debug=query_debug,
            active_branches=active_branch_names,
            branch_snapshots=branch_snapshots,
            fused_doc_ids=retrieval_service._benchmark_doc_ids(fused_candidates),
            reranked_doc_ids=retrieval_service._benchmark_doc_ids(reranked_candidates),
            reranked_chunk_ids=[candidate.chunk_id for candidate in reranked_candidates],
            top_chunk_benchmark_doc_id=top_chunk_benchmark_doc_id,
            unmapped_chunk_ids=unmapped_chunk_ids,
            mapping_debug_info=mapping_debug_info,
        )



def _rerank_delta(
    *,
    fused_doc_ids: Sequence[str],
    reranked_doc_ids: Sequence[str],
    gold_doc_ids: Sequence[str],
    top_k: int,
) -> tuple[bool, bool]:
    fused_rank = _first_relevant_rank(list(fused_doc_ids), list(gold_doc_ids), top_k)
    reranked_rank = _first_relevant_rank(list(reranked_doc_ids), list(gold_doc_ids), top_k)
    if fused_rank is None and reranked_rank is None:
        return False, False
    if fused_rank is None and reranked_rank is not None:
        return True, False
    if fused_rank is not None and reranked_rank is None:
        return False, True
    if reranked_rank is not None and fused_rank is not None and reranked_rank < fused_rank:
        return True, False
    if reranked_rank is not None and fused_rank is not None and reranked_rank > fused_rank:
        return False, True
    return False, False

def _snapshot_branch(candidates: Sequence[Any]) -> _BranchSnapshot:
    chunk_ids: list[str] = []
    doc_ids: list[str] = []
    benchmark_doc_ids: list[str] = []
    unmapped_chunk_ids: list[str] = []
    mapping_debug: list[dict[str, object]] = []
    seen_doc_ids: set[str] = set()
    seen_benchmark_doc_ids: set[str] = set()
    for candidate in candidates:
        chunk_ids.append(candidate.chunk_id)
        normalized_doc_id = _normalize_doc_id(getattr(candidate, "doc_id", None))
        if normalized_doc_id is not None and normalized_doc_id not in seen_doc_ids:
            seen_doc_ids.add(normalized_doc_id)
            doc_ids.append(normalized_doc_id)
        normalized_benchmark_doc_id = _normalize_doc_id(getattr(candidate, "benchmark_doc_id", None))
        if normalized_benchmark_doc_id is None:
            unmapped_chunk_ids.append(candidate.chunk_id)
            mapping_debug.append(_mapping_debug(candidate))
        elif normalized_benchmark_doc_id not in seen_benchmark_doc_ids:
            seen_benchmark_doc_ids.add(normalized_benchmark_doc_id)
            benchmark_doc_ids.append(normalized_benchmark_doc_id)
    return _BranchSnapshot(
        chunk_ids=chunk_ids,
        doc_ids=doc_ids,
        benchmark_doc_ids=benchmark_doc_ids,
        unmapped_chunk_ids=unmapped_chunk_ids,
        mapping_debug=mapping_debug,
    )


def _mapping_debug(candidate: Any) -> dict[str, object]:
    metadata = getattr(candidate, "metadata", None) or {}
    return {
        "chunk_id": getattr(candidate, "chunk_id", None),
        "doc_id": getattr(candidate, "doc_id", None),
        "benchmark_doc_id": getattr(candidate, "benchmark_doc_id", None),
        "chunk_metadata_benchmark_doc_id": metadata.get("benchmark_doc_id"),
        "parent_doc_id": metadata.get("parent_doc_id"),
        "source_id": getattr(candidate, "source_id", None),
        "source_kind": getattr(candidate, "source_kind", None),
    }


def _normalize_doc_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _branch_hits_gold(doc_ids: Sequence[str], gold: Iterable[str] | Mapping[str, int], top_k: int) -> bool:
    gold_ids = set(gold) if not isinstance(gold, Mapping) else {doc_id for doc_id, score in gold.items() if score > 0}
    return any(doc_id in gold_ids for doc_id in doc_ids[:top_k])


def _branches_hitting_gold(
    branch_snapshots: Mapping[str, _BranchSnapshot],
    gold: Iterable[str] | Mapping[str, int],
    top_k: int,
) -> list[str]:
    return [
        branch
        for branch, snapshot in branch_snapshots.items()
        if _branch_hits_gold(snapshot.benchmark_doc_ids, gold, top_k)
    ]


def _branches_hitting_gold_only(
    branch_snapshots: Mapping[str, _BranchSnapshot],
    gold: Iterable[str] | Mapping[str, int],
    top_k: int,
) -> list[str]:
    hit_branches = _branches_hitting_gold(branch_snapshots, gold, top_k)
    return hit_branches if len(hit_branches) == 1 else []


def _branch_overlap_with_vector(branch_snapshots: Mapping[str, _BranchSnapshot], top_k: int) -> dict[str, float]:
    vector_docs = set(branch_snapshots.get("vector", _BranchSnapshot([], [], [], [], [])).benchmark_doc_ids[:top_k])
    overlaps: dict[str, float] = {}
    for branch, snapshot in branch_snapshots.items():
        if branch == "vector":
            continue
        branch_docs = set(snapshot.benchmark_doc_ids[:top_k])
        union = vector_docs | branch_docs
        overlaps[branch] = 0.0 if not union else len(vector_docs & branch_docs) / len(union)
    return overlaps


def _branch_added_doc_count(branch_snapshots: Mapping[str, _BranchSnapshot], top_k: int) -> dict[str, int]:
    vector_docs = set(branch_snapshots.get("vector", _BranchSnapshot([], [], [], [], [])).benchmark_doc_ids[:top_k])
    return {
        branch: len(set(snapshot.benchmark_doc_ids[:top_k]) - vector_docs)
        for branch, snapshot in branch_snapshots.items()
        if branch != "vector"
    }


_QUERY_STOP_TERMS = {
    "的",
    "了",
    "吗",
    "么",
    "呢",
    "和",
    "与",
    "及",
    "请问",
    "如何",
    "什么",
    "多少",
    "怎么",
    "为什么",
    "是否",
    "可以",
    "能",
    "会",
    "不会",
}
_MEDICAL_HINT_TERMS = (
    "病",
    "症",
    "炎",
    "癌",
    "瘤",
    "综合征",
    "治疗",
    "药",
    "手术",
    "血压",
    "肿瘤",
    "感染",
    "婴儿",
    "母乳",
)


def _query_feature_row(
    *,
    query_text: str,
    gold_doc_ids: Sequence[str],
    documents_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    text = query_text.strip()
    visible_units = _query_units(text)
    overlap_terms = [term for term in search_terms(text) if term]
    unique_overlap_terms = list(dict.fromkeys(overlap_terms))
    gold_term_coverage = _gold_term_overlap(unique_overlap_terms, gold_doc_ids, documents_by_id)
    contains_ascii = bool(re.search(r"[A-Za-z]", text))
    contains_digit = bool(re.search(r"\d", text))
    contains_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    likely_abbreviation = bool(
        re.search(r"\b[A-Z]{2,}[A-Z0-9-]{0,10}\b", text)
        or (contains_ascii and len(visible_units) <= 4)
        or any(symbol in text for symbol in ("/", "-", "_", "(", ")", ":"))
    )
    return {
        "query_text": text,
        "terms": visible_units,
        "char_length": len(text),
        "token_count": len(visible_units),
        "short_query": len(visible_units) <= 4,
        "contains_ascii": contains_ascii,
        "contains_digit": contains_digit,
        "contains_cjk": contains_cjk,
        "mixed_script": contains_ascii and contains_cjk,
        "likely_abbreviation": likely_abbreviation,
        "contains_medical_hint": any(hint in text for hint in _MEDICAL_HINT_TERMS),
        "contains_negation_or_question": any(
            token in text for token in ("不", "没", "无", "否", "吗", "会不会", "可不可以")
        ),
        "multi_clause_like": any(token in text for token in ("，", ",", "；", ";", "和", "及", "并且", "还是")),
        "query_term_overlap_with_gold": gold_term_coverage,
    }


def _query_units(text: str) -> list[str]:
    units: list[str] = []
    for match in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text):
        normalized = match.strip().lower()
        if normalized:
            units.append(normalized)
    return list(dict.fromkeys(units))


def _gold_term_overlap(
    query_terms: Sequence[str],
    gold_doc_ids: Sequence[str],
    documents_by_id: Mapping[str, Mapping[str, object]],
) -> float:
    if not query_terms:
        return 0.0
    best_overlap = 0.0
    query_term_set = set(query_terms)
    for doc_id in gold_doc_ids:
        document = documents_by_id.get(doc_id)
        if not document:
            continue
        gold_text = f"{document.get('title', '')} {document.get('text', '')}"
        gold_terms = set(search_terms(gold_text))
        if not gold_terms:
            continue
        overlap = len(query_term_set & gold_terms) / len(query_term_set)
        if overlap > best_overlap:
            best_overlap = overlap
    return round(best_overlap, 6)


def _summarize_query_feature_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "avg_query_char_length": 0.0,
            "avg_query_token_count": 0.0,
            "short_query_ratio": 0.0,
            "contains_ascii_ratio": 0.0,
            "contains_digit_ratio": 0.0,
            "mixed_script_ratio": 0.0,
            "likely_abbreviation_ratio": 0.0,
            "contains_medical_hint_ratio": 0.0,
            "contains_negation_or_question_ratio": 0.0,
            "multi_clause_like_ratio": 0.0,
            "avg_query_term_overlap_with_gold": 0.0,
        }
    total = len(rows)
    return {
        "avg_query_char_length": round(_mean(float(row["char_length"]) for row in rows), 6),
        "avg_query_token_count": round(_mean(float(row["token_count"]) for row in rows), 6),
        "short_query_ratio": round(_safe_ratio(sum(1 for row in rows if bool(row["short_query"])), total), 6),
        "contains_ascii_ratio": round(_safe_ratio(sum(1 for row in rows if bool(row["contains_ascii"])), total), 6),
        "contains_digit_ratio": round(_safe_ratio(sum(1 for row in rows if bool(row["contains_digit"])), total), 6),
        "mixed_script_ratio": round(_safe_ratio(sum(1 for row in rows if bool(row["mixed_script"])), total), 6),
        "likely_abbreviation_ratio": round(
            _safe_ratio(sum(1 for row in rows if bool(row["likely_abbreviation"])), total), 6
        ),
        "contains_medical_hint_ratio": round(
            _safe_ratio(sum(1 for row in rows if bool(row["contains_medical_hint"])), total), 6
        ),
        "contains_negation_or_question_ratio": round(
            _safe_ratio(sum(1 for row in rows if bool(row["contains_negation_or_question"])), total), 6
        ),
        "multi_clause_like_ratio": round(
            _safe_ratio(sum(1 for row in rows if bool(row["multi_clause_like"])), total), 6
        ),
        "avg_query_term_overlap_with_gold": round(
            _mean(float(row["query_term_overlap_with_gold"]) for row in rows),
            6,
        ),
    }


def _feature_delta_vs_baseline(
    target_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    target_summary = _summarize_query_feature_rows(target_rows)
    baseline_summary = _summarize_query_feature_rows(baseline_rows)
    keys = (
        "short_query_ratio",
        "contains_ascii_ratio",
        "contains_digit_ratio",
        "mixed_script_ratio",
        "likely_abbreviation_ratio",
        "contains_medical_hint_ratio",
        "contains_negation_or_question_ratio",
        "multi_clause_like_ratio",
        "avg_query_term_overlap_with_gold",
    )
    return _profile_delta(target_summary, baseline_summary, keys=keys)


def _profile_delta(
    target_summary: Mapping[str, object],
    baseline_summary: Mapping[str, object],
    *,
    keys: Sequence[str],
) -> dict[str, float]:
    return {
        key: round(float(target_summary.get(key, 0.0)) - float(baseline_summary.get(key, 0.0)), 6)
        for key in keys
    }


def _feature_flags(feature_row: Mapping[str, object]) -> list[str]:
    flags: list[str] = []
    for key in (
        "short_query",
        "contains_ascii",
        "contains_digit",
        "mixed_script",
        "likely_abbreviation",
        "contains_medical_hint",
        "contains_negation_or_question",
        "multi_clause_like",
    ):
        if bool(feature_row.get(key)):
            flags.append(key)
    return flags


def _top_terms(rows: Sequence[Mapping[str, object]], *, limit: int = 20) -> list[dict[str, object]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for term in row.get("terms", []):
            if not isinstance(term, str):
                continue
            normalized = term.strip()
            if not normalized or normalized in _QUERY_STOP_TERMS:
                continue
            if len(normalized) == 1 and not normalized.isdigit():
                continue
            counter[normalized] += 1
    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(limit)
    ]


def _mean_present(values: Sequence[int | None]) -> float:
    present = [float(value) for value in values if value is not None]
    return round(_mean(present), 6)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _load_documents_by_id(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    documents: dict[str, dict[str, object]] = {}
    for record in iter_jsonl(path):
        doc_id = _coerce_required_str(record.get("doc_id"), field_name="doc_id")
        documents[doc_id] = record
    return documents


def _collect_overlap_pairs(records: Sequence[BranchDiagnosticsRecord]) -> dict[str, list[float]]:
    pairs: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for branch, overlap in record.branch_overlap_with_vector.items():
            pairs[f"vector__{branch}"].append(overlap)
    return pairs


def _mean(values: Sequence[float] | Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _coerce_required_str(value: object, *, field_name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"Missing required string field: {field_name}")


def _coerce_optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def load_run_summary(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_diagnostic_context(
    *,
    dataset: str,
    run_id: str,
    variant: str,
    profile_id: str,
    storage_root: Path,
    queries_path: Path,
    qrels_path: Path,
    retrieval_mode: str,
    top_k: int,
    chunk_top_k: int,
    retrieval_pool_k: int | None,
    rerank_enabled: bool,
    rerank_pool_k: int | None,
    enable_parent_backfill: bool,
    split: str | None = None,
    query_limit: int | None = None,
    enable_query_understanding_llm: bool = False,
    embedding_provider_kind: str | None = None,
    embedding_model: str | None = None,
    embedding_model_path: str | None = None,
    chat_provider_kind: str | None = None,
    chat_model: str | None = None,
    chat_model_path: str | None = None,
    chat_backend: str | None = None,
) -> DiagnosticRunContext:
    spec = benchmark_dataset_spec(dataset)
    return DiagnosticRunContext(
        run_id=run_id,
        dataset=dataset,
        variant=variant,
        split=split or spec.default_split,
        profile_id=profile_id,
        storage_root=storage_root,
        queries_path=queries_path,
        qrels_path=qrels_path,
        retrieval_mode=normalize_query_mode(retrieval_mode).value,
        top_k=max(top_k, 1),
        chunk_top_k=max(chunk_top_k, top_k),
        retrieval_pool_k=retrieval_pool_k,
        rerank_enabled=rerank_enabled,
        rerank_pool_k=rerank_pool_k,
        enable_parent_backfill=enable_parent_backfill,
        query_limit=query_limit,
        enable_query_understanding_llm=enable_query_understanding_llm,
        embedding_provider_kind=embedding_provider_kind,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        chat_provider_kind=chat_provider_kind,
        chat_model=chat_model,
        chat_model_path=chat_model_path,
        chat_backend=chat_backend,
    )


def build_runtime_for_diagnostics(context: DiagnosticRunContext) -> RAGRuntime:
    runtime = build_runtime_for_benchmark(
        storage_root=context.storage_root,
        profile_id=context.profile_id,
        require_chat=context.enable_query_understanding_llm,
        require_rerank=context.rerank_enabled,
        embedding_provider_kind=context.embedding_provider_kind,
        embedding_model=context.embedding_model,
        embedding_model_path=context.embedding_model_path,
        chat_provider_kind=context.chat_provider_kind,
        chat_model=context.chat_model,
        chat_model_path=context.chat_model_path,
        chat_backend=context.chat_backend,
    )
    runtime.retrieval_service.query_understanding_service._enable_llm = context.enable_query_understanding_llm
    return runtime


__all__ = [
    "analyze_full_text_profile",
    "analyze_recall_failure_profile",
    "analyze_rerank_profile",
    "BenchmarkDiagnosticsPostProcessor",
    "BranchDiagnosticsRecord",
    "DiagnosticRunContext",
    "FailureAnalysisRecord",
    "build_diagnostic_context",
    "build_runtime_for_diagnostics",
    "classify_failure_case",
    "generate_diagnostic_recommendations",
    "load_run_summary",
    "summarize_branch_records",
    "summarize_failure_records",
]
