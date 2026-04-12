from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag import CapabilityRequirements
from rag.assembly import CapabilityAssemblyService
from rag.benchmarks import (
    _embedding_model_name,
    _mean,
    _p95,
    _progress,
    append_jsonl,
    benchmark_run_id,
    group_qrels_by_query,
    iter_jsonl,
    load_qrels,
    load_queries,
    write_json,
)
from rag.retrieval import QueryOptions
from rag.runtime import RAGRuntime
from rag.schema.query import GroundedAnswer
from rag.schema.runtime import ExecutionLocationPreference

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_MEDICAL_HINT_TOKENS = (
    "痛",
    "炎",
    "癌",
    "瘤",
    "症",
    "咳",
    "热",
    "痒",
    "麻",
    "胀",
    "血",
    "便",
    "尿",
    "头晕",
    "头痛",
    "发烧",
    "月经",
    "怀孕",
)

EvidenceConsistencyLabel = Literal[
    "grounded_answer_with_citations",
    "grounded_answer_without_citations",
    "appropriate_abstain_due_to_retrieval_gap",
    "missed_answer_despite_retrieval_hit",
    "unsupported_answer",
]
JudgeVerdict = Literal["correct", "partially_correct", "incorrect", "insufficient", "unclear"]


@dataclass(frozen=True, slots=True)
class AnswerPerQueryRecord:
    run_id: str
    dataset: str
    query_id: str
    query_text: str
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    cited_doc_ids: list[str]
    answer_text: str
    citation_count: int
    groundedness_flag: bool
    insufficient_evidence_flag: bool
    evidence_consistency_label: EvidenceConsistencyLabel
    retrieval_hit_at_10: bool
    citation_hit_at_10: bool
    latency_ms: float
    generation_provider: str | None
    generation_model: str | None

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "gold_doc_ids": self.gold_doc_ids,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "cited_doc_ids": self.cited_doc_ids,
            "answer_text": self.answer_text,
            "citation_count": self.citation_count,
            "groundedness_flag": self.groundedness_flag,
            "insufficient_evidence_flag": self.insufficient_evidence_flag,
            "evidence_consistency_label": self.evidence_consistency_label,
            "retrieval_hit_at_10": self.retrieval_hit_at_10,
            "citation_hit_at_10": self.citation_hit_at_10,
            "latency_ms": round(self.latency_ms, 3),
            "generation_provider": self.generation_provider,
            "generation_model": self.generation_model,
        }


@dataclass(frozen=True, slots=True)
class AnswerJudgeDecision:
    verdict: JudgeVerdict
    confidence: float
    rationale: str
    provider: str
    model: str | None

    def as_json(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 6),
            "rationale": self.rationale,
            "provider": self.provider,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class AnswerJudgeRecord:
    run_id: str
    dataset: str
    query_id: str
    query_text: str
    answer_text: str
    gold_doc_ids: list[str]
    gold_reference_texts: list[str]
    local_decision: AnswerJudgeDecision
    review_required: bool
    review_decision: AnswerJudgeDecision | None = None
    final_verdict: JudgeVerdict = "unclear"
    evidence_consistency_label: EvidenceConsistencyLabel = "unsupported_answer"
    retrieval_hit_at_10: bool = False

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "answer_text": self.answer_text,
            "gold_doc_ids": self.gold_doc_ids,
            "gold_reference_texts": self.gold_reference_texts,
            "local_decision": self.local_decision.as_json(),
            "review_required": self.review_required,
            "review_decision": None if self.review_decision is None else self.review_decision.as_json(),
            "final_verdict": self.final_verdict,
            "evidence_consistency_label": self.evidence_consistency_label,
            "retrieval_hit_at_10": self.retrieval_hit_at_10,
        }


@dataclass(frozen=True, slots=True)
class AnswerRunSummary:
    run_id: str
    dataset: str
    variant: str
    query_count: int
    judge_subset_count: int
    embedding_model: str | None
    generation_provider: str | None
    generation_model: str | None
    retrieval_mode: str
    rerank_enabled: bool
    answer_context_top_k: int | None
    evidence_consistent_rate: float
    grounded_answer_rate: float
    citation_presence_rate: float
    citation_gold_hit_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    answer_correct_rate: float | None = None

    @property
    def queries_per_second(self) -> float:
        if self.avg_latency_ms <= 0:
            return 0.0
        return 1000.0 / self.avg_latency_ms

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "variant": self.variant,
            "query_count": self.query_count,
            "judge_subset_count": self.judge_subset_count,
            "embedding_model": self.embedding_model,
            "generation_provider": self.generation_provider,
            "generation_model": self.generation_model,
            "retrieval_mode": self.retrieval_mode,
            "rerank_enabled": self.rerank_enabled,
            "answer_context_top_k": self.answer_context_top_k,
            "evidence_consistent_rate": round(self.evidence_consistent_rate, 6),
            "grounded_answer_rate": round(self.grounded_answer_rate, 6),
            "citation_presence_rate": round(self.citation_presence_rate, 6),
            "citation_gold_hit_rate": round(self.citation_gold_hit_rate, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "queries_per_second": round(self.queries_per_second, 3),
            "answer_correct_rate": None if self.answer_correct_rate is None else round(self.answer_correct_rate, 6),
        }


@dataclass(frozen=True, slots=True)
class _PreparedDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class _JudgeBinding:
    provider: str
    model: str | None
    chat: Callable[[str], str]


class AnswerBenchmarkEvaluator:
    def __init__(
        self,
        *,
        runtime: RAGRuntime,
        dataset: str,
        variant: str,
        retrieval_mode: str,
        top_k: int,
        chunk_top_k: int,
        retrieval_pool_k: int | None,
        rerank_enabled: bool,
        rerank_pool_k: int | None,
        answer_context_top_k: int | None = None,
        judge_subset_size: int = 250,
        judge_seed: int = 42,
        local_judge: _JudgeBinding | None = None,
        review_judge: _JudgeBinding | None = None,
        review_confidence_threshold: float = 0.75,
        execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_ONLY,
    ) -> None:
        self.runtime = runtime
        self.dataset = dataset
        self.variant = variant
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k
        self.retrieval_pool_k = retrieval_pool_k
        self.rerank_enabled = rerank_enabled
        self.rerank_pool_k = rerank_pool_k
        self.answer_context_top_k = answer_context_top_k
        self.judge_subset_size = judge_subset_size
        self.judge_seed = judge_seed
        self.local_judge = local_judge
        self.review_judge = review_judge
        self.review_confidence_threshold = review_confidence_threshold
        self.execution_location_preference = execution_location_preference

    def evaluate(
        self,
        *,
        queries_path: Path,
        qrels_path: Path,
        documents_path: Path,
        output_root: Path,
        query_limit: int | None = None,
    ) -> dict[str, object]:
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = answer_benchmark_run_id(self.dataset)
        run_dir = output_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        queries = load_queries(queries_path)
        if query_limit is not None:
            queries = queries[: max(query_limit, 0)]
        qrels = load_qrels(qrels_path)
        gold_by_query = group_qrels_by_query(qrels)
        documents = load_prepared_documents(documents_path)

        per_query_records: list[AnswerPerQueryRecord] = []
        latencies: list[float] = []
        per_query_path = run_dir / "per_query_answers.jsonl"
        cumulative_per_query_path = output_root / "per_query_answers.jsonl"

        with per_query_path.open("w", encoding="utf-8") as handle:
            for query_record in _progress(
                queries,
                total=len(queries),
                desc=f"Evaluating {self.dataset} answers",
                unit="query",
            ):
                gold_relevances = gold_by_query.get(query_record.query_id)
                if not gold_relevances:
                    continue
                started = time.perf_counter()
                result = self.runtime.query(
                    query_record.query_text,
                    options=QueryOptions(
                        mode=self.retrieval_mode,
                        top_k=self.top_k,
                        chunk_top_k=self.chunk_top_k,
                        retrieval_pool_k=self.retrieval_pool_k,
                        enable_rerank=self.rerank_enabled,
                        rerank_pool_k=self.rerank_pool_k,
                        answer_context_top_k=self.answer_context_top_k,
                    ),
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                latencies.append(latency_ms)
                record = build_answer_record(
                    run_id=run_id,
                    dataset=self.dataset,
                    query_id=query_record.query_id,
                    query_text=query_record.query_text,
                    result=result,
                    gold_doc_ids=sorted(gold_relevances),
                    latency_ms=latency_ms,
                    top_k=self.top_k,
                )
                handle.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
                append_jsonl(cumulative_per_query_path, record.as_json())
                per_query_records.append(record)

        evidence_summary = summarize_evidence_consistency(per_query_records)
        write_json(run_dir / "evidence_consistency_summary.json", evidence_summary)

        judge_records: list[AnswerJudgeRecord] = []
        judge_summary: dict[str, object] = {
            "subset_query_count": 0,
            "review_enabled": self.review_judge is not None,
            "review_executed_count": 0,
            "note": "Judge stage skipped because no local judge is configured."
            if self.local_judge is None
            else None,
        }
        judge_subset_path = run_dir / "judge_subset.jsonl"
        if self.local_judge is not None:
            judge_subset = select_answer_judge_subset(
                per_query_records,
                subset_size=self.judge_subset_size,
                seed=self.judge_seed,
            )
            try:
                with judge_subset_path.open("w", encoding="utf-8") as handle:
                    for record in _progress(
                        judge_subset,
                        total=len(judge_subset),
                        desc=f"Judging {self.dataset} answers",
                        unit="query",
                    ):
                        gold_reference_texts = _gold_reference_texts(record.gold_doc_ids, documents)
                        local_decision = run_answer_judge(
                            self.local_judge,
                            query_text=record.query_text,
                            answer_text=record.answer_text,
                            gold_reference_texts=gold_reference_texts,
                        )
                        review_required = should_review_local_judge(
                            record,
                            local_decision,
                            confidence_threshold=self.review_confidence_threshold,
                        )
                        review_decision: AnswerJudgeDecision | None = None
                        if review_required and self.review_judge is not None:
                            try:
                                review_decision = run_answer_judge(
                                    self.review_judge,
                                    query_text=record.query_text,
                                    answer_text=record.answer_text,
                                    gold_reference_texts=gold_reference_texts,
                                )
                            except Exception as exc:
                                review_decision = None
                                judge_summary = _judge_failure_summary(
                                    subset_query_count=len(judge_subset),
                                    completed_judge_count=len(judge_records),
                                    review_enabled=True,
                                    local_judge=self.local_judge,
                                    exc=exc,
                                    stage="review",
                                )
                        final_verdict = (
                            review_decision.verdict if review_decision is not None else local_decision.verdict
                        )
                        judge_record = AnswerJudgeRecord(
                            run_id=run_id,
                            dataset=self.dataset,
                            query_id=record.query_id,
                            query_text=record.query_text,
                            answer_text=record.answer_text,
                            gold_doc_ids=record.gold_doc_ids,
                            gold_reference_texts=gold_reference_texts,
                            local_decision=local_decision,
                            review_required=review_required,
                            review_decision=review_decision,
                            final_verdict=final_verdict,
                            evidence_consistency_label=record.evidence_consistency_label,
                            retrieval_hit_at_10=record.retrieval_hit_at_10,
                        )
                        handle.write(json.dumps(judge_record.as_json(), ensure_ascii=False) + "\n")
                        judge_records.append(judge_record)
                if "judge_stage_error" not in judge_summary:
                    judge_summary = summarize_answer_judging(judge_records)
            except Exception as exc:
                judge_summary = _judge_failure_summary(
                    subset_query_count=len(judge_subset),
                    completed_judge_count=len(judge_records),
                    review_enabled=self.review_judge is not None,
                    local_judge=self.local_judge,
                    exc=exc,
                    stage="local",
                )
            write_json(run_dir / "judge_summary.json", judge_summary)

        recommendations = generate_answer_recommendations(
            evidence_summary=evidence_summary,
            judge_summary=judge_summary,
        )
        write_json(run_dir / "answer_recommendations.json", {"recommendations": recommendations})

        summary = AnswerRunSummary(
            run_id=run_id,
            dataset=self.dataset,
            variant=self.variant,
            query_count=len(per_query_records),
            judge_subset_count=len(judge_records),
            embedding_model=_embedding_model_name(self.runtime),
            generation_provider=_first_non_blank(record.generation_provider for record in per_query_records),
            generation_model=_first_non_blank(record.generation_model for record in per_query_records),
            retrieval_mode=self.retrieval_mode,
            rerank_enabled=self.rerank_enabled,
            answer_context_top_k=self.answer_context_top_k,
            evidence_consistent_rate=float(evidence_summary["evidence_consistent_ratio"]),
            grounded_answer_rate=float(evidence_summary["grounded_answer_rate"]),
            citation_presence_rate=float(evidence_summary["citation_presence_rate"]),
            citation_gold_hit_rate=float(evidence_summary["citation_gold_hit_rate"]),
            avg_latency_ms=_mean(latencies),
            p95_latency_ms=_p95(latencies),
            answer_correct_rate=judge_summary.get("final_correct_ratio") if judge_records else None,
        )
        append_answer_baseline_row(output_root / "baseline.csv", summary)
        append_jsonl(output_root / "run_history.jsonl", summary.as_json())
        write_json(output_root / "run_summary.json", summary.as_json())
        write_json(run_dir / "run_summary.json", summary.as_json())

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "summary": summary.as_json(),
            "evidence_consistency_summary_path": str(run_dir / "evidence_consistency_summary.json"),
            "judge_summary_path": str(run_dir / "judge_summary.json"),
            "recommendations_path": str(run_dir / "answer_recommendations.json"),
        }


def classify_evidence_consistency(
    *,
    retrieval_hit_at_10: bool,
    citation_count: int,
    citation_hit_at_10: bool,
    groundedness_flag: bool,
    insufficient_evidence_flag: bool,
) -> EvidenceConsistencyLabel:
    if insufficient_evidence_flag and not retrieval_hit_at_10:
        return "appropriate_abstain_due_to_retrieval_gap"
    if insufficient_evidence_flag and retrieval_hit_at_10:
        return "missed_answer_despite_retrieval_hit"
    if groundedness_flag and citation_count > 0:
        return "grounded_answer_with_citations"
    if groundedness_flag and citation_count == 0:
        return "grounded_answer_without_citations"
    return "unsupported_answer"


def build_answer_record(
    *,
    run_id: str,
    dataset: str,
    query_id: str,
    query_text: str,
    result: Any,
    gold_doc_ids: list[str],
    latency_ms: float,
    top_k: int,
) -> AnswerPerQueryRecord:
    answer: GroundedAnswer = result.answer
    retrieved_doc_ids = list(result.retrieval.reranked_benchmark_doc_ids[:top_k])
    cited_doc_ids = [
        citation_id
        for citation in answer.citations
        for citation_id in [_normalize_doc_id(citation.benchmark_doc_id) or _normalize_doc_id(citation.doc_id)]
        if citation_id is not None
    ]
    citation_hit_at_10 = any(doc_id in set(gold_doc_ids) for doc_id in cited_doc_ids)
    retrieval_hit_at_10 = any(doc_id in set(gold_doc_ids) for doc_id in retrieved_doc_ids)
    label = classify_evidence_consistency(
        retrieval_hit_at_10=retrieval_hit_at_10,
        citation_count=len(answer.citations),
        citation_hit_at_10=citation_hit_at_10,
        groundedness_flag=answer.groundedness_flag,
        insufficient_evidence_flag=answer.insufficient_evidence_flag,
    )
    return AnswerPerQueryRecord(
        run_id=run_id,
        dataset=dataset,
        query_id=query_id,
        query_text=query_text,
        gold_doc_ids=gold_doc_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        cited_doc_ids=cited_doc_ids,
        answer_text=answer.answer_text,
        citation_count=len(answer.citations),
        groundedness_flag=answer.groundedness_flag,
        insufficient_evidence_flag=answer.insufficient_evidence_flag,
        evidence_consistency_label=label,
        retrieval_hit_at_10=retrieval_hit_at_10,
        citation_hit_at_10=citation_hit_at_10,
        latency_ms=latency_ms,
        generation_provider=result.generation_provider,
        generation_model=result.generation_model,
    )


def summarize_evidence_consistency(records: Sequence[AnswerPerQueryRecord]) -> dict[str, object]:
    total = len(records)
    labels = Counter(record.evidence_consistency_label for record in records)
    grounded_count = sum(1 for record in records if record.groundedness_flag)
    citation_presence_count = sum(1 for record in records if record.citation_count > 0)
    retrieval_hit_count = sum(1 for record in records if record.retrieval_hit_at_10)
    citation_gold_hit_count = sum(1 for record in records if record.citation_hit_at_10)
    consistent_count = (
        labels["grounded_answer_with_citations"]
        + labels["grounded_answer_without_citations"]
        + labels["appropriate_abstain_due_to_retrieval_gap"]
    )
    return {
        "total_queries": total,
        "retrieval_hit_at_10_count": retrieval_hit_count,
        "retrieval_hit_at_10_ratio": _ratio(retrieval_hit_count, total),
        "citation_presence_count": citation_presence_count,
        "citation_presence_rate": _ratio(citation_presence_count, total),
        "citation_gold_hit_count": citation_gold_hit_count,
        "citation_gold_hit_rate": _ratio(citation_gold_hit_count, total),
        "grounded_answer_count": grounded_count,
        "grounded_answer_rate": _ratio(grounded_count, total),
        "evidence_consistent_count": consistent_count,
        "evidence_consistent_ratio": _ratio(consistent_count, total),
        "grounded_answer_with_citations_count": labels["grounded_answer_with_citations"],
        "grounded_answer_without_citations_count": labels["grounded_answer_without_citations"],
        "appropriate_abstain_due_to_retrieval_gap_count": labels["appropriate_abstain_due_to_retrieval_gap"],
        "missed_answer_despite_retrieval_hit_count": labels["missed_answer_despite_retrieval_hit"],
        "unsupported_answer_count": labels["unsupported_answer"],
    }


def select_answer_judge_subset(
    records: Sequence[AnswerPerQueryRecord],
    *,
    subset_size: int,
    seed: int = 42,
) -> list[AnswerPerQueryRecord]:
    if subset_size <= 0:
        return []
    priority_labels = {
        "unsupported_answer",
        "missed_answer_despite_retrieval_hit",
        "appropriate_abstain_due_to_retrieval_gap",
    }
    prioritized = [record for record in records if record.evidence_consistency_label in priority_labels]
    prioritized.sort(key=lambda item: item.query_id)
    if len(prioritized) >= subset_size:
        return prioritized[:subset_size]
    remaining = [record for record in records if record not in prioritized]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    needed = subset_size - len(prioritized)
    return [*prioritized, *remaining[:needed]]


def should_review_local_judge(
    record: AnswerPerQueryRecord,
    decision: AnswerJudgeDecision,
    *,
    confidence_threshold: float,
) -> bool:
    if decision.verdict == "unclear":
        return True
    if decision.confidence < confidence_threshold:
        return True
    if record.evidence_consistency_label == "unsupported_answer" and decision.verdict == "correct":
        return True
    if (
        record.evidence_consistency_label == "grounded_answer_with_citations"
        and decision.verdict == "incorrect"
    ):
        return True
    return False


def summarize_answer_judging(records: Sequence[AnswerJudgeRecord]) -> dict[str, object]:
    total = len(records)
    local_counts = Counter(record.local_decision.verdict for record in records)
    final_counts = Counter(record.final_verdict for record in records)
    review_required_count = sum(1 for record in records if record.review_required)
    review_executed_count = sum(1 for record in records if record.review_decision is not None)
    review_disagreement_count = sum(
        1
        for record in records
        if record.review_decision is not None and record.review_decision.verdict != record.local_decision.verdict
    )
    return {
        "subset_query_count": total,
        "local_verdict_counts": dict(local_counts),
        "final_verdict_counts": dict(final_counts),
        "review_required_count": review_required_count,
        "review_executed_count": review_executed_count,
        "local_review_disagreement_count": review_disagreement_count,
        "final_correct_ratio": _ratio(final_counts["correct"], total),
        "final_incorrect_ratio": _ratio(final_counts["incorrect"], total),
        "local_judge_provider": records[0].local_decision.provider if records else None,
        "local_judge_model": records[0].local_decision.model if records else None,
        "review_judge_provider": next(
            (record.review_decision.provider for record in records if record.review_decision is not None),
            None,
        ),
        "review_judge_model": next(
            (record.review_decision.model for record in records if record.review_decision is not None),
            None,
        ),
    }


def generate_answer_recommendations(
    *,
    evidence_summary: Mapping[str, object],
    judge_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    retrieval_hit = float(evidence_summary.get("retrieval_hit_at_10_ratio", 0.0) or 0.0)
    missed_answer = int(evidence_summary.get("missed_answer_despite_retrieval_hit_count", 0) or 0)
    unsupported = int(evidence_summary.get("unsupported_answer_count", 0) or 0)
    citation_gold_hit_rate = float(evidence_summary.get("citation_gold_hit_rate", 0.0) or 0.0)
    answer_correct_rate = judge_summary.get("final_correct_ratio")
    if retrieval_hit < 0.8:
        recommendations.append(
            {
                "category": "retrieval_dependency",
                "recommendation": "答案生成实验前，先继续看召回缺口；检索命中率仍不足时，生成层收益会被上限压住。",
                "evidence": {"retrieval_hit_at_10_ratio": round(retrieval_hit, 6)},
            }
        )
    if missed_answer > 0:
        recommendations.append(
            {
                "category": "answer_synthesis",
                "recommendation": (
                    "已经召回到 gold 但仍有样本没有答出来，下一步更该做 "
                    "answer synthesis / prompt 结构实验，而不是继续只调检索。"
                ),
                "evidence": {"missed_answer_despite_retrieval_hit_count": missed_answer},
            }
        )
    if unsupported > 0 or citation_gold_hit_rate < 0.5:
        recommendations.append(
            {
                "category": "citation_guard",
                "recommendation": (
                    "存在 unsupported answer 或 citation 对 gold 覆盖不足，"
                    "下一步优先加 citation / groundedness guard，而不是放宽生成自由度。"
                ),
                "evidence": {
                    "unsupported_answer_count": unsupported,
                    "citation_gold_hit_rate": round(citation_gold_hit_rate, 6),
                },
            }
        )
    if isinstance(answer_correct_rate, (int, float)) and answer_correct_rate < 0.75:
        recommendations.append(
            {
                "category": "answer_synthesis",
                "recommendation": "judge 子集正确率不高，下一步优先做生成 prompt、答案结构和引用约束实验。",
                "evidence": {"final_correct_ratio": round(float(answer_correct_rate), 6)},
            }
        )
    review_disagreement = int(judge_summary.get("local_review_disagreement_count", 0) or 0)
    review_executed = int(judge_summary.get("review_executed_count", 0) or 0)
    if review_executed > 0 and review_disagreement / max(review_executed, 1) > 0.2:
        recommendations.append(
            {
                "category": "judge_quality",
                "recommendation": (
                    "本地 judge 和强模型复核分歧偏多，后续别直接拿本地 judge "
                    "做最终裁决，应保留复核或增强 rubric。"
                ),
                "evidence": {
                    "review_executed_count": review_executed,
                    "local_review_disagreement_count": review_disagreement,
                },
            }
        )
    return recommendations


def _judge_failure_summary(
    *,
    subset_query_count: int,
    completed_judge_count: int,
    review_enabled: bool,
    local_judge: _JudgeBinding | None,
    exc: Exception,
    stage: Literal["local", "review"],
) -> dict[str, object]:
    stage_name = "本地初筛 judge" if stage == "local" else "强模型复核 judge"
    return {
        "subset_query_count": subset_query_count,
        "completed_judge_count": completed_judge_count,
        "review_enabled": review_enabled,
        "review_executed_count": 0,
        "local_judge_provider": None if local_judge is None else local_judge.provider,
        "local_judge_model": None if local_judge is None else local_judge.model,
        "judge_stage_error": str(exc),
        "note": (
            f"{stage_name}调用失败，answer benchmark 已保留证据一致性结果，"
            "答案正确性 judge 阶段本轮跳过。"
        ),
    }


def answer_benchmark_run_id(dataset: str) -> str:
    return f"{benchmark_run_id(dataset)}-answers"


def append_answer_baseline_row(path: Path, summary: AnswerRunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "dataset",
        "variant",
        "query_count",
        "judge_subset_count",
        "embedding_model",
        "generation_provider",
        "generation_model",
        "retrieval_mode",
        "rerank_enabled",
        "answer_context_top_k",
        "evidence_consistent_rate",
        "grounded_answer_rate",
        "citation_presence_rate",
        "citation_gold_hit_rate",
        "answer_correct_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        "queries_per_second",
    ]
    row = summary.as_json()
    write_header = not path.exists() or path.stat().st_size == 0
    existing_rows: list[dict[str, str]] = []
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_header = reader.fieldnames or []
            if existing_header != fieldnames:
                existing_rows = [dict(item) for item in reader]
                write_header = True
    if write_header and existing_rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({name: existing.get(name, "") for name in fieldnames})
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header and not existing_rows:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_chat_judge(
    *,
    profile_id: str,
    require_cloud: bool = False,
    allow_missing: bool = True,
) -> _JudgeBinding | None:
    service = CapabilityAssemblyService()
    try:
        request = service.request_for_profile(
            profile_id,
            requirements=CapabilityRequirements(
                require_embedding=False,
                require_chat=True,
                require_rerank=False,
                allow_degraded=allow_missing,
            ),
        )
        bundle = service.assemble_request(request)
    except Exception:
        if allow_missing:
            return None
        raise
    bindings = list(bundle.chat_bindings)
    if require_cloud:
        bindings = [binding for binding in bindings if binding.location == "cloud"]
    if not bindings:
        if allow_missing:
            return None
        raise RuntimeError(f"No chat-capable provider available for judge profile {profile_id!r}")
    binding = bindings[0]
    return _JudgeBinding(
        provider=binding.provider_name,
        model=binding.model_name,
        chat=binding.chat,
    )


def run_answer_judge(
    binding: _JudgeBinding,
    *,
    query_text: str,
    answer_text: str,
    gold_reference_texts: Sequence[str],
) -> AnswerJudgeDecision:
    prompt = build_answer_judge_prompt(
        query_text=query_text,
        answer_text=answer_text,
        gold_reference_texts=gold_reference_texts,
    )
    raw = binding.chat(prompt)
    payload = _parse_judge_payload(raw)
    verdict = str(payload.get("verdict", "unclear")).strip().lower()
    if verdict not in {"correct", "partially_correct", "incorrect", "insufficient", "unclear"}:
        verdict = "unclear"
    confidence = payload.get("confidence", 0.0)
    try:
        confidence_value = max(0.0, min(float(confidence), 1.0))
    except (TypeError, ValueError):
        confidence_value = 0.0
    rationale = str(payload.get("rationale", "")).strip() or "Judge returned no rationale."
    return AnswerJudgeDecision(
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence_value,
        rationale=rationale,
        provider=binding.provider,
        model=binding.model,
    )


def build_answer_judge_prompt(
    *,
    query_text: str,
    answer_text: str,
    gold_reference_texts: Sequence[str],
) -> str:
    references = "\n\n".join(f"[REF {index}]\n{text}" for index, text in enumerate(gold_reference_texts, start=1))
    return (
        "你是公开 benchmark 的答案正确性评审器。"
        "只使用给定参考文档作为事实依据，不允许引入外部知识。"
        "返回一个 JSON 对象，字段固定为 verdict, confidence, rationale。"
        'verdict 只能是 "correct" / "partially_correct" / "incorrect" / "insufficient" / "unclear"。'
        "confidence 取 0 到 1。rationale 用一句中文解释。\n\n"
        f"问题：{query_text}\n\n"
        f"候选答案：{answer_text}\n\n"
        f"参考文档：\n{references}\n"
    )


def load_prepared_documents(path: Path) -> dict[str, _PreparedDocument]:
    documents: dict[str, _PreparedDocument] = {}
    for record in iter_jsonl(path):
        doc_id = str(record.get("doc_id", "")).strip()
        if not doc_id:
            continue
        title = str(record.get("title") or doc_id).strip() or doc_id
        text = str(record.get("text") or "").strip()
        documents[doc_id] = _PreparedDocument(doc_id=doc_id, title=title, text=text)
    return documents


def _gold_reference_texts(doc_ids: Sequence[str], documents: Mapping[str, _PreparedDocument]) -> list[str]:
    texts: list[str] = []
    for doc_id in doc_ids:
        document = documents.get(doc_id)
        if document is None:
            continue
        text = document.text.strip()
        if document.title and document.title != doc_id:
            text = f"{document.title}\n{text}".strip()
        texts.append(text[:1600])
    return texts[:3]


def _parse_judge_payload(raw: str) -> dict[str, object]:
    cleaned = raw.strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return {"verdict": "unclear", "confidence": 0.0, "rationale": cleaned[:500]}


def _ratio(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _first_non_blank(values: Iterable[str | None]) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_doc_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


__all__ = [
    "AnswerBenchmarkEvaluator",
    "AnswerJudgeDecision",
    "AnswerJudgeRecord",
    "AnswerPerQueryRecord",
    "AnswerRunSummary",
    "append_answer_baseline_row",
    "answer_benchmark_run_id",
    "build_answer_record",
    "build_chat_judge",
    "classify_evidence_consistency",
    "generate_answer_recommendations",
    "load_prepared_documents",
    "run_answer_judge",
    "select_answer_judge_subset",
    "should_review_local_judge",
    "summarize_answer_judging",
    "summarize_evidence_consistency",
]
