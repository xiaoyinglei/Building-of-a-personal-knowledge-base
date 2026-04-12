from __future__ import annotations

from types import SimpleNamespace

from rag.answer_benchmarks import (
    AnswerJudgeDecision,
    AnswerJudgeRecord,
    AnswerPerQueryRecord,
    _judge_failure_summary,
    build_answer_record,
    classify_evidence_consistency,
    generate_answer_recommendations,
    select_answer_judge_subset,
    should_review_local_judge,
    summarize_answer_judging,
    summarize_evidence_consistency,
)
from rag.schema.query import AnswerCitation, GroundedAnswer


def _record(
    query_id: str,
    *,
    label: str,
    retrieval_hit: bool,
    citation_hit: bool,
    grounded: bool,
    insufficient: bool = False,
    query_text: str = "常见问题",
) -> AnswerPerQueryRecord:
    return AnswerPerQueryRecord(
        run_id="run-1",
        dataset="medical_retrieval",
        query_id=query_id,
        query_text=query_text,
        gold_doc_ids=[f"doc-{query_id}"],
        retrieved_doc_ids=[f"doc-{query_id}"] if retrieval_hit else ["other-doc"],
        cited_doc_ids=[f"doc-{query_id}"] if citation_hit else [],
        answer_text="答案",
        citation_count=1 if citation_hit else 0,
        groundedness_flag=grounded,
        insufficient_evidence_flag=insufficient,
        evidence_consistency_label=label,
        retrieval_hit_at_10=retrieval_hit,
        citation_hit_at_10=citation_hit,
        latency_ms=100.0,
        generation_provider="ollama",
        generation_model="qwen3:14b",
    )


def test_classify_evidence_consistency_distinguishes_supported_abstain_and_unsupported() -> None:
    assert (
        classify_evidence_consistency(
            retrieval_hit_at_10=True,
            citation_count=2,
            citation_hit_at_10=True,
            groundedness_flag=True,
            insufficient_evidence_flag=False,
        )
        == "grounded_answer_with_citations"
    )
    assert (
        classify_evidence_consistency(
            retrieval_hit_at_10=False,
            citation_count=0,
            citation_hit_at_10=False,
            groundedness_flag=False,
            insufficient_evidence_flag=True,
        )
        == "appropriate_abstain_due_to_retrieval_gap"
    )
    assert (
        classify_evidence_consistency(
            retrieval_hit_at_10=True,
            citation_count=0,
            citation_hit_at_10=False,
            groundedness_flag=False,
            insufficient_evidence_flag=True,
        )
        == "missed_answer_despite_retrieval_hit"
    )
    assert (
        classify_evidence_consistency(
            retrieval_hit_at_10=True,
            citation_count=0,
            citation_hit_at_10=False,
            groundedness_flag=False,
            insufficient_evidence_flag=False,
        )
        == "unsupported_answer"
    )


def test_summarize_evidence_consistency_reports_generation_vs_retrieval_signals() -> None:
    records = [
        _record(
            "q1",
            label="grounded_answer_with_citations",
            retrieval_hit=True,
            citation_hit=True,
            grounded=True,
        ),
        _record(
            "q2",
            label="appropriate_abstain_due_to_retrieval_gap",
            retrieval_hit=False,
            citation_hit=False,
            grounded=False,
            insufficient=True,
        ),
        _record(
            "q3",
            label="missed_answer_despite_retrieval_hit",
            retrieval_hit=True,
            citation_hit=False,
            grounded=False,
            insufficient=True,
        ),
        _record(
            "q4",
            label="unsupported_answer",
            retrieval_hit=True,
            citation_hit=False,
            grounded=False,
        ),
    ]

    summary = summarize_evidence_consistency(records)

    assert summary["total_queries"] == 4
    assert summary["retrieval_hit_at_10_count"] == 3
    assert summary["grounded_answer_with_citations_count"] == 1
    assert summary["appropriate_abstain_due_to_retrieval_gap_count"] == 1
    assert summary["missed_answer_despite_retrieval_hit_count"] == 1
    assert summary["unsupported_answer_count"] == 1
    assert summary["evidence_consistent_ratio"] == 0.5


def test_select_answer_judge_subset_prioritizes_problem_cases_before_filling() -> None:
    records = [
        _record(
            "bad-1",
            label="unsupported_answer",
            retrieval_hit=True,
            citation_hit=False,
            grounded=False,
            query_text="bad-1",
        ),
        _record(
            "bad-2",
            label="missed_answer_despite_retrieval_hit",
            retrieval_hit=True,
            citation_hit=False,
            grounded=False,
            insufficient=True,
            query_text="bad-2",
        ),
        _record(
            "gap-1",
            label="appropriate_abstain_due_to_retrieval_gap",
            retrieval_hit=False,
            citation_hit=False,
            grounded=False,
            insufficient=True,
            query_text="gap-1",
        ),
        _record(
            "good-1",
            label="grounded_answer_with_citations",
            retrieval_hit=True,
            citation_hit=True,
            grounded=True,
            query_text="good-1",
        ),
        _record(
            "good-2",
            label="grounded_answer_with_citations",
            retrieval_hit=True,
            citation_hit=True,
            grounded=True,
            query_text="good-2",
        ),
    ]

    subset = select_answer_judge_subset(records, subset_size=3, seed=7)

    subset_ids = [record.query_id for record in subset]
    assert subset_ids[:3] == ["bad-1", "bad-2", "gap-1"]


def test_should_review_local_judge_on_low_confidence_and_signal_conflict() -> None:
    record = _record(
        "q1",
        label="unsupported_answer",
        retrieval_hit=True,
        citation_hit=False,
        grounded=False,
    )
    low_confidence = AnswerJudgeDecision(
        verdict="correct",
        confidence=0.62,
        rationale="uncertain",
        provider="ollama",
        model="qwen3:14b",
    )
    assert should_review_local_judge(record, low_confidence, confidence_threshold=0.75) is True

    conflicting = AnswerJudgeDecision(
        verdict="correct",
        confidence=0.91,
        rationale="looks fine",
        provider="ollama",
        model="qwen3:14b",
    )
    assert should_review_local_judge(record, conflicting, confidence_threshold=0.75) is True

    clean = AnswerJudgeDecision(
        verdict="correct",
        confidence=0.91,
        rationale="supported",
        provider="ollama",
        model="qwen3:14b",
    )
    consistent_record = _record(
        "q2",
        label="grounded_answer_with_citations",
        retrieval_hit=True,
        citation_hit=True,
        grounded=True,
    )
    assert should_review_local_judge(consistent_record, clean, confidence_threshold=0.75) is False


def test_judge_summary_and_recommendations_point_to_generation_work() -> None:
    judged = [
        AnswerJudgeRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q1",
            query_text="q1",
            answer_text="a1",
            gold_doc_ids=["d1"],
            gold_reference_texts=["ref1"],
            local_decision=AnswerJudgeDecision(
                verdict="correct",
                confidence=0.91,
                rationale="ok",
                provider="ollama",
                model="qwen3:14b",
            ),
            review_required=False,
            final_verdict="correct",
            evidence_consistency_label="grounded_answer_with_citations",
            retrieval_hit_at_10=True,
        ),
        AnswerJudgeRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q2",
            query_text="q2",
            answer_text="a2",
            gold_doc_ids=["d2"],
            gold_reference_texts=["ref2"],
            local_decision=AnswerJudgeDecision(
                verdict="incorrect",
                confidence=0.82,
                rationale="wrong",
                provider="ollama",
                model="qwen3:14b",
            ),
            review_required=True,
            review_decision=AnswerJudgeDecision(
                verdict="incorrect",
                confidence=0.95,
                rationale="still wrong",
                provider="openai-compatible",
                model="gpt-4.1-mini",
            ),
            final_verdict="incorrect",
            evidence_consistency_label="grounded_answer_with_citations",
            retrieval_hit_at_10=True,
        ),
        AnswerJudgeRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q3",
            query_text="q3",
            answer_text="a3",
            gold_doc_ids=["d3"],
            gold_reference_texts=["ref3"],
            local_decision=AnswerJudgeDecision(
                verdict="incorrect",
                confidence=0.88,
                rationale="hallucinated",
                provider="ollama",
                model="qwen3:14b",
            ),
            review_required=False,
            final_verdict="incorrect",
            evidence_consistency_label="unsupported_answer",
            retrieval_hit_at_10=True,
        ),
    ]

    judge_summary = summarize_answer_judging(judged)
    assert judge_summary["subset_query_count"] == 3
    assert judge_summary["final_verdict_counts"]["correct"] == 1
    assert judge_summary["final_verdict_counts"]["incorrect"] == 2
    assert judge_summary["review_required_count"] == 1

    evidence_summary = summarize_evidence_consistency(
        [
            _record(
                "q1",
                label="grounded_answer_with_citations",
                retrieval_hit=True,
                citation_hit=True,
                grounded=True,
            ),
            _record(
                "q2",
                label="grounded_answer_with_citations",
                retrieval_hit=True,
                citation_hit=True,
                grounded=True,
            ),
            _record(
                "q3",
                label="unsupported_answer",
                retrieval_hit=True,
                citation_hit=False,
                grounded=False,
            ),
        ]
    )
    recommendations = generate_answer_recommendations(
        evidence_summary=evidence_summary,
        judge_summary=judge_summary,
    )

    categories = {item["category"] for item in recommendations}
    assert "answer_synthesis" in categories
    assert "citation_guard" in categories


def test_judge_failure_summary_preserves_answer_eval_run() -> None:
    summary = _judge_failure_summary(
        subset_query_count=12,
        completed_judge_count=3,
        review_enabled=True,
        local_judge=None,
        exc=RuntimeError("connection refused"),
        stage="local",
    )

    assert summary["subset_query_count"] == 12
    assert summary["completed_judge_count"] == 3
    assert summary["review_enabled"] is True
    assert "connection refused" in str(summary["judge_stage_error"])
    assert "judge" in str(summary["note"])


def test_build_answer_record_prefers_benchmark_doc_id_for_citations() -> None:
    answer = GroundedAnswer(
        answer_text="答案",
        answer_sections=[],
        citations=[
            AnswerCitation(
                citation_id="cit-1",
                chunk_id="chunk-1",
                chunk_type="child",
                doc_id="document-abc",
                source_id="src-1",
                source_type="plain_text",
                benchmark_doc_id="bench-1",
            )
        ],
        evidence_links=[],
        groundedness_flag=True,
        insufficient_evidence_flag=False,
    )
    result = SimpleNamespace(
        answer=answer,
        retrieval=SimpleNamespace(reranked_benchmark_doc_ids=["bench-1"]),
        generation_provider="ollama",
        generation_model="qwen3:14b",
    )

    record = build_answer_record(
        run_id="run-1",
        dataset="medical_retrieval",
        query_id="q1",
        query_text="问题",
        result=result,
        gold_doc_ids=["bench-1"],
        latency_ms=12.0,
        top_k=10,
    )

    assert record.cited_doc_ids == ["bench-1"]
    assert record.citation_hit_at_10 is True
