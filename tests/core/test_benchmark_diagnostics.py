from __future__ import annotations

from rag.benchmark_diagnostics import (
    BranchDiagnosticsRecord,
    FailureAnalysisRecord,
    analyze_full_text_profile,
    analyze_recall_failure_profile,
    analyze_rerank_profile,
    classify_failure_case,
    generate_diagnostic_recommendations,
    summarize_branch_records,
    summarize_failure_records,
)


def test_classify_failure_case_distinguishes_top_hits_fusion_and_mapping() -> None:
    top1 = classify_failure_case(
        predicted_doc_ids=["d1", "d2"],
        gold_doc_ids=["d1"],
        reranked_chunk_ids=["c1", "c2"],
        final_unmapped_chunk_ids=[],
        active_branch_hits={"vector": False},
        inactive_branch_hits={},
        query_text="alpha beta gamma",
        top_k=10,
    )
    assert top1["failure_bucket"] == "top1_hit"
    assert top1["failure_subtype"] is None
    assert top1["first_relevant_rank"] == 1

    low_rank = classify_failure_case(
        predicted_doc_ids=["d9", "d4", "d2"],
        gold_doc_ids=["d2"],
        reranked_chunk_ids=["c9", "c4", "c2"],
        final_unmapped_chunk_ids=[],
        active_branch_hits={"vector": True},
        inactive_branch_hits={},
        query_text="alpha beta gamma delta",
        top_k=10,
    )
    assert low_rank["failure_bucket"] == "top10_hit_but_low_rank"
    assert low_rank["failure_subtype"] is None
    assert low_rank["first_relevant_rank"] == 3

    fusion_loss = classify_failure_case(
        predicted_doc_ids=["d9", "d4"],
        gold_doc_ids=["gold-1"],
        reranked_chunk_ids=["c9", "c4"],
        final_unmapped_chunk_ids=[],
        active_branch_hits={"vector": False, "full_text": True},
        inactive_branch_hits={"local": True},
        query_text="ABC risk ratio",
        top_k=10,
    )
    assert fusion_loss["failure_bucket"] == "top10_miss"
    assert fusion_loss["failure_subtype"] == "fusion_loss"
    assert "possible_query_expression_issue" in fusion_loss["heuristic_labels"]
    assert "possible_branch_coverage_issue" in fusion_loss["heuristic_labels"]

    mapping_failure = classify_failure_case(
        predicted_doc_ids=[],
        gold_doc_ids=["gold-1"],
        reranked_chunk_ids=["c1", "c2"],
        final_unmapped_chunk_ids=["c1", "c2"],
        active_branch_hits={"vector": False},
        inactive_branch_hits={},
        query_text="long enough descriptive query text",
        top_k=10,
    )
    assert mapping_failure["failure_bucket"] == "empty_or_invalid_prediction"
    assert mapping_failure["failure_subtype"] == "mapping_failure"


def test_summaries_capture_failure_mix_branch_value_and_recommendations() -> None:
    failure_records = [
        FailureAnalysisRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q1",
            query_text="q1",
            gold_doc_ids=["d1"],
            predicted_doc_ids=["d1"],
            hit_at_1=True,
            hit_at_10=True,
            first_relevant_rank=1,
            failure_bucket="top1_hit",
            failure_subtype=None,
            latency_ms=10.0,
            retrieval_mode="naive",
            rerank_enabled=True,
        ),
        FailureAnalysisRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q2",
            query_text="ABC",
            gold_doc_ids=["d2"],
            predicted_doc_ids=["d9", "d2"],
            hit_at_1=False,
            hit_at_10=True,
            first_relevant_rank=2,
            failure_bucket="top10_hit_but_low_rank",
            failure_subtype=None,
            latency_ms=11.0,
            retrieval_mode="naive",
            rerank_enabled=True,
            heuristic_labels=["possible_query_expression_issue"],
        ),
        FailureAnalysisRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q3",
            query_text="q3",
            gold_doc_ids=["d3"],
            predicted_doc_ids=[],
            hit_at_1=False,
            hit_at_10=False,
            first_relevant_rank=None,
            failure_bucket="empty_or_invalid_prediction",
            failure_subtype="mapping_failure",
            latency_ms=12.0,
            retrieval_mode="naive",
            rerank_enabled=True,
        ),
        FailureAnalysisRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q4",
            query_text="q4",
            gold_doc_ids=["d4"],
            predicted_doc_ids=["d9"],
            hit_at_1=False,
            hit_at_10=False,
            first_relevant_rank=None,
            failure_bucket="top10_miss",
            failure_subtype="fusion_loss",
            latency_ms=13.0,
            retrieval_mode="naive",
            rerank_enabled=True,
            heuristic_labels=["possible_branch_coverage_issue"],
        ),
        FailureAnalysisRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q5",
            query_text="q5",
            gold_doc_ids=["d5"],
            predicted_doc_ids=["d8"],
            hit_at_1=False,
            hit_at_10=False,
            first_relevant_rank=None,
            failure_bucket="top10_miss",
            failure_subtype="recall_failure",
            latency_ms=14.0,
            retrieval_mode="naive",
            rerank_enabled=True,
        ),
    ]
    failure_summary = summarize_failure_records(failure_records)
    assert failure_summary["total_queries"] == 5
    assert failure_summary["top1_hit_count"] == 1
    assert failure_summary["top10_hit_but_low_rank_count"] == 1
    assert failure_summary["mapping_failure_count"] == 1
    assert failure_summary["fusion_loss_count"] == 1
    assert failure_summary["recall_failure_count"] == 1
    assert failure_summary["possible_query_expression_issue_count"] == 1
    assert failure_summary["possible_branch_coverage_issue_count"] == 1

    branch_records = [
        BranchDiagnosticsRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q1",
            query_text="q1",
            gold_doc_ids=["d1"],
            predicted_doc_ids=["d1"],
            retrieval_mode="naive",
            rerank_enabled=True,
            branch_candidate_doc_ids={
                "vector": ["d1", "d9"],
                "full_text": ["d1", "d8"],
                "local": ["d1"],
                "global": ["d7"],
            },
            branch_candidate_benchmark_doc_ids={
                "vector": ["d1", "d9"],
                "full_text": ["d1", "d8"],
                "local": ["d1"],
                "global": ["d7"],
            },
            branch_candidate_chunk_ids={"vector": ["c1"], "full_text": ["c2"]},
            branch_hit_at_10={"vector": True, "full_text": True, "local": True, "global": False},
            branch_overlap_with_vector={"full_text": 1 / 3, "local": 0.5, "global": 0.0},
            branch_added_doc_count_vs_vector={"full_text": 1, "local": 0, "global": 1},
            active_branches=["vector"],
            branches_hitting_gold=["vector", "full_text", "local"],
            branches_hitting_gold_only=["vector"],
            fused_doc_ids=["d1", "d9"],
            reranked_doc_ids=["d1"],
            gold_in_fused_top_k=True,
            gold_in_reranked_top_k=True,
            fusion_lost_gold=False,
            rerank_helped=False,
            rerank_hurt=False,
        ),
        BranchDiagnosticsRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q2",
            query_text="q2",
            gold_doc_ids=["d2"],
            predicted_doc_ids=["d9", "d2"],
            retrieval_mode="naive",
            rerank_enabled=True,
            branch_candidate_doc_ids={
                "vector": ["d9", "d2"],
                "full_text": ["d8", "d2"],
                "local": ["d2"],
            },
            branch_candidate_benchmark_doc_ids={
                "vector": ["d9", "d2"],
                "full_text": ["d8", "d2"],
                "local": ["d2"],
            },
            branch_candidate_chunk_ids={"vector": ["c3"], "full_text": ["c4"], "local": ["c5"]},
            branch_hit_at_10={"vector": True, "full_text": True, "local": True},
            branch_overlap_with_vector={"full_text": 1 / 3, "local": 0.5},
            branch_added_doc_count_vs_vector={"full_text": 1, "local": 0},
            active_branches=["vector"],
            branches_hitting_gold=["vector", "full_text", "local"],
            branches_hitting_gold_only=["local"],
            fused_doc_ids=["d9", "d2"],
            reranked_doc_ids=["d2", "d9"],
            gold_in_fused_top_k=True,
            gold_in_reranked_top_k=True,
            fusion_lost_gold=False,
            rerank_helped=True,
            rerank_hurt=False,
        ),
        BranchDiagnosticsRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q3",
            query_text="q3",
            gold_doc_ids=["d3"],
            predicted_doc_ids=[],
            retrieval_mode="naive",
            rerank_enabled=True,
            branch_candidate_doc_ids={"vector": ["d8"], "full_text": [], "metadata": ["d8"]},
            branch_candidate_benchmark_doc_ids={"vector": ["d8"], "full_text": [], "metadata": ["d8"]},
            branch_candidate_chunk_ids={"vector": ["c6"], "metadata": ["c10"]},
            branch_hit_at_10={"vector": False, "full_text": False, "metadata": False},
            branch_overlap_with_vector={"full_text": 0.0, "metadata": 1.0},
            branch_added_doc_count_vs_vector={"full_text": 0, "metadata": 0},
            active_branches=["vector"],
            branches_hitting_gold=[],
            branches_hitting_gold_only=[],
            fused_doc_ids=["d8"],
            reranked_doc_ids=[],
            gold_in_fused_top_k=False,
            gold_in_reranked_top_k=False,
            fusion_lost_gold=False,
            rerank_helped=False,
            rerank_hurt=True,
            unmapped_chunk_ids=["c6"],
            mapping_debug_info=[{"chunk_id": "c6", "benchmark_doc_id": None}],
        ),
        BranchDiagnosticsRecord(
            run_id="run-1",
            dataset="medical_retrieval",
            query_id="q4",
            query_text="q4",
            gold_doc_ids=["d4"],
            predicted_doc_ids=["d9"],
            retrieval_mode="naive",
            rerank_enabled=True,
            branch_candidate_doc_ids={"vector": ["d9"], "full_text": ["d4"], "global": ["d4"]},
            branch_candidate_benchmark_doc_ids={"vector": ["d9"], "full_text": ["d4"], "global": ["d4"]},
            branch_candidate_chunk_ids={"vector": ["c7"], "full_text": ["c8"], "global": ["c9"]},
            branch_hit_at_10={"vector": False, "full_text": True, "global": True},
            branch_overlap_with_vector={"full_text": 0.0, "global": 0.0},
            branch_added_doc_count_vs_vector={"full_text": 1, "global": 1},
            active_branches=["vector"],
            branches_hitting_gold=["full_text", "global"],
            branches_hitting_gold_only=["full_text", "global"],
            fused_doc_ids=["d9"],
            reranked_doc_ids=["d9"],
            gold_in_fused_top_k=False,
            gold_in_reranked_top_k=False,
            fusion_lost_gold=True,
            rerank_helped=False,
            rerank_hurt=False,
        ),
    ]
    branch_summary = summarize_branch_records(branch_records)
    assert branch_summary["total_queries"] == 4
    assert branch_summary["branches"]["vector"]["hit_at_10_count"] == 2
    assert branch_summary["branches"]["local"]["independent_hit_count"] == 1
    assert branch_summary["branches"]["full_text"]["independent_hit_count"] == 1
    assert branch_summary["branches"]["global"]["independent_hit_count"] == 1
    assert branch_summary["fusion_loss_query_count"] == 1
    assert branch_summary["rerank_helped_query_count"] == 1
    assert branch_summary["rerank_hurt_query_count"] == 1

    recommendations = generate_diagnostic_recommendations(
        failure_summary=failure_summary,
        branch_summary=branch_summary,
    )
    categories = {item["category"] for item in recommendations}
    assert "mapping" in categories
    assert "rerank" in categories
    assert "fusion" in categories
    assert "branch_pruning" in categories


def test_profiles_capture_recall_rerank_and_full_text_patterns() -> None:
    failure_records = [
        FailureAnalysisRecord(
            run_id="run-2",
            dataset="medical_retrieval",
            query_id="q-short-en",
            query_text="ABC 19 指南",
            gold_doc_ids=["d1"],
            predicted_doc_ids=["d9"],
            hit_at_1=False,
            hit_at_10=False,
            first_relevant_rank=None,
            failure_bucket="top10_miss",
            failure_subtype="recall_failure",
            latency_ms=9.0,
            retrieval_mode="naive",
            rerank_enabled=True,
            heuristic_labels=["possible_query_expression_issue"],
        ),
        FailureAnalysisRecord(
            run_id="run-2",
            dataset="medical_retrieval",
            query_id="q-short-zh",
            query_text="肾病能喝酒吗",
            gold_doc_ids=["d2"],
            predicted_doc_ids=["d8"],
            hit_at_1=False,
            hit_at_10=False,
            first_relevant_rank=None,
            failure_bucket="top10_miss",
            failure_subtype="recall_failure",
            latency_ms=10.0,
            retrieval_mode="naive",
            rerank_enabled=True,
        ),
        FailureAnalysisRecord(
            run_id="run-2",
            dataset="medical_retrieval",
            query_id="q-rank",
            query_text="多囊卵巢综合征 影响 母乳 吗",
            gold_doc_ids=["d3"],
            predicted_doc_ids=["d8", "d3"],
            hit_at_1=False,
            hit_at_10=True,
            first_relevant_rank=2,
            failure_bucket="top10_hit_but_low_rank",
            failure_subtype=None,
            latency_ms=11.0,
            retrieval_mode="naive",
            rerank_enabled=True,
        ),
    ]
    branch_records = [
        BranchDiagnosticsRecord(
            run_id="run-2",
            dataset="medical_retrieval",
            query_id="q-short-en",
            query_text="ABC 19 指南",
            gold_doc_ids=["d1"],
            predicted_doc_ids=["d9"],
            retrieval_mode="naive",
            rerank_enabled=True,
            branch_candidate_doc_ids={"vector": ["d9"], "full_text": ["d1", "d7"]},
            branch_candidate_benchmark_doc_ids={"vector": ["d9"], "full_text": ["d1", "d7"]},
            branch_candidate_chunk_ids={"vector": ["c1"], "full_text": ["c2"]},
            branch_hit_at_10={"vector": False, "full_text": True},
            branch_overlap_with_vector={"full_text": 0.0},
            branch_added_doc_count_vs_vector={"full_text": 2},
            active_branches=["vector"],
            branches_hitting_gold=["full_text"],
            branches_hitting_gold_only=["full_text"],
            fused_doc_ids=["d9"],
            reranked_doc_ids=["d9"],
            gold_in_fused_top_k=False,
            gold_in_reranked_top_k=False,
            fusion_lost_gold=False,
            rerank_helped=False,
            rerank_hurt=False,
        ),
        BranchDiagnosticsRecord(
            run_id="run-2",
            dataset="medical_retrieval",
            query_id="q-short-zh",
            query_text="肾病能喝酒吗",
            gold_doc_ids=["d2"],
            predicted_doc_ids=["d8"],
            retrieval_mode="naive",
            rerank_enabled=True,
            branch_candidate_doc_ids={"vector": ["d8"], "full_text": ["d6"]},
            branch_candidate_benchmark_doc_ids={"vector": ["d8"], "full_text": ["d6"]},
            branch_candidate_chunk_ids={"vector": ["c3"], "full_text": ["c4"]},
            branch_hit_at_10={"vector": False, "full_text": False},
            branch_overlap_with_vector={"full_text": 0.0},
            branch_added_doc_count_vs_vector={"full_text": 1},
            active_branches=["vector"],
            branches_hitting_gold=[],
            branches_hitting_gold_only=[],
            fused_doc_ids=["d8"],
            reranked_doc_ids=["d8"],
            gold_in_fused_top_k=False,
            gold_in_reranked_top_k=False,
            fusion_lost_gold=False,
            rerank_helped=False,
            rerank_hurt=False,
        ),
        BranchDiagnosticsRecord(
            run_id="run-2",
            dataset="medical_retrieval",
            query_id="q-helped",
            query_text="高血压 饮食 禁忌",
            gold_doc_ids=["d4"],
            predicted_doc_ids=["d4", "d5"],
            retrieval_mode="naive",
            rerank_enabled=True,
            branch_candidate_doc_ids={"vector": ["d5", "d4"], "full_text": ["d4"]},
            branch_candidate_benchmark_doc_ids={"vector": ["d5", "d4"], "full_text": ["d4"]},
            branch_candidate_chunk_ids={"vector": ["c5"], "full_text": ["c6"]},
            branch_hit_at_10={"vector": True, "full_text": True},
            branch_overlap_with_vector={"full_text": 0.5},
            branch_added_doc_count_vs_vector={"full_text": 0},
            active_branches=["vector"],
            branches_hitting_gold=["vector", "full_text"],
            branches_hitting_gold_only=[],
            fused_doc_ids=["d5", "d4"],
            reranked_doc_ids=["d4", "d5"],
            gold_in_fused_top_k=True,
            gold_in_reranked_top_k=True,
            fusion_lost_gold=False,
            rerank_helped=True,
            rerank_hurt=False,
        ),
        BranchDiagnosticsRecord(
            run_id="run-2",
            dataset="medical_retrieval",
            query_id="q-hurt",
            query_text="婴儿 发烧 处理",
            gold_doc_ids=["d5"],
            predicted_doc_ids=["d9", "d5"],
            retrieval_mode="naive",
            rerank_enabled=True,
            branch_candidate_doc_ids={"vector": ["d5", "d9"], "full_text": ["d5"]},
            branch_candidate_benchmark_doc_ids={"vector": ["d5", "d9"], "full_text": ["d5"]},
            branch_candidate_chunk_ids={"vector": ["c7"], "full_text": ["c8"]},
            branch_hit_at_10={"vector": True, "full_text": True},
            branch_overlap_with_vector={"full_text": 0.5},
            branch_added_doc_count_vs_vector={"full_text": 0},
            active_branches=["vector"],
            branches_hitting_gold=["vector", "full_text"],
            branches_hitting_gold_only=[],
            fused_doc_ids=["d5", "d9"],
            reranked_doc_ids=["d9", "d5"],
            gold_in_fused_top_k=True,
            gold_in_reranked_top_k=True,
            fusion_lost_gold=False,
            rerank_helped=False,
            rerank_hurt=True,
        ),
    ]
    documents_by_id = {
        "d1": {"doc_id": "d1", "text": "ABC19 指南 与 用药 说明", "metadata": {}},
        "d2": {"doc_id": "d2", "text": "肾病患者 饮酒 风险 与 禁忌", "metadata": {}},
        "d3": {"doc_id": "d3", "text": "多囊卵巢综合征 影响 母乳 哺乳", "metadata": {}},
        "d4": {"doc_id": "d4", "text": "高血压 饮食 禁忌 与 食疗 建议", "metadata": {}},
        "d5": {"doc_id": "d5", "text": "婴儿 发烧 处理 退烧 观察", "metadata": {}},
    }

    recall_profile = analyze_recall_failure_profile(
        failure_records=failure_records,
        branch_records=branch_records,
        documents_by_id=documents_by_id,
    )
    assert recall_profile["recall_failure_count"] == 2
    assert recall_profile["feature_summary"]["short_query_ratio"] == 1.0
    assert recall_profile["feature_summary"]["contains_ascii_ratio"] == 0.5
    assert recall_profile["feature_summary"]["contains_digit_ratio"] == 0.5
    assert recall_profile["feature_summary"]["avg_query_term_overlap_with_gold"] > 0.2
    assert recall_profile["top_terms"]

    rerank_profile = analyze_rerank_profile(branch_records=branch_records, top_k=10)
    assert rerank_profile["helped"]["count"] == 1
    assert rerank_profile["hurt"]["count"] == 1
    assert rerank_profile["helped"]["avg_fused_first_relevant_rank"] == 2.0
    assert rerank_profile["hurt"]["avg_fused_first_relevant_rank"] == 1.0
    assert rerank_profile["hurt"]["vector_already_top3_ratio"] == 1.0

    full_text_profile = analyze_full_text_profile(
        branch_records=branch_records,
        documents_by_id=documents_by_id,
        top_k=10,
    )
    assert full_text_profile["full_text_independent_hit_count"] == 1
    assert full_text_profile["vector_miss_full_text_hit_count"] == 1
    assert full_text_profile["feature_summary"]["contains_ascii_ratio"] == 1.0
    assert full_text_profile["sample_queries"][0]["query_id"] == "q-short-en"
