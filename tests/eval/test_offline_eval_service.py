from pathlib import Path

from pkp.eval.offline_eval_service import OfflineEvalService


def test_offline_eval_service_writes_reports_and_scores_expected_chunk_types(
    tmp_path: Path,
) -> None:
    result = OfflineEvalService(output_dir=tmp_path).run_builtin_pack()

    assert result.report.summary.total_documents == 4
    assert result.report.summary.total_questions >= 6
    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()

    table_question = next(
        item
        for item in result.report.question_results
        if item.question_id == "docx_alert_table"
    )
    assert table_question.corpus_has_expected_answer is True
    assert table_question.runtime_hit is True
    assert any(
        hit.special_chunk_type == "table"
        for hit in table_question.runtime_top_k
        if hit.is_expected_hit
    )

    ocr_question = next(
        item
        for item in result.report.question_results
        if item.question_id == "image_ocr_revenue"
    )
    assert ocr_question.runtime_hit is True
    assert any(
        hit.special_chunk_type == "ocr_region"
        for hit in ocr_question.runtime_top_k
        if hit.is_expected_hit
    )

    summary_question = next(
        item
        for item in result.report.question_results
        if item.question_id == "image_summary_scene"
    )
    assert summary_question.runtime_hit is True
    assert any(
        hit.special_chunk_type == "image_summary"
        for hit in summary_question.runtime_top_k
        if hit.is_expected_hit
    )

    parent_question = next(
        item
        for item in result.report.question_results
        if item.question_id == "markdown_release_gates"
    )
    assert parent_question.corpus_has_expected_answer is True
    assert parent_question.parent_backfill_improves is True

    report_text = result.report_markdown_path.read_text(encoding="utf-8")
    assert "人工检查怎么做" in report_text
    assert "table chunk" in report_text
    assert "image_summary" in report_text


def test_offline_eval_service_supports_single_user_file_and_question_bank(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "user-note.md"
    document_path.write_text(
        "# 项目复盘\n\n"
        "## 发布门槛\n\n"
        "第一道门槛是检索精度稳定在 95% 以上。"
        "第二道门槛是 metadata 缺失为零。\n\n"
        "## 数据表\n\n"
        "| 指标 | 数值 |\n"
        "| --- | ---: |\n"
        "| 告警 | 7 |\n",
        encoding="utf-8",
    )
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        """{
  "questions": [
    {
      "question_id": "release_gate",
      "question": "发布前要满足什么门槛？",
      "category": "parent_backfill",
      "expected_terms": ["95%", "metadata 缺失为零"],
      "min_expected_terms": 2,
      "expected_chunk_role": "child",
      "expect_parent_backfill": true
    },
    {
      "question_id": "table_alerts",
      "question": "数据表里的告警是多少？",
      "category": "table",
      "expected_terms": ["7"],
      "expected_chunk_role": "special",
      "expected_special_chunk_type": "table"
    }
  ]
}""",
        encoding="utf-8",
    )

    result = OfflineEvalService(output_dir=tmp_path / "single-file").run_file(
        file_path=document_path,
        questions_path=questions_path,
    )

    assert result.report.summary.total_documents == 1
    assert result.report.summary.total_questions == 2
    assert result.report.fixtures[0].path == document_path
    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()

    release_gate = next(
        item for item in result.report.question_results if item.question_id == "release_gate"
    )
    assert release_gate.runtime_hit is True

    table_question = next(
        item for item in result.report.question_results if item.question_id == "table_alerts"
    )
    assert any(
        hit.special_chunk_type == "table"
        for hit in table_question.runtime_top_k
        if hit.is_expected_hit
    )
