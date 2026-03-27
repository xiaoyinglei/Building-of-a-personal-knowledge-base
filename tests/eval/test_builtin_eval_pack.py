from pathlib import Path

from pkp.eval.sample_pack import prepare_builtin_eval_pack
from pkp.types.content import SourceType


def test_prepare_builtin_eval_pack_generates_four_supported_file_types(tmp_path: Path) -> None:
    pack = prepare_builtin_eval_pack(tmp_path)

    assert {fixture.source_type for fixture in pack.fixtures} == {
        SourceType.MARKDOWN,
        SourceType.DOCX,
        SourceType.PDF,
        SourceType.IMAGE,
    }
    assert {fixture.path.name for fixture in pack.fixtures} == {
        "quarterly-review.md",
        "operations-brief.docx",
        "research-notes.pdf",
        "dashboard-metrics.png",
    }
    assert all(fixture.path.exists() for fixture in pack.fixtures)
    assert len(pack.questions) >= 6
    assert {question.question_id for question in pack.questions} >= {
        "markdown_revenue_reason",
        "markdown_release_gates",
        "docx_alert_table",
        "pdf_deep_path_conflict",
        "image_ocr_revenue",
        "image_summary_scene",
    }
