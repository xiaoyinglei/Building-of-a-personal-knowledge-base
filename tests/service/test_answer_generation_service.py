from __future__ import annotations

from pkp.llm.generation import AnswerGenerationService
from pkp.types import ChunkRole, EvidenceItem, RuntimeMode


def _evidence_item(
    *,
    chunk_id: str,
    text: str,
    chunk_role: ChunkRole = ChunkRole.CHILD,
    special_chunk_type: str | None = None,
    file_name: str = "monthly-report.docx",
    section_path: list[str] | None = None,
    page_start: int | None = 2,
    page_end: int | None = 2,
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_id="src-1",
        citation_anchor="专项工作",
        text=text,
        score=0.95,
        evidence_kind="internal",
        chunk_role=chunk_role,
        special_chunk_type=special_chunk_type,
        parent_chunk_id="parent-1",
        file_name=file_name,
        section_path=section_path or ["专项工作", "月返抽查"],
        page_start=page_start,
        page_end=page_end,
        chunk_type=special_chunk_type or chunk_role.value,
        source_type="docx",
    )


def test_answer_generation_service_returns_grounded_answer_with_structured_citations() -> None:
    service = AnswerGenerationService()

    def fake_model(prompt: str) -> str:
        assert "E1" in prompt
        assert "只基于证据" in prompt
        return """
        {
          "answer_text": "一月我完成了问题核查、异常复核和整改跟进。",
          "answer_sections": [
            {
              "title": "直接回答",
              "text": "一月我完成了问题核查、异常复核和整改跟进。",
              "evidence_ids": ["E1"]
            }
          ],
          "insufficient_evidence_flag": false
        }
        """

    result = service.generate(
        query="龙牌集团月返抽查专项工作中，一月我完成了哪些工作？",
        evidence_pack=[
            _evidence_item(
                chunk_id="chunk-1",
                text="一月完成了问题核查、异常复核、整改跟进和结果回填。",
            )
        ],
        model_generate=fake_model,
        runtime_mode=RuntimeMode.DEEP,
    )

    assert result.answer_text == "一月我完成了问题核查、异常复核和整改跟进。"
    assert result.groundedness_flag is True
    assert result.insufficient_evidence_flag is False
    assert len(result.answer_sections) == 1
    assert len(result.citations) == 1
    assert result.citations[0].file_name == "monthly-report.docx"
    assert result.citations[0].section_path == ["专项工作", "月返抽查"]
    assert result.citations[0].page_start == 2
    assert result.citations[0].page_end == 2
    assert result.citations[0].chunk_id == "chunk-1"
    assert result.citations[0].chunk_type == "child"
    assert result.answer_sections[0].citation_ids == [result.citations[0].citation_id]
    assert result.evidence_links[0].evidence_chunk_id == "chunk-1"


def test_answer_generation_service_marks_insufficient_evidence_without_calling_model() -> None:
    service = AnswerGenerationService()

    def fail_model(_prompt: str) -> str:
        raise AssertionError("model should not be called when evidence is insufficient")

    result = service.generate(
        query="这个文档说明了三月份新增预算是多少？",
        evidence_pack=[
            _evidence_item(
                chunk_id="chunk-2",
                text="本月主要完成了现场抽查和问题登记。",
            )
        ],
        model_generate=fail_model,
        runtime_mode=RuntimeMode.DEEP,
    )

    assert result.insufficient_evidence_flag is True
    assert result.groundedness_flag is True
    assert "不足" in result.answer_text
    assert result.citations == []
    assert result.evidence_links == []


def test_answer_generation_service_preserves_table_and_image_special_chunk_citations() -> None:
    service = AnswerGenerationService()

    def fake_model(_prompt: str) -> str:
        return """
        {
          "answer_text": "表格显示告警数量为 7，图片摘要说明页面展示了 Fast Path 和 Deep Path。",
          "answer_sections": [
            {
              "title": "表格信息",
              "text": "表格显示告警数量为 7。",
              "evidence_ids": ["E1"]
            },
            {
              "title": "图片信息",
              "text": "图片摘要说明页面展示了 Fast Path 和 Deep Path。",
              "evidence_ids": ["E2"]
            }
          ],
          "insufficient_evidence_flag": false
        }
        """

    result = service.generate(
        query="表格里的告警数量和图片展示内容分别是什么？",
        evidence_pack=[
            _evidence_item(
                chunk_id="table-1",
                text="| 指标 | 数值 |\\n| 告警 | 7 |",
                chunk_role=ChunkRole.SPECIAL,
                special_chunk_type="table",
                file_name="ops-report.pdf",
                section_path=["专项工作", "统计表"],
                page_start=3,
                page_end=3,
            ),
            _evidence_item(
                chunk_id="image-1",
                text="页面展示了 Fast Path 和 Deep Path 两个入口。",
                chunk_role=ChunkRole.SPECIAL,
                special_chunk_type="image_summary",
                file_name="ops-report.pdf",
                section_path=["界面截图"],
                page_start=4,
                page_end=4,
            ),
        ],
        model_generate=fake_model,
        runtime_mode=RuntimeMode.DEEP,
    )

    assert result.groundedness_flag is True
    assert {citation.chunk_type for citation in result.citations} == {"table", "image_summary"}
    assert {link.evidence_chunk_id for link in result.evidence_links} == {"table-1", "image-1"}
