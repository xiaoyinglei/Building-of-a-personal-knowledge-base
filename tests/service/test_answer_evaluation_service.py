from __future__ import annotations

from pkp.service.answer_evaluation_service import AnswerEvaluationCase, AnswerEvaluationService
from pkp.types.generation import (
    AnswerCitation,
    AnswerEvidenceLink,
    AnswerSection,
    GroundedAnswer,
)


def _answer(*, insufficient: bool, grounded: bool, chunk_id: str, chunk_type: str) -> GroundedAnswer:
    citation = AnswerCitation(
        citation_id="cit-1",
        file_name="report.docx",
        section_path=["专项工作"],
        page_start=2,
        page_end=2,
        chunk_id=chunk_id,
        chunk_type=chunk_type,
    )
    return GroundedAnswer(
        answer_text="answer",
        answer_sections=[
            AnswerSection(
                section_id="sec-1",
                title="直接回答",
                text="answer",
                citation_ids=["cit-1"],
                evidence_chunk_ids=[chunk_id],
            )
        ],
        citations=[citation],
        evidence_links=[
            AnswerEvidenceLink(
                link_id="link-1",
                answer_section_id="sec-1",
                answer_excerpt="answer",
                evidence_chunk_id=chunk_id,
                citation_id="cit-1",
                support_score=0.9,
            )
        ],
        groundedness_flag=grounded,
        insufficient_evidence_flag=insufficient,
    )


def test_answer_evaluation_service_scores_groundedness_citation_and_special_usage() -> None:
    service = AnswerEvaluationService()
    summary = service.evaluate(
        cases=[
            AnswerEvaluationCase(
                case_id="case-1",
                query="表格里的告警数量是多少？",
                expected_chunk_ids=["table-1"],
                expected_special_chunk_types=["table"],
                expected_insufficient=False,
                answer=_answer(insufficient=False, grounded=True, chunk_id="table-1", chunk_type="table"),
            ),
            AnswerEvaluationCase(
                case_id="case-2",
                query="这个文档说明了三月份新增预算是多少？",
                expected_chunk_ids=[],
                expected_special_chunk_types=[],
                expected_insufficient=True,
                answer=_answer(insufficient=True, grounded=True, chunk_id="none", chunk_type="child"),
            ),
        ]
    )

    assert summary.case_count == 2
    assert summary.grounded_rate == 1.0
    assert summary.citation_presence_rate == 1.0
    assert summary.insufficient_evidence_accuracy == 1.0
    assert summary.expected_chunk_hit_rate == 1.0
    assert summary.special_chunk_hit_rate == 1.0
