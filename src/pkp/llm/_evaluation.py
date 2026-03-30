from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from pkp.schema._types.generation import GroundedAnswer


@dataclass(frozen=True)
class AnswerEvaluationCase:
    case_id: str
    query: str
    expected_chunk_ids: list[str]
    expected_special_chunk_types: list[str]
    expected_insufficient: bool
    answer: GroundedAnswer


class AnswerEvaluationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_count: int
    grounded_rate: float
    citation_presence_rate: float
    insufficient_evidence_accuracy: float
    expected_chunk_hit_rate: float
    special_chunk_hit_rate: float
    case_ids: list[str] = Field(default_factory=list)


class AnswerEvaluationService:
    def evaluate(self, *, cases: list[AnswerEvaluationCase]) -> AnswerEvaluationSummary:
        if not cases:
            return AnswerEvaluationSummary(
                case_count=0,
                grounded_rate=0.0,
                citation_presence_rate=0.0,
                insufficient_evidence_accuracy=0.0,
                expected_chunk_hit_rate=0.0,
                special_chunk_hit_rate=0.0,
                case_ids=[],
            )

        grounded_scores = [1.0 if case.answer.groundedness_flag else 0.0 for case in cases]
        citation_scores = [1.0 if case.answer.citations else 0.0 for case in cases]
        insufficient_scores = [
            1.0 if case.answer.insufficient_evidence_flag == case.expected_insufficient else 0.0 for case in cases
        ]
        chunk_scores = [1.0 if self._hits_expected_chunks(case) else 0.0 for case in cases]
        special_scores = [1.0 if self._hits_expected_special_chunks(case) else 0.0 for case in cases]
        return AnswerEvaluationSummary(
            case_count=len(cases),
            grounded_rate=mean(grounded_scores),
            citation_presence_rate=mean(citation_scores),
            insufficient_evidence_accuracy=mean(insufficient_scores),
            expected_chunk_hit_rate=mean(chunk_scores),
            special_chunk_hit_rate=mean(special_scores),
            case_ids=[case.case_id for case in cases],
        )

    @staticmethod
    def _hits_expected_chunks(case: AnswerEvaluationCase) -> bool:
        if not case.expected_chunk_ids:
            return True
        actual = {link.evidence_chunk_id for link in case.answer.evidence_links}
        return bool(actual & set(case.expected_chunk_ids))

    @staticmethod
    def _hits_expected_special_chunks(case: AnswerEvaluationCase) -> bool:
        if not case.expected_special_chunk_types:
            return True
        actual = {citation.chunk_type for citation in case.answer.citations}
        return bool(actual & set(case.expected_special_chunk_types))
