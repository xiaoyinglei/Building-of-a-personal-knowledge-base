from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pkp.service.query_understanding_service import QueryUnderstandingService
from pkp.types.text import (
    focus_terms,
    keyword_overlap,
    looks_command_like,
    looks_definition_query,
    looks_definition_text,
    looks_structure_query,
    looks_structure_text,
    search_terms,
)


class CandidateLike(Protocol):
    chunk_id: str
    text: str
    score: float
    section_path: Sequence[str]
    special_chunk_type: str | None
    chunk_role: object | None
    metadata: dict[str, str] | None


class HeuristicRerankService:
    def __init__(self, query_understanding_service: QueryUnderstandingService | None = None) -> None:
        self._query_understanding_service = query_understanding_service or QueryUnderstandingService()

    def rerank(self, query: str, candidates: Sequence[CandidateLike]) -> list[CandidateLike]:
        query_understanding = self._query_understanding_service.analyze(query)
        query_terms = search_terms(query)
        query_focus_terms = focus_terms(query)
        normalized_query = query.strip().lower()
        query_is_command_like = looks_command_like(query)
        query_is_definition_like = looks_definition_query(query)
        query_is_structure_like = looks_structure_query(query)
        preferred_section_terms = self._as_str_list(
            query_understanding.structure_constraints.get("preferred_section_terms")
        )
        page_filters = self._as_str_list(query_understanding.metadata_filters.get("page_numbers"))

        def _score(candidate: CandidateLike) -> tuple[float, float, str]:
            text = candidate.text
            section_text = " ".join(candidate.section_path)
            candidate_is_command_like = looks_command_like(text)
            special_chunk_type = (candidate.special_chunk_type or "").lower()
            metadata = candidate.metadata or {}
            overlap = keyword_overlap(query_terms, text)
            section_overlap = keyword_overlap(query_terms, section_text)
            focus_overlap = keyword_overlap(query_focus_terms, text)
            focus_section_overlap = keyword_overlap(query_focus_terms, section_text)
            phrase_bonus = 0.0
            command_penalty = 0.0
            definition_bonus = 0.0
            structure_bonus = 0.0
            special_bonus = 0.0
            metadata_bonus = 0.0
            parent_penalty = 0.0
            text_overlap_weight = 0.8
            section_overlap_weight = 0.5
            focus_overlap_weight = 0.6
            focus_section_overlap_weight = 0.8
            lowered_text = text.lower()
            lowered_section = section_text.lower()
            if normalized_query and normalized_query in lowered_text and (
                query_is_command_like or not candidate_is_command_like
            ):
                phrase_bonus += 2.0
            if normalized_query and normalized_query in lowered_section:
                phrase_bonus += 1.0
            if not query_is_command_like and candidate_is_command_like:
                text_overlap_weight = 0.2
                section_overlap_weight = 0.2
                focus_overlap_weight = 0.15
                focus_section_overlap_weight = 0.15
                command_penalty = 8.0
                if query_is_definition_like and not query_is_structure_like:
                    command_penalty += 4.0
            if (
                query_is_definition_like
                and not query_is_structure_like
                and not candidate_is_command_like
            ):
                if looks_definition_text(text):
                    definition_bonus += 5.0
                if looks_definition_text(section_text):
                    definition_bonus += 2.0
            if query_is_structure_like:
                if looks_structure_text(section_text) and focus_section_overlap > 0:
                    structure_bonus += 4.0
                if looks_structure_text(text) and focus_overlap > 0:
                    structure_bonus += 2.0
                structure_bonus += focus_section_overlap * 2.0
                structure_bonus += focus_overlap * 0.8
            if preferred_section_terms:
                for term in preferred_section_terms:
                    lowered_term = term.lower()
                    if lowered_term and lowered_term in lowered_section:
                        structure_bonus += 5.0
                    if lowered_term and lowered_term in lowered_text:
                        structure_bonus += 1.5
            if query_understanding.special_targets:
                if special_chunk_type in query_understanding.special_targets:
                    special_bonus += 6.0
                elif special_chunk_type:
                    special_bonus -= 1.5
            if page_filters:
                page_no = metadata.get("page_no", "")
                if page_no in page_filters:
                    metadata_bonus += 4.0
            chunk_role = getattr(candidate, "chunk_role", None)
            chunk_role_value = getattr(chunk_role, "value", chunk_role)
            if chunk_role_value == "parent":
                parent_penalty += 0.4
            combined = (
                float(candidate.score)
                + overlap * text_overlap_weight
                + section_overlap * section_overlap_weight
                + focus_overlap * focus_overlap_weight
                + focus_section_overlap * focus_section_overlap_weight
                + phrase_bonus
                + definition_bonus
                + structure_bonus
                + special_bonus
                + metadata_bonus
                - command_penalty
                - parent_penalty
            )
            return (combined, float(candidate.score), candidate.chunk_id)

        return sorted(candidates, key=_score, reverse=True)

    @staticmethod
    def _as_str_list(value: list[str] | str | bool | None) -> list[str]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        if isinstance(value, str):
            return [value]
        return []
