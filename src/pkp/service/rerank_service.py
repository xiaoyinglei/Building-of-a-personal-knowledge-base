from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pkp.types.text import (
    keyword_overlap,
    looks_command_like,
    looks_definition_query,
    looks_definition_text,
    search_terms,
)


class CandidateLike(Protocol):
    chunk_id: str
    text: str
    score: float
    section_path: Sequence[str]


class HeuristicRerankService:
    def rerank(self, query: str, candidates: Sequence[CandidateLike]) -> list[CandidateLike]:
        query_terms = search_terms(query)
        normalized_query = query.strip().lower()
        query_is_command_like = looks_command_like(query)
        query_is_definition_like = looks_definition_query(query)

        def _score(candidate: CandidateLike) -> tuple[float, float, str]:
            text = candidate.text
            section_text = " ".join(candidate.section_path)
            overlap = keyword_overlap(query_terms, text)
            section_overlap = keyword_overlap(query_terms, section_text)
            phrase_bonus = 0.0
            command_penalty = 0.0
            definition_bonus = 0.0
            lowered_text = text.lower()
            lowered_section = section_text.lower()
            if normalized_query and normalized_query in lowered_text:
                phrase_bonus += 2.0
            if normalized_query and normalized_query in lowered_section:
                phrase_bonus += 1.0
            if not query_is_command_like and looks_command_like(text):
                command_penalty = 5.0
            if query_is_definition_like and not looks_command_like(text) and looks_definition_text(text):
                definition_bonus = 4.0
            combined = (
                float(candidate.score)
                + overlap * 0.8
                + section_overlap * 0.5
                + phrase_bonus
                + definition_bonus
                - command_penalty
            )
            return (combined, float(candidate.score), candidate.chunk_id)

        return sorted(candidates, key=_score, reverse=True)
