from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pkp.llm._rerank.models import FeatureRecord, RerankCandidate, RerankRequest
from pkp.schema._types.text import (
    keyword_overlap,
    looks_command_like,
    looks_definition_query,
    looks_definition_text,
    looks_structure_query,
    looks_structure_text,
    search_terms,
    text_unit_count,
)


class RerankFeatureExtractor:
    def extract(
        self,
        *,
        request: RerankRequest,
        candidates: Sequence[RerankCandidate],
        cross_encoder_scores: Sequence[float],
    ) -> list[FeatureRecord]:
        query_terms = search_terms(request.query)
        query_numbers = {term for term in query_terms if term.isdigit()}
        query_is_command_like = looks_command_like(request.query)
        query_is_definition_like = looks_definition_query(request.query)
        query_is_structure_like = looks_structure_query(request.query)
        parent_counts = Counter(candidate.parent_id for candidate in candidates if candidate.parent_id)
        max_order_index = max((int(candidate.metadata.get("order_index", "0")) for candidate in candidates), default=0)

        records: list[FeatureRecord] = []
        for index, candidate in enumerate(candidates):
            heading_text = candidate.heading_text or ""
            section_text = " ".join(candidate.section_path)
            parent_text = candidate.parent_text or ""
            metadata = candidate.metadata
            text_numbers = {term for term in search_terms(candidate.text) if term.isdigit()}
            page_numbers = {
                value
                for value in (
                    str(candidate.page_start) if candidate.page_start is not None else "",
                    str(candidate.page_end) if candidate.page_end is not None else "",
                    metadata.get("page_no", ""),
                )
                if value
            }
            preferred_sections = self._as_str_set(
                request.query_analysis.structure_constraints.get("preferred_section_terms")
            )
            requested_pages = self._as_str_set(request.query_analysis.metadata_filters.get("page_numbers"))
            feature_dict: dict[str, float | int | bool | str] = {
                "dense_score": round(candidate.dense_score or 0.0, 6),
                "sparse_score": round(candidate.sparse_score or 0.0, 6),
                "special_score": round(candidate.special_score or 0.0, 6),
                "structure_score": round(candidate.structure_score or 0.0, 6),
                "metadata_score": round(candidate.metadata_score or 0.0, 6),
                "fusion_score": round(candidate.fusion_score or 0.0, 6),
                "rrf_score": round(candidate.rrf_score or 0.0, 6),
                "retrieval_rank": candidate.unified_rank,
                "retrieval_channel_count": len(candidate.retrieval_channels),
                "title_hit": keyword_overlap(query_terms, heading_text),
                "section_path_hit": keyword_overlap(query_terms, section_text),
                "token_overlap": keyword_overlap(query_terms, candidate.text),
                "number_match": bool(query_numbers & (text_numbers | page_numbers)),
                "exact_phrase_hit": request.query.strip().lower() in candidate.text.lower(),
                "heading_level_match": 1 if preferred_sections & set(candidate.section_path) else 0,
                "candidate_is_command_like": looks_command_like(candidate.text),
                "query_is_command_like": query_is_command_like,
                "query_is_definition_like": query_is_definition_like,
                "query_is_structure_like": query_is_structure_like,
                "definition_text_hit": looks_definition_text(candidate.text),
                "definition_section_hit": looks_definition_text(section_text),
                "structure_text_hit": looks_structure_text(candidate.text),
                "structure_section_hit": looks_structure_text(section_text),
                "is_table": candidate.chunk_type == "table",
                "is_figure": candidate.chunk_type == "figure",
                "is_ocr_region": candidate.chunk_type == "ocr_region",
                "is_image_summary": candidate.chunk_type == "image_summary",
                "parent_section_match": bool(preferred_sections & set(candidate.section_path)),
                "page_match": bool(requested_pages & page_numbers),
                "source_type": metadata.get("source_type", ""),
                "candidate_length": text_unit_count(candidate.text),
                "parent_length": text_unit_count(parent_text),
                "same_parent_candidate_count": 0
                if candidate.parent_id is None
                else parent_counts.get(candidate.parent_id, 0),
                "relative_position": (
                    0.0
                    if max_order_index <= 0
                    else round(int(metadata.get("order_index", "0")) / max_order_index, 6)
                ),
                "cross_encoder_score": round(float(cross_encoder_scores[index]), 6),
            }
            records.append(FeatureRecord(chunk_id=candidate.chunk_id, feature_dict=feature_dict))
        return records

    @staticmethod
    def _as_str_set(value: list[str] | str | bool | None) -> set[str]:
        if isinstance(value, list):
            return {item for item in value if isinstance(item, str)}
        if isinstance(value, str):
            return {value}
        return set()
