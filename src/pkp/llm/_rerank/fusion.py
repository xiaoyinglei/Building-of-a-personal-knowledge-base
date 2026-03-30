from __future__ import annotations

from math import sqrt

from pkp.llm._rerank.interfaces import ScoreCombinerProtocol
from pkp.llm._rerank.models import FeatureRecord, RerankCandidate


class FeatureBasedScoreCombiner(ScoreCombinerProtocol):
    def combine(
        self,
        *,
        candidate: RerankCandidate,
        feature_record: FeatureRecord,
    ) -> tuple[float, dict[str, float | int | bool | str]]:
        features = feature_record.feature_dict
        retrieval_signal = max(
            float(features.get("dense_score", 0.0)),
            float(features.get("sparse_score", 0.0)),
            float(features.get("special_score", 0.0)),
            float(features.get("fusion_score", 0.0)),
            float(features.get("rrf_score", 0.0)),
        )
        cross_encoder_signal = float(features.get("cross_encoder_score", 0.0))
        query_match_signal = min(
            1.0,
            (
                float(features.get("title_hit", 0))
                + float(features.get("section_path_hit", 0))
                + float(features.get("token_overlap", 0))
            )
            / 8.0
            + (0.18 if bool(features.get("exact_phrase_hit", False)) else 0.0)
            + (0.08 if bool(features.get("number_match", False)) else 0.0),
        )
        structure_signal = min(
            1.0,
            (0.22 if bool(features.get("heading_level_match", 0)) else 0.0)
            + (0.18 if bool(features.get("parent_section_match", False)) else 0.0)
            + (0.18 if bool(features.get("page_match", False)) else 0.0)
            + (0.22 if bool(features.get("is_table", False)) else 0.0)
            + (0.12 if bool(features.get("is_ocr_region", False)) else 0.0)
            + (0.12 if bool(features.get("is_image_summary", False)) else 0.0),
        )
        channel_bonus = min(0.12, float(features.get("retrieval_channel_count", 0)) * 0.03)
        parent_penalty = 0.08 if int(features.get("same_parent_candidate_count", 0)) > 1 else 0.0
        if bool(features.get("query_is_definition_like", False)) and bool(features.get("definition_text_hit", False)):
            query_match_signal = min(1.0, query_match_signal + 0.2)
        if (
            not bool(features.get("query_is_command_like", False))
            and bool(features.get("candidate_is_command_like", False))
        ):
            parent_penalty += 0.3
        if bool(features.get("query_is_structure_like", False)) and bool(features.get("structure_section_hit", False)):
            structure_signal = min(1.0, structure_signal + 0.18)
        length_penalty = 0.0
        candidate_length = int(features.get("candidate_length", 0))
        if candidate_length < 10:
            length_penalty += 0.06
        if candidate_length > 420:
            length_penalty += 0.05

        base_signal = (
            sqrt(max(cross_encoder_signal, 0.0)) * 0.5
            + sqrt(max(query_match_signal, 0.0)) * 0.22
            + sqrt(max(retrieval_signal, 0.0)) * 0.18
            + sqrt(max(structure_signal, 0.0)) * 0.1
        )
        final_score = round(
            max(
                0.0,
                min(1.0, base_signal * (1.0 + channel_bonus) - parent_penalty - length_penalty),
            ),
            6,
        )

        return final_score, {
            "retrieval_signal": round(retrieval_signal, 6),
            "cross_encoder_signal": round(cross_encoder_signal, 6),
            "query_match_signal": round(query_match_signal, 6),
            "structure_signal": round(structure_signal, 6),
            "channel_bonus": round(channel_bonus, 6),
            "parent_penalty": round(parent_penalty, 6),
            "length_penalty": round(length_penalty, 6),
            "chunk_type": candidate.chunk_type,
        }
