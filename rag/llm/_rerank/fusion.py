from __future__ import annotations

from rag.llm._rerank.interfaces import ScoreCombinerProtocol
from rag.llm._rerank.models import FeatureRecord, RerankCandidate


class FeatureBasedScoreCombiner(ScoreCombinerProtocol):
    def combine(
        self,
        *,
        candidate: RerankCandidate,
        feature_record: FeatureRecord,
    ) -> tuple[float, dict[str, float | int | bool | str]]:
        del candidate
        features = feature_record.feature_dict
        retrieval_signal = max(
            float(features.get("dense_score", 0.0)),
            float(features.get("sparse_score", 0.0)),
            float(features.get("special_score", 0.0)),
            float(features.get("fusion_score", 0.0)),
            float(features.get("rrf_score", 0.0)),
        )
        cross_encoder_signal = float(features.get("cross_encoder_score", 0.0))
        special_target_match = bool(features.get("special_target_match", False))
        query_requires_special = bool(features.get("query_requires_special", False))
        constraint_match = bool(
            features.get("page_match", False)
            or features.get("parent_section_match", False)
            or features.get("exact_phrase_hit", False)
            or special_target_match
        )
        if special_target_match and query_requires_special:
            final_score = round(max(cross_encoder_signal, retrieval_signal), 6)
        else:
            final_score = round(cross_encoder_signal if cross_encoder_signal > 0.0 else retrieval_signal, 6)
        return final_score, {
            "cross_encoder_signal": round(cross_encoder_signal, 6),
            "retrieval_signal": round(retrieval_signal, 6),
            "constraint_match": constraint_match,
            "special_target_match": special_target_match,
        }
