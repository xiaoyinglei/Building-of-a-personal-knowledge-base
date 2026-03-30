from __future__ import annotations

from enum import StrEnum
from typing import Any, cast

from rag.llm._rerank.models import RerankResponse, TrainingCandidateSnapshot, TrainingSample


class ExportFormat(StrEnum):
    PAIRWISE = "pairwise"
    POINTWISE = "pointwise"
    LISTWISE = "listwise"


class TrainingSampleExporter:
    def export(
        self,
        *,
        response: RerankResponse,
        positive_chunk_ids: set[str],
        export_format: ExportFormat,
        hard_negative_limit: int = 3,
    ) -> list[TrainingSample]:
        positives = [item for item in response.items if item.chunk_id in positive_chunk_ids]
        negatives = [item for item in response.items if item.chunk_id not in positive_chunk_ids][:hard_negative_limit]
        if not positives:
            return []

        if export_format is ExportFormat.LISTWISE:
            return [
                TrainingSample(
                    query=response.query,
                    positive_candidate=self._snapshot(positives[0]),
                    hard_negative_candidates=[self._snapshot(item) for item in negatives],
                    retrieval_context=[item.chunk_id for item in response.raw_candidates],
                    rerank_context={"export_format": export_format.value},
                    feature_snapshot=positives[0].feature_summary,
                )
            ]

        if export_format is ExportFormat.POINTWISE:
            return [
                TrainingSample(
                    query=response.query,
                    positive_candidate=self._snapshot(positive),
                    hard_negative_candidates=[],
                    retrieval_context=[item.chunk_id for item in response.raw_candidates],
                    rerank_context={"export_format": export_format.value, "rank_after": positive.rank_after},
                    feature_snapshot=positive.feature_summary,
                )
                for positive in positives
            ]

        samples: list[TrainingSample] = []
        for positive in positives:
            for negative in negatives:
                samples.append(
                    TrainingSample(
                        query=response.query,
                        positive_candidate=self._snapshot(positive),
                        hard_negative_candidates=[self._snapshot(negative)],
                        retrieval_context=[item.chunk_id for item in response.raw_candidates],
                        rerank_context={
                            "export_format": export_format.value,
                            "positive_rank_after": positive.rank_after,
                            "negative_rank_after": negative.rank_after,
                        },
                        feature_snapshot=positive.feature_summary,
                    )
                )
        return samples

    @staticmethod
    def _snapshot(item: object) -> TrainingCandidateSnapshot:
        item_any = cast(Any, item)
        metadata = getattr(item_any, "metadata", {})
        return TrainingCandidateSnapshot(
            chunk_id=item_any.chunk_id,
            doc_id=item_any.doc_id,
            parent_id=getattr(item_any, "parent_id", None),
            chunk_type=item_any.chunk_type,
            text=item_any.text,
            metadata=metadata if isinstance(metadata, dict) else {},
        )
