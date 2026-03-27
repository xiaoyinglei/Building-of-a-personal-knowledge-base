from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pkp.rerank.models import RerankEvaluationCase, RerankEvaluationSummary, StageMetrics


class RerankEvaluator:
    def evaluate(
        self,
        *,
        cases: list[RerankEvaluationCase],
        stage_rankings: dict[str, dict[str, list[str]]],
    ) -> RerankEvaluationSummary:
        by_stage = {
            stage: self._metrics_for_cases(cases, ranking_map)
            for stage, ranking_map in stage_rankings.items()
        }
        return RerankEvaluationSummary(
            by_stage=by_stage,
            by_query_type=self._grouped_metrics(cases, stage_rankings, key="query_type"),
            by_source_type=self._grouped_metrics(cases, stage_rankings, key="source_type"),
            by_chunk_type=self._grouped_metrics(cases, stage_rankings, key="expected_chunk_type"),
        )

    def _grouped_metrics(
        self,
        cases: list[RerankEvaluationCase],
        stage_rankings: dict[str, dict[str, list[str]]],
        *,
        key: str,
    ) -> dict[str, dict[str, StageMetrics]]:
        grouped_cases: dict[str, list[RerankEvaluationCase]] = defaultdict(list)
        for case in cases:
            value = getattr(case, key) or "unknown"
            grouped_cases[str(value)].append(case)
        return {
            group: {
                stage: self._metrics_for_cases(group_cases, ranking_map)
                for stage, ranking_map in stage_rankings.items()
            }
            for group, group_cases in grouped_cases.items()
        }

    def _metrics_for_cases(
        self,
        cases: Iterable[RerankEvaluationCase],
        ranking_map: dict[str, list[str]],
    ) -> StageMetrics:
        hit_at_1 = 0.0
        hit_at_3 = 0.0
        hit_at_5 = 0.0
        reciprocal_rank = 0.0
        ndcg = 0.0
        case_list = list(cases)
        total = len(case_list)
        if total == 0:
            return StageMetrics(hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr=0.0, ndcg_at_5=0.0)

        for case in case_list:
            rankings = ranking_map.get(case.query_id, [])
            relevant = set(case.expected_chunk_ids)
            if rankings[:1] and rankings[0] in relevant:
                hit_at_1 += 1.0
            if any(chunk_id in relevant for chunk_id in rankings[:3]):
                hit_at_3 += 1.0
            if any(chunk_id in relevant for chunk_id in rankings[:5]):
                hit_at_5 += 1.0
            reciprocal_rank += self._reciprocal_rank(rankings, relevant)
            ndcg += self._ndcg_at_5(rankings, relevant)

        return StageMetrics(
            hit_at_1=round(hit_at_1 / total, 4),
            hit_at_3=round(hit_at_3 / total, 4),
            hit_at_5=round(hit_at_5 / total, 4),
            mrr=round(reciprocal_rank / total, 4),
            ndcg_at_5=round(ndcg / total, 4),
        )

    @staticmethod
    def _reciprocal_rank(rankings: list[str], relevant: set[str]) -> float:
        for index, chunk_id in enumerate(rankings, start=1):
            if chunk_id in relevant:
                return 1.0 / index
        return 0.0

    @staticmethod
    def _ndcg_at_5(rankings: list[str], relevant: set[str]) -> float:
        dcg = 0.0
        for index, chunk_id in enumerate(rankings[:5], start=1):
            if chunk_id in relevant:
                dcg += 1.0 if index == 1 else 1.0 / index
        ideal = 1.0
        return 0.0 if ideal == 0 else dcg / ideal
