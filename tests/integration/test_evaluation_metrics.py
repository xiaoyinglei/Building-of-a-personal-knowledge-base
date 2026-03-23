from pkp.service.telemetry_service import summarize_evaluation_metrics
from pkp.types.telemetry import EvaluationMetricInput


def test_compute_evaluation_metrics_summarizes_quality_dimensions() -> None:
    metrics = summarize_evaluation_metrics(
        [
            EvaluationMetricInput(
                citation_precision=0.9,
                evidence_sufficient=True,
                conflict_detected=True,
                simple_query_latency_seconds=1.2,
                deep_query_completion_quality=0.8,
                preservation_useful=True,
            ),
            EvaluationMetricInput(
                citation_precision=0.7,
                evidence_sufficient=False,
                conflict_detected=False,
                simple_query_latency_seconds=2.8,
                deep_query_completion_quality=0.6,
                preservation_useful=False,
            ),
        ]
    )

    assert metrics.citation_precision == 0.8
    assert metrics.evidence_sufficiency_rate == 0.5
    assert metrics.conflict_detection_quality == 0.5
    assert metrics.simple_query_latency == 2.0
    assert metrics.deep_query_completion_quality == 0.7
    assert metrics.preservation_usefulness == 0.5
