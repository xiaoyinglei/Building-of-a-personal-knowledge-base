from pkp.service.telemetry_service import compute_evaluation_metrics, summarize_evaluation_metrics
from pkp.types.telemetry import EvaluationMetricInput


def test_compute_evaluation_metrics_accepts_legacy_dicts() -> None:
    metrics = compute_evaluation_metrics(
        [
            {
                "citation_precision": 1,
                "evidence_sufficient": True,
                "conflict_detected": False,
                "latency_seconds": 2,
                "deep_quality": 0.4,
                "preservation_useful": True,
            },
            {
                "citation_precision": 0.5,
                "evidence_sufficient": False,
                "conflict_detected": True,
                "latency_seconds": 4,
                "deep_quality": 0.8,
                "preservation_useful": False,
            },
        ]
    )

    assert metrics == {
        "citation_precision": 0.75,
        "evidence_sufficiency_rate": 0.5,
        "conflict_detection_quality": 0.5,
        "simple_query_latency": 3.0,
        "deep_query_completion_quality": 0.6,
        "preservation_usefulness": 0.5,
    }


def test_summarize_evaluation_metrics_handles_empty_input() -> None:
    metrics = summarize_evaluation_metrics([])

    assert metrics.model_dump() == {
        "citation_precision": 0.0,
        "evidence_sufficiency_rate": 0.0,
        "conflict_detection_quality": 0.0,
        "simple_query_latency": 0.0,
        "deep_query_completion_quality": 0.0,
        "preservation_usefulness": 0.0,
    }


def test_evaluation_metric_input_supports_legacy_field_names() -> None:
    sample = EvaluationMetricInput.model_validate(
        {
            "citation_precision": 0.9,
            "evidence_sufficient": True,
            "conflict_detected": False,
            "latency_seconds": 1.5,
            "deep_quality": 0.7,
            "preservation_useful": True,
        }
    )

    assert sample.simple_query_latency_seconds == 1.5
    assert sample.deep_query_completion_quality == 0.7
