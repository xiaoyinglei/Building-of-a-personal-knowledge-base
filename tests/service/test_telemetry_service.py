from pkp.service.telemetry_service import TelemetryService
from pkp.types.telemetry import TelemetryEvent


def test_telemetry_service_records_typed_events() -> None:
    service = TelemetryService.create_in_memory()

    event = TelemetryEvent(
        name="retrieval.branch_used",
        category="retrieval",
        payload={"branch": "vector", "count": 3},
    )
    service.record(event)

    events = service.list_events()
    assert len(events) == 1
    assert events[0].name == "retrieval.branch_used"
    assert events[0].payload["branch"] == "vector"


def test_telemetry_service_records_standard_runtime_events() -> None:
    service = TelemetryService.create_in_memory()

    service.record_branch_usage(branch="full_text", hit_count=2, runtime_mode="deep")
    service.record_graph_expansion(seed_count=2, added_count=1)
    service.record_fast_to_deep_escalation(reason="insufficient_evidence")
    service.record_claim_citation_failure(response_mode="fast", evidence_count=1)
    service.record_artifact_approved(artifact_id="artifact-1", artifact_type="topic_page")

    events = service.list_events()

    assert [event.name for event in events] == [
        "retrieval.branch_used",
        "retrieval.graph_expanded",
        "runtime.escalated_to_deep",
        "runtime.claim_citation_failed",
        "artifact.approved",
    ]
    assert events[1].payload["added_count"] == 1
    assert events[2].payload["reason"] == "insufficient_evidence"
    assert events[4].payload["artifact_type"] == "topic_page"


def test_telemetry_service_records_retrieval_effectiveness_and_preservation_events() -> None:
    service = TelemetryService.create_in_memory()

    service.record_rrf_fusion(
        branch_count=3,
        candidate_count=7,
        fused_count=4,
        duplicate_count=3,
    )
    service.record_rerank_effectiveness(
        input_count=4,
        output_count=4,
        reordered=True,
        top1_changed=True,
    )
    service.record_preservation_suggestion(
        artifact_type="comparison_page",
        runtime_mode="deep",
        evidence_count=4,
        conflict_count=1,
    )

    events = service.list_events()

    assert [event.name for event in events] == [
        "retrieval.rrf_fused",
        "retrieval.rerank_effectiveness",
        "artifact.preservation_suggested",
    ]
    assert events[0].payload["duplicate_count"] == 3
    assert events[1].payload["top1_changed"] is True
    assert events[2].payload["artifact_type"] == "comparison_page"


def test_telemetry_service_records_local_fallback_event() -> None:
    service = TelemetryService.create_in_memory()

    service.record_local_fallback(
        from_location="cloud",
        to_location="local",
        failed_provider_count=2,
    )

    events = service.list_events()

    assert len(events) == 1
    assert events[0].name == "runtime.local_fallback"
    assert events[0].payload["failed_provider_count"] == 2
