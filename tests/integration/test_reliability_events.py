from pkp.service.telemetry_service import TelemetryService
from pkp.types.telemetry import TelemetryEvent


def test_telemetry_service_groups_and_counts_events() -> None:
    service = TelemetryService.create_in_memory()
    service.record(TelemetryEvent(name="retrieval.branch_used", category="retrieval"))
    service.record(TelemetryEvent(name="retrieval.rrf_fused", category="retrieval"))
    service.record(TelemetryEvent(name="retrieval.rerank_effectiveness", category="retrieval"))
    service.record(TelemetryEvent(name="runtime.local_fallback", category="runtime"))
    service.record(TelemetryEvent(name="artifact.preservation_suggested", category="artifact"))
    service.record(TelemetryEvent(name="runtime.escalated_to_deep", category="runtime"))

    assert service.count_by_name("retrieval.branch_used") == 1
    assert service.count_by_name("retrieval.rrf_fused") == 1
    assert service.count_by_name("artifact.preservation_suggested") == 1
    assert service.count_by_category("retrieval") == 3
    assert service.count_by_category("runtime") == 2
