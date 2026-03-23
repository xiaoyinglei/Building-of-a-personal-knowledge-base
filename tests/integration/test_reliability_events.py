from pkp.service.telemetry_service import TelemetryService
from pkp.types.telemetry import TelemetryEvent


def test_telemetry_service_groups_and_counts_events() -> None:
    service = TelemetryService.create_in_memory()
    service.record(TelemetryEvent(name="retrieval.branch_used", category="retrieval"))
    service.record(TelemetryEvent(name="retrieval.branch_used", category="retrieval"))
    service.record(TelemetryEvent(name="runtime.escalated_to_deep", category="runtime"))

    assert service.count_by_name("retrieval.branch_used") == 2
    assert service.count_by_category("runtime") == 1
