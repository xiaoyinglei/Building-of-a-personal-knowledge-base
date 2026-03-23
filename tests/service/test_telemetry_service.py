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
