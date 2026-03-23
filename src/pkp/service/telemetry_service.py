from __future__ import annotations

from pathlib import Path

from pkp.repo.telemetry.local_event_repo import LocalEventRepo
from pkp.types.telemetry import TelemetryEvent


class TelemetryService:
    def __init__(self, repo: LocalEventRepo) -> None:
        self._repo = repo

    @classmethod
    def create_in_memory(cls) -> TelemetryService:
        return cls(LocalEventRepo())

    @classmethod
    def create_jsonl(cls, path: Path) -> TelemetryService:
        return cls(LocalEventRepo(path))

    def record(self, event: TelemetryEvent) -> None:
        self._repo.append(event)

    def list_events(self) -> list[TelemetryEvent]:
        return self._repo.list_events()

    def count_by_name(self, name: str) -> int:
        return sum(1 for event in self.list_events() if event.name == name)

    def count_by_category(self, category: str) -> int:
        return sum(1 for event in self.list_events() if event.category == category)


def compute_evaluation_metrics(
    evaluations: list[dict[str, float | bool]],
) -> dict[str, float]:
    total = len(evaluations)
    if total == 0:
        return {
            "citation_precision": 0.0,
            "evidence_sufficiency_rate": 0.0,
            "conflict_detection_quality": 0.0,
            "simple_query_latency": 0.0,
            "deep_query_completion_quality": 0.0,
            "preservation_usefulness": 0.0,
        }

    citation_precision = sum(float(item["citation_precision"]) for item in evaluations) / total
    evidence_sufficiency_rate = sum(1.0 for item in evaluations if bool(item["evidence_sufficient"])) / total
    conflict_detection_quality = sum(1.0 for item in evaluations if bool(item["conflict_detected"])) / total
    simple_query_latency = sum(float(item["latency_seconds"]) for item in evaluations) / total
    deep_query_completion_quality = sum(float(item["deep_quality"]) for item in evaluations) / total
    preservation_usefulness = sum(1.0 for item in evaluations if bool(item["preservation_useful"])) / total
    return {
        "citation_precision": round(citation_precision, 4),
        "evidence_sufficiency_rate": round(evidence_sufficiency_rate, 4),
        "conflict_detection_quality": round(conflict_detection_quality, 4),
        "simple_query_latency": round(simple_query_latency, 4),
        "deep_query_completion_quality": round(deep_query_completion_quality, 4),
        "preservation_usefulness": round(preservation_usefulness, 4),
    }
