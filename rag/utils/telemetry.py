from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rag.schema.runtime import (
    EvaluationMetricInput,
    EvaluationMetricSummary,
    TelemetryEvent,
    coerce_evaluation_metric_input,
)


class LocalEventRepo:
    def __init__(self, sink_path: Path | None = None) -> None:
        self._sink_path = sink_path
        self._events: list[TelemetryEvent] = []

    def append(self, event: TelemetryEvent) -> None:
        self._events.append(event)
        if self._sink_path is None:
            return
        self._sink_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event.model_dump(mode="json"), ensure_ascii=True)
        with self._sink_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")

    def list_events(self) -> list[TelemetryEvent]:
        return list(self._events)


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

    def _record(self, *, name: str, category: str, payload: dict[str, str | int | float | bool]) -> None:
        self.record(TelemetryEvent(name=name, category=category, payload=payload))

    def record_branch_usage(self, *, branch: str, hit_count: int, runtime_mode: str) -> None:
        self._record(
            name="retrieval.branch_used",
            category="retrieval",
            payload={
                "branch": branch,
                "count": hit_count,
                "runtime_mode": runtime_mode,
            },
        )

    def record_rrf_fusion(
        self,
        *,
        branch_count: int,
        candidate_count: int,
        fused_count: int,
        duplicate_count: int,
    ) -> None:
        self._record(
            name="retrieval.rrf_fused",
            category="retrieval",
            payload={
                "branch_count": branch_count,
                "candidate_count": candidate_count,
                "fused_count": fused_count,
                "duplicate_count": duplicate_count,
            },
        )

    def record_rerank_effectiveness(
        self,
        *,
        input_count: int,
        output_count: int,
        reordered: bool,
        top1_changed: bool,
    ) -> None:
        self._record(
            name="retrieval.rerank_effectiveness",
            category="retrieval",
            payload={
                "input_count": input_count,
                "output_count": output_count,
                "reordered": reordered,
                "top1_changed": top1_changed,
            },
        )

    def record_graph_expansion(self, *, seed_count: int, added_count: int) -> None:
        self._record(
            name="retrieval.graph_expanded",
            category="retrieval",
            payload={
                "seed_count": seed_count,
                "added_count": added_count,
            },
        )

    def record_fast_to_deep_escalation(self, *, reason: str) -> None:
        self._record(
            name="runtime.escalated_to_deep",
            category="runtime",
            payload={"reason": reason},
        )

    def record_claim_citation_failure(self, *, response_mode: str, evidence_count: int) -> None:
        self._record(
            name="runtime.claim_citation_failed",
            category="runtime",
            payload={
                "response_mode": response_mode,
                "evidence_count": evidence_count,
            },
        )

    def record_artifact_approved(self, *, artifact_id: str, artifact_type: str) -> None:
        self._record(
            name="artifact.approved",
            category="artifact",
            payload={
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
            },
        )

    def record_local_fallback(
        self,
        *,
        from_location: str,
        to_location: str,
        failed_provider_count: int,
    ) -> None:
        self._record(
            name="runtime.local_fallback",
            category="runtime",
            payload={
                "from_location": from_location,
                "to_location": to_location,
                "failed_provider_count": failed_provider_count,
            },
        )

    def record_preservation_suggestion(
        self,
        *,
        artifact_type: str,
        runtime_mode: str,
        evidence_count: int,
        conflict_count: int,
    ) -> None:
        self._record(
            name="artifact.preservation_suggested",
            category="artifact",
            payload={
                "artifact_type": artifact_type,
                "runtime_mode": runtime_mode,
                "evidence_count": evidence_count,
                "conflict_count": conflict_count,
                "suggested": True,
            },
        )

    def list_events(self) -> list[TelemetryEvent]:
        return self._repo.list_events()

    def count_by_name(self, name: str) -> int:
        return sum(1 for event in self.list_events() if event.name == name)

    def count_by_category(self, category: str) -> int:
        return sum(1 for event in self.list_events() if event.category == category)


def summarize_evaluation_metrics(
    evaluations: Sequence[EvaluationMetricInput | Mapping[str, Any]],
) -> EvaluationMetricSummary:
    total = len(evaluations)
    if total == 0:
        return EvaluationMetricSummary(
            citation_precision=0.0,
            evidence_sufficiency_rate=0.0,
            conflict_detection_quality=0.0,
            simple_query_latency=0.0,
            deep_query_completion_quality=0.0,
            preservation_usefulness=0.0,
        )

    samples = [coerce_evaluation_metric_input(item) for item in evaluations]
    citation_precision = sum(sample.citation_precision for sample in samples) / total
    evidence_sufficiency_rate = sum(1.0 for sample in samples if sample.evidence_sufficient) / total
    conflict_detection_quality = sum(1.0 for sample in samples if sample.conflict_detected) / total
    simple_query_latency = sum(sample.simple_query_latency_seconds for sample in samples) / total
    deep_query_completion_quality = sum(sample.deep_query_completion_quality for sample in samples) / total
    preservation_usefulness = sum(1.0 for sample in samples if sample.preservation_useful) / total
    return EvaluationMetricSummary(
        citation_precision=round(citation_precision, 4),
        evidence_sufficiency_rate=round(evidence_sufficiency_rate, 4),
        conflict_detection_quality=round(conflict_detection_quality, 4),
        simple_query_latency=round(simple_query_latency, 4),
        deep_query_completion_quality=round(deep_query_completion_quality, 4),
        preservation_usefulness=round(preservation_usefulness, 4),
    )


def compute_evaluation_metrics(
    evaluations: Sequence[EvaluationMetricInput | Mapping[str, Any]],
) -> dict[str, float]:
    return summarize_evaluation_metrics(evaluations).as_dict()
