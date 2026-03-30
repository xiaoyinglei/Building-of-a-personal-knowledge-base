from __future__ import annotations

from typing import Protocol

from pkp.utils._telemetry import TelemetryService
from pkp.schema._types import (
    EvidenceItem,
    ExecutionPolicy,
    ModelDiagnostics,
    PreservationSuggestion,
    QueryDiagnostics,
    QueryResponse,
    RuntimeMode,
)


class RetrievalServiceProtocol(Protocol):
    def retrieve(
        self,
        query: str,
        policy: ExecutionPolicy,
        mode: RuntimeMode,
        round_index: int,
    ) -> list[EvidenceItem]: ...

    def rerank(
        self,
        query: str,
        hits: list[EvidenceItem],
        policy: ExecutionPolicy,
    ) -> list[EvidenceItem]: ...


class EvidenceServiceProtocol(Protocol):
    def fast_path_sufficient(self, hits: list[EvidenceItem], policy: ExecutionPolicy) -> bool: ...

    def detect_conflicts(self, hits: list[EvidenceItem]) -> list[str]: ...

    def build_fast_response(self, query: str, hits: list[EvidenceItem]) -> QueryResponse: ...

    def claim_citation_aligned(self, response: QueryResponse) -> bool: ...


class DeepRuntimeProtocol(Protocol):
    def run(self, query: str, policy: ExecutionPolicy) -> QueryResponse: ...


class FastQueryRuntime:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalServiceProtocol,
        evidence_service: EvidenceServiceProtocol,
        deep_runtime: DeepRuntimeProtocol,
        telemetry_service: TelemetryService | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._evidence_service = evidence_service
        self._deep_runtime = deep_runtime
        self._telemetry_service = telemetry_service

    def run(self, query: str, policy: ExecutionPolicy) -> QueryResponse:
        hits = self._retrieval_service.retrieve(query, policy, RuntimeMode.FAST, round_index=1)
        retrieval_diagnostics = getattr(getattr(self._retrieval_service, "last_result", None), "diagnostics", None)
        reranked = self._retrieval_service.rerank(query, hits, policy)
        if not reranked:
            return QueryResponse(
                conclusion="Insufficient evidence in indexed sources.",
                evidence=[],
                differences_or_conflicts=[],
                uncertainty="high",
                preservation_suggestion=PreservationSuggestion(suggested=False),
                runtime_mode=RuntimeMode.FAST,
                diagnostics=QueryDiagnostics(
                    retrieval=retrieval_diagnostics or QueryDiagnostics().retrieval,
                    model=ModelDiagnostics(
                        degraded_to_retrieval_only=True,
                        failed_stage="retrieval",
                    ),
                ),
            )
        conflicts = self._evidence_service.detect_conflicts(reranked)
        sufficient = self._evidence_service.fast_path_sufficient(reranked, policy)
        if conflicts or not sufficient:
            if self._telemetry_service is not None:
                self._telemetry_service.record_fast_to_deep_escalation(
                    reason="conflict" if conflicts else "insufficient_evidence"
                )
            return self._deep_runtime.run(query, policy)

        response = self._evidence_service.build_fast_response(query, reranked)
        response = response.model_copy(
            update={
                "diagnostics": QueryDiagnostics(
                    retrieval=retrieval_diagnostics or QueryDiagnostics().retrieval,
                    model=response.diagnostics.model,
                )
            }
        )
        if not self._evidence_service.claim_citation_aligned(response):
            if self._telemetry_service is not None:
                self._telemetry_service.record_claim_citation_failure(
                    response_mode=response.runtime_mode.value,
                    evidence_count=len(response.evidence),
                )
                self._telemetry_service.record_fast_to_deep_escalation(reason="claim_citation_failure")
            return self._deep_runtime.run(query, policy)
        return response
