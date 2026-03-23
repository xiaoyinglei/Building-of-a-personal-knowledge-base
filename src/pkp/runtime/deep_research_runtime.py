from __future__ import annotations

from collections.abc import Callable
from time import monotonic
from typing import Protocol

from pkp.runtime.session_runtime import SessionRuntime
from pkp.types import (
    AccessPolicy,
    EvidenceItem,
    ExecutionLocation,
    ExecutionLocationPreference,
    ExecutionPolicy,
    QueryResponse,
    RuntimeMode,
)


def resolve_execution_locations(
    access_policy: AccessPolicy,
    preference: ExecutionLocationPreference,
) -> list[ExecutionLocation]:
    if access_policy.local_only or preference is ExecutionLocationPreference.LOCAL_ONLY:
        return [ExecutionLocation.LOCAL]

    preferred_order = (
        [ExecutionLocation.LOCAL, ExecutionLocation.CLOUD]
        if preference is ExecutionLocationPreference.LOCAL_FIRST
        else [ExecutionLocation.CLOUD, ExecutionLocation.LOCAL]
    )
    return [location for location in preferred_order if access_policy.allows_location(location)]


class RoutingServiceProtocol(Protocol):
    def decompose(self, query: str) -> list[str]: ...

    def expand(
        self,
        query: str,
        evidence_matrix: list[dict[str, object]],
        round_index: int,
    ) -> list[str]: ...


class RetrievalServiceProtocol(Protocol):
    def retrieve(
        self,
        query: str,
        policy: ExecutionPolicy,
        mode: RuntimeMode,
        round_index: int,
    ) -> list[EvidenceItem]: ...


class EvidenceServiceProtocol(Protocol):
    def build_evidence_matrix(self, hits: list[EvidenceItem]) -> list[dict[str, object]]: ...

    def evidence_sufficient(
        self,
        evidence_matrix: list[dict[str, object]],
        round_index: int,
    ) -> bool: ...

    def build_deep_response(
        self,
        query: str,
        evidence_matrix: list[dict[str, object]],
        *,
        location: str,
    ) -> QueryResponse: ...

    def build_retrieval_only_response(
        self,
        query: str,
        hits: list[EvidenceItem],
    ) -> QueryResponse: ...


class DeepResearchRuntime:
    def __init__(
        self,
        *,
        routing_service: RoutingServiceProtocol,
        retrieval_service: RetrievalServiceProtocol,
        evidence_service: EvidenceServiceProtocol,
        session_runtime: SessionRuntime,
        max_rounds: int = 4,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._routing_service = routing_service
        self._retrieval_service = retrieval_service
        self._evidence_service = evidence_service
        self.session_runtime = session_runtime
        self._max_rounds = max_rounds
        self._clock = clock or monotonic

    def run(
        self,
        query: str,
        policy: ExecutionPolicy,
        *,
        session_id: str = "default",
    ) -> QueryResponse:
        started_at = self._clock()
        sub_questions = self._routing_service.decompose(query)
        self.session_runtime.store_sub_questions(session_id, sub_questions)

        current_queries = sub_questions or [query]
        final_hits: list[EvidenceItem] = []
        evidence_matrix: list[dict[str, object]] = []
        consumed_tokens = 0
        for round_index in range(1, self._max_rounds + 1):
            if self._wall_clock_budget_exhausted(policy, started_at):
                break
            if self._token_budget_exhausted(policy, consumed_tokens):
                break

            round_hits: list[EvidenceItem] = []
            for item in current_queries:
                hits = self._retrieval_service.retrieve(item, policy, RuntimeMode.DEEP, round_index)
                round_hits.extend(hits)
                consumed_tokens += self._estimate_token_cost(hits)
                if self._token_budget_exhausted(policy, consumed_tokens):
                    break
            final_hits = round_hits
            evidence_matrix = self._evidence_service.build_evidence_matrix(round_hits)
            self.session_runtime.store_evidence_matrix(session_id, evidence_matrix)
            if self._evidence_service.evidence_sufficient(evidence_matrix, round_index):
                break
            if self._wall_clock_budget_exhausted(policy, started_at):
                break
            if self._token_budget_exhausted(policy, consumed_tokens):
                break

            current_queries = self._routing_service.expand(query, evidence_matrix, round_index)
            if not current_queries:
                break

        for location in resolve_execution_locations(
            policy.effective_access_policy,
            policy.execution_location_preference,
        ):
            try:
                return self._build_deep_response(query, evidence_matrix, location.value)
            except RuntimeError:
                continue

        return self._evidence_service.build_retrieval_only_response(query, final_hits)

    def _build_deep_response(
        self,
        query: str,
        evidence_matrix: list[dict[str, object]],
        location: str,
    ) -> QueryResponse:
        try:
            return self._evidence_service.build_deep_response(
                query,
                evidence_matrix,
                location=location,
            )
        except TypeError:
            return self._evidence_service.build_deep_response(query, evidence_matrix)  # type: ignore[call-arg]

    def _wall_clock_budget_exhausted(
        self,
        policy: ExecutionPolicy,
        started_at: float,
    ) -> bool:
        return policy.latency_budget >= 0 and (self._clock() - started_at) > policy.latency_budget

    @staticmethod
    def _token_budget_exhausted(
        policy: ExecutionPolicy,
        consumed_tokens: int,
    ) -> bool:
        return policy.token_budget is not None and consumed_tokens >= policy.token_budget

    @staticmethod
    def _estimate_token_cost(hits: list[EvidenceItem]) -> int:
        return sum(len(item.text.split()) for item in hits)
