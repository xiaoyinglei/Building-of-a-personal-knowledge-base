from __future__ import annotations

from collections.abc import Callable, Sequence
from time import monotonic
from typing import Protocol, cast

from pkp.interfaces._runtime.session_runtime import SessionRuntime
from pkp.schema._types import (
    AccessPolicy,
    EvidenceItem,
    ExecutionLocation,
    ExecutionLocationPreference,
    ExecutionPolicy,
    ModelDiagnostics,
    QueryDiagnostics,
    QueryResponse,
    RetrievalDiagnostics,
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
    def decompose(self, query: str, memory_hints: Sequence[str] = ()) -> list[str]: ...

    def expand(
        self,
        query: str,
        evidence_matrix: list[dict[str, object]],
        round_index: int,
        memory_hints: Sequence[str] = (),
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


class MemoryServiceProtocol(Protocol):
    def recall(self, query: str, source_scope: list[str]) -> list[str]: ...

    def record_episode(
        self,
        *,
        session_id: str,
        query: str,
        response: QueryResponse,
        evidence_matrix: list[dict[str, object]],
        source_scope: list[str],
    ) -> str: ...


class DeepResearchRuntime:
    def __init__(
        self,
        *,
        routing_service: RoutingServiceProtocol,
        retrieval_service: RetrievalServiceProtocol,
        evidence_service: EvidenceServiceProtocol,
        session_runtime: SessionRuntime,
        memory_service: MemoryServiceProtocol | None = None,
        max_rounds: int = 4,
        max_recursive_depth: int = 2,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._routing_service = routing_service
        self._retrieval_service = retrieval_service
        self._evidence_service = evidence_service
        self.session_runtime = session_runtime
        self._memory_service = memory_service
        self._max_rounds = max_rounds
        self._max_recursive_depth = max_recursive_depth
        self._clock = clock or monotonic

    def run(
        self,
        query: str,
        policy: ExecutionPolicy,
        *,
        session_id: str = "default",
    ) -> QueryResponse:
        started_at = self._clock()
        source_scope = list(policy.source_scope)
        memory_hints = self._memory_service.recall(query, source_scope) if self._memory_service is not None else []
        self.session_runtime.store_memory_hints(session_id, memory_hints)
        sub_questions = self._decompose(query, memory_hints)
        self.session_runtime.store_sub_questions(session_id, sub_questions)

        current_queries = sub_questions or [query]
        final_hits: list[EvidenceItem] = []
        evidence_matrix: list[dict[str, object]] = []
        retrieval_diagnostics = RetrievalDiagnostics()
        consumed_tokens = 0
        recursive_depth = 0
        for round_index in range(1, self._max_rounds + 1):
            if self._wall_clock_budget_exhausted(policy, started_at):
                break
            if self._token_budget_exhausted(policy, consumed_tokens):
                break

            round_hits: list[EvidenceItem] = []
            for item in current_queries:
                hits = self._retrieval_service.retrieve(item, policy, RuntimeMode.DEEP, round_index)
                round_hits.extend(hits)
                retrieval_diagnostics = self._merge_retrieval_diagnostics(
                    retrieval_diagnostics,
                    self._current_retrieval_diagnostics(),
                )
                consumed_tokens += self._estimate_token_cost(hits)
                if self._token_budget_exhausted(policy, consumed_tokens):
                    break
            final_hits = self._merge_hits(final_hits, round_hits)
            evidence_matrix = self._evidence_service.build_evidence_matrix(final_hits)
            self.session_runtime.store_evidence_matrix(session_id, evidence_matrix)
            if self._evidence_service.evidence_sufficient(evidence_matrix, round_index):
                break
            if self._wall_clock_budget_exhausted(policy, started_at):
                break
            if self._token_budget_exhausted(policy, consumed_tokens):
                break
            if recursive_depth >= self._max_recursive_depth:
                break

            current_queries = self._expand(query, evidence_matrix, round_index, memory_hints)
            if not current_queries:
                break
            recursive_depth += 1

        if not final_hits or not evidence_matrix:
            response = self._evidence_service.build_retrieval_only_response(query, final_hits)
            return self._attach_retrieval_diagnostics(response, retrieval_diagnostics)

        deep_response: QueryResponse | None = None
        for location in resolve_execution_locations(
            policy.effective_access_policy,
            policy.execution_location_preference,
        ):
            try:
                deep_response = self._build_deep_response(query, evidence_matrix, location.value)
                break
            except RuntimeError:
                continue

        if deep_response is None:
            response = self._evidence_service.build_retrieval_only_response(query, final_hits)
            return self._attach_retrieval_diagnostics(response, retrieval_diagnostics)

        if self._memory_service is not None:
            episode_id = self._memory_service.record_episode(
                session_id=session_id,
                query=query,
                response=deep_response,
                evidence_matrix=evidence_matrix,
                source_scope=source_scope,
            )
            self.session_runtime.store_episode_id(session_id, episode_id)
        return self._attach_retrieval_diagnostics(deep_response, retrieval_diagnostics)

    @staticmethod
    def _merge_hits(existing: list[EvidenceItem], new_hits: list[EvidenceItem]) -> list[EvidenceItem]:
        merged: list[EvidenceItem] = list(existing)
        seen = {item.chunk_id for item in merged}
        for hit in new_hits:
            if hit.chunk_id in seen:
                continue
            merged.append(hit)
            seen.add(hit.chunk_id)
        return merged

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
            fallback = cast(
                Callable[[str, list[dict[str, object]]], QueryResponse],
                self._evidence_service.build_deep_response,
            )
            return fallback(query, evidence_matrix)

    def _current_retrieval_diagnostics(self) -> RetrievalDiagnostics:
        result = getattr(self._retrieval_service, "last_result", None)
        diagnostics = getattr(result, "diagnostics", None)
        return diagnostics if isinstance(diagnostics, RetrievalDiagnostics) else RetrievalDiagnostics()

    @staticmethod
    def _merge_retrieval_diagnostics(
        existing: RetrievalDiagnostics,
        current: RetrievalDiagnostics,
    ) -> RetrievalDiagnostics:
        branch_hits = dict(existing.branch_hits)
        for branch, count in current.branch_hits.items():
            branch_hits[branch] = branch_hits.get(branch, 0) + count
        embedding_provider = current.embedding_provider or existing.embedding_provider
        rerank_provider = current.rerank_provider or existing.rerank_provider
        attempts = [*existing.attempts, *current.attempts]
        reranked_chunk_ids = list(existing.reranked_chunk_ids)
        for chunk_id in current.reranked_chunk_ids:
            if chunk_id not in reranked_chunk_ids:
                reranked_chunk_ids.append(chunk_id)
        return RetrievalDiagnostics(
            branch_hits=branch_hits,
            reranked_chunk_ids=reranked_chunk_ids,
            embedding_provider=embedding_provider,
            rerank_provider=rerank_provider,
            attempts=attempts,
            fusion_input_count=existing.fusion_input_count + current.fusion_input_count,
            fused_count=max(existing.fused_count, current.fused_count),
            graph_expanded=existing.graph_expanded or current.graph_expanded,
        )

    @staticmethod
    def _attach_retrieval_diagnostics(
        response: QueryResponse,
        retrieval_diagnostics: RetrievalDiagnostics,
    ) -> QueryResponse:
        return response.model_copy(
            update={
                "diagnostics": QueryDiagnostics(
                    retrieval=retrieval_diagnostics,
                    model=response.diagnostics.model
                    if isinstance(response.diagnostics.model, ModelDiagnostics)
                    else ModelDiagnostics(),
                )
            }
        )

    def _decompose(self, query: str, memory_hints: list[str]) -> list[str]:
        try:
            return self._routing_service.decompose(query, memory_hints)
        except TypeError:
            return self._routing_service.decompose(query)

    def _expand(
        self,
        query: str,
        evidence_matrix: list[dict[str, object]],
        round_index: int,
        memory_hints: list[str],
    ) -> list[str]:
        try:
            return self._routing_service.expand(query, evidence_matrix, round_index, memory_hints)
        except TypeError:
            return self._routing_service.expand(query, evidence_matrix, round_index)

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
