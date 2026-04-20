from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.analysis import RoutingDecision
from rag.retrieval.evidence import EvidenceBundle, SelfCheckResult
from rag.retrieval.runtime_coordinator import CoreRetrievalPayload, RuntimeCoordinator, inflate_legacy_retrieval_result
from rag.schema.query import ComplexityLevel, TaskType
from rag.schema.runtime import RuntimeMode


@dataclass(frozen=True)
class _FakeCandidate:
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str = "internal"
    source_id: str | None = None
    section_path: tuple[str, ...] = ()
    chunk_role: object | None = None
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    benchmark_doc_id: str | None = None


async def _identity(value: int) -> int:
    return value


def test_runtime_coordinator_runs_async_pipeline_synchronously() -> None:
    coordinator = RuntimeCoordinator()
    assert coordinator.run_sync(_identity(7)) == 7


def test_runtime_coordinator_inflates_legacy_retrieval_result_with_safe_defaults() -> None:
    payload = CoreRetrievalPayload(
        decision=RoutingDecision(
            task_type=TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L1_DIRECT,
            runtime_mode=RuntimeMode.FAST,
            source_scope=[],
            web_search_allowed=False,
            graph_expansion_allowed=False,
            rerank_required=True,
        ),
        evidence=EvidenceBundle(),
        self_check=SelfCheckResult(retrieve_more=False, evidence_sufficient=False, claim_supported=False),
        clean_items=[_FakeCandidate("chunk-a", "doc-a", "alpha", "#a", 0.8, 1)],
        reranked_benchmark_doc_ids=[],
    )

    result = inflate_legacy_retrieval_result(payload)

    assert result.graph_expanded is False
    assert result.reranked_chunk_ids == ["chunk-a"]
    assert result.diagnostics.branch_hits == {}
    assert result.diagnostics.branch_limits == {}
    assert result.diagnostics.parent_backfilled_count == 0
    assert result.diagnostics.collapsed_candidate_count == 0
