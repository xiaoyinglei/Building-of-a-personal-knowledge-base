from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rag.retrieval.evidence import EvidenceService
from rag.retrieval.models import QueryMode
from rag.retrieval.planning_graph import ComplexityGate, PlanningState, PredicatePlan, QueryVariant, RetrievalPath
from rag.retrieval.retrieval_adapter import RetrievalAdapter
from rag.schema.query import QueryUnderstanding
from rag.schema.runtime import AccessPolicy, RuntimeMode


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


class _AsyncVectorRetriever:
    absorbs_sparse_branch = True

    def __init__(self) -> None:
        self.calls = 0

    async def aretrieve_with_plan(self, *, query: str, source_scope: list[str], plan: PlanningState, **kwargs):
        del query, source_scope, plan, kwargs
        self.calls += 1
        return [_FakeCandidate("chunk-a", "doc-a", "alpha", "#a", 0.9, 1)]


class _FullTextRetriever:
    def __call__(self, query: str, source_scope: list[str], query_understanding: QueryUnderstanding):
        raise AssertionError("full_text branch should be skipped when vector hybrid absorbs sparse recall")


class _BranchRegistry:
    def __init__(self, *, vector_retriever: object, full_text_retriever: object) -> None:
        self._retrievers = {
            "vector": vector_retriever,
            "full_text": full_text_retriever,
        }

    def get(self, branch: str):
        return self._retrievers[branch]

    def collect_web(
        self,
        *,
        query: str,
        source_scope: list[str],
        query_understanding: QueryUnderstanding,
    ) -> list[_FakeCandidate]:
        del query, source_scope, query_understanding
        return []


def test_retrieval_adapter_awaits_async_plan_aware_vector_and_skips_full_text() -> None:
    vector_retriever = _AsyncVectorRetriever()
    adapter = RetrievalAdapter(
        branch_registry=_BranchRegistry(
            vector_retriever=vector_retriever,
            full_text_retriever=_FullTextRetriever(),
        ),
        evidence_service=EvidenceService(),
    )
    plan = PlanningState(
        original_query="alpha",
        rewritten_query="alpha",
        sparse_query="alpha",
        mode=QueryMode.MIX,
        mode_executor="mix",
        complexity_gate=ComplexityGate.STANDARD,
        semantic_route="text_first",
        target_collections=("section_summary",),
        predicate_plan=PredicatePlan(),
        retrieval_paths=(
            RetrievalPath("vector", 2, QueryVariant.DENSE),
            RetrievalPath("full_text", 2, QueryVariant.SPARSE),
        ),
        allow_web=False,
        allow_graph_expansion=False,
        web_limit=0,
        graph_limit=0,
    )

    result = asyncio.run(
        adapter.acollect_internal(
            plan=plan,
            source_scope=[],
            access_policy=AccessPolicy.default(),
            runtime_mode=RuntimeMode.FAST,
            query_understanding=QueryUnderstanding(),
        )
    )

    assert vector_retriever.calls == 1
    assert result.branch_hits == {"vector": 1, "full_text": 0}
    assert [branch for branch, _items in result.branches] == ["vector"]
