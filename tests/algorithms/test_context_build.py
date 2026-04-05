from __future__ import annotations

from rag.query.context import ContextEvidenceMerger, EvidenceBundle, EvidenceTruncator, SelfCheckResult
from rag.query.query import QueryMode
from rag.query.routing import RoutingDecision
from rag.schema._types.access import RuntimeMode
from rag.schema._types.content import ChunkRole
from rag.schema._types.envelope import EvidenceItem, PreservationSuggestion
from rag.schema._types.query import ComplexityLevel, TaskType
from rag.schema._types.retrieval import RetrievalResult


def _item(
    chunk_id: str,
    *,
    doc_id: str = "doc-1",
    text: str,
    score: float = 1.0,
    evidence_kind: str = "internal",
    special_chunk_type: str | None = None,
    retrieval_channels: list[str] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_id=doc_id,
        citation_anchor=chunk_id,
        text=text,
        score=score,
        evidence_kind=evidence_kind,
        chunk_role=ChunkRole.SPECIAL if special_chunk_type else ChunkRole.CHILD,
        special_chunk_type=special_chunk_type,
        parent_chunk_id=None,
        file_name=f"{doc_id}.md",
        section_path=["Section"],
        page_start=None,
        page_end=None,
        chunk_type=special_chunk_type or "child",
        source_type="markdown",
        retrieval_channels=list(retrieval_channels or []),
    )


def _retrieval_result(*, internal: list[EvidenceItem], graph: list[EvidenceItem]) -> RetrievalResult:
    return RetrievalResult(
        decision=RoutingDecision(
            task_type=TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L1_DIRECT,
            runtime_mode=RuntimeMode.FAST,
        ),
        evidence=EvidenceBundle(internal=internal, graph=graph, external=[]),
        self_check=SelfCheckResult(
            retrieve_more=False,
            evidence_sufficient=True,
            claim_supported=True,
        ),
        reranked_chunk_ids=[item.chunk_id for item in internal],
        graph_expanded=bool(graph),
        preservation_suggestion=PreservationSuggestion(suggested=False),
    )


def test_context_evidence_merger_merges_graph_duplicates_into_internal_chunk() -> None:
    merger = ContextEvidenceMerger()
    retrieval = _retrieval_result(
        internal=[
            _item(
                "chunk-1",
                text="Alpha Engine coordinates the retrieval pipeline.",
                score=0.82,
            )
        ],
        graph=[
            _item(
                "chunk-1",
                text="Alpha Engine coordinates the retrieval pipeline and links entities.",
                score=1.15,
                evidence_kind="graph",
            )
        ],
    )

    merged = merger.merge(retrieval)

    assert len(merged) == 1
    assert merged[0].chunk_id == "chunk-1"
    assert merged[0].evidence_kind == "internal"
    assert merged[0].score == 1.15
    assert "links entities" in merged[0].text


def test_evidence_truncator_keeps_special_and_graph_evidence_under_tight_budget() -> None:
    truncator = EvidenceTruncator()
    evidence = [
        _item(
            "chunk-internal-1",
            doc_id="doc-1",
            text=" ".join(["Alpha engine pipeline"] * 140),
            score=1.30,
        ),
        _item(
            "chunk-special-table",
            doc_id="doc-1",
            text=" ".join(["table metric value"] * 28),
            score=1.05,
            special_chunk_type="table",
        ),
        _item(
            "chunk-graph-1",
            doc_id="doc-2",
            text=" ".join(["entity relation support"] * 24),
            score=0.98,
            evidence_kind="graph",
        ),
        _item(
            "chunk-internal-2",
            doc_id="doc-2",
            text=" ".join(["Beta service downstream dependency"] * 100),
            score=0.94,
        ),
    ]

    result = truncator.truncate(
        evidence,
        token_budget=120,
        max_evidence_chunks=3,
    )

    selected_ids = {item.chunk_id for item in result.evidence}
    assert "chunk-special-table" in selected_ids
    assert "chunk-graph-1" in selected_ids
    assert result.token_count <= 120
    assert len(result.evidence) == 3
    assert any(item.truncated for item in result.evidence)


def test_evidence_truncator_local_mode_prioritizes_kg_and_multimodal_over_vector_fallback() -> None:
    truncator = EvidenceTruncator()
    evidence = [
        _item(
            "chunk-local-1",
            text=" ".join(["Alpha entity context"] * 50),
            score=1.45,
            retrieval_channels=["local"],
        ),
        _item(
            "chunk-local-2",
            text=" ".join(["Beta entity context"] * 44),
            score=1.32,
            retrieval_channels=["local"],
        ),
        _item(
            "chunk-vector-1",
            text=" ".join(["fallback semantic match"] * 48),
            score=1.60,
            retrieval_channels=["vector"],
        ),
        _item(
            "chunk-table-1",
            text=" ".join(["table metric value"] * 22),
            score=0.62,
            special_chunk_type="table",
            retrieval_channels=["special"],
        ),
    ]

    result = truncator.truncate(
        evidence,
        token_budget=96,
        max_evidence_chunks=3,
        mode=QueryMode.LOCAL,
    )

    selected_ids = {item.chunk_id for item in result.evidence}
    selected_families = {item.retrieval_family for item in result.evidence}
    assert selected_ids == {"chunk-local-1", "chunk-local-2", "chunk-table-1"}
    assert selected_families == {"kg", "multimodal"}


def test_evidence_truncator_mix_mode_preserves_kg_vector_and_multimodal_families_under_tight_budget() -> None:
    truncator = EvidenceTruncator()
    evidence = [
        _item(
            "chunk-local-1",
            text=" ".join(["Alpha entity relation"] * 44),
            score=1.34,
            retrieval_channels=["local"],
        ),
        _item(
            "chunk-global-1",
            text=" ".join(["dependency relation evidence"] * 40),
            score=1.28,
            retrieval_channels=["global"],
        ),
        _item(
            "chunk-vector-1",
            text=" ".join(["semantic overview paragraph"] * 42),
            score=1.38,
            retrieval_channels=["vector"],
        ),
        _item(
            "chunk-sparse-1",
            text=" ".join(["keyword hit paragraph"] * 38),
            score=1.18,
            retrieval_channels=["full_text"],
        ),
        _item(
            "chunk-table-1",
            text=" ".join(["table metric value"] * 20),
            score=0.72,
            special_chunk_type="table",
            retrieval_channels=["special"],
        ),
    ]

    result = truncator.truncate(
        evidence,
        token_budget=90,
        max_evidence_chunks=3,
        mode="mix",
    )

    selected_families = {item.retrieval_family for item in result.evidence}
    assert selected_families == {"kg", "vector", "multimodal"}
    assert result.token_count <= 90
