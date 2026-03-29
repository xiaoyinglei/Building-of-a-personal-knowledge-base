from __future__ import annotations

from pkp.query.context import ContextEvidenceMerger, EvidenceBundle, EvidenceTruncator, RoutingDecision, SelfCheckResult
from pkp.types.access import RuntimeMode
from pkp.types.content import ChunkRole
from pkp.types.envelope import EvidenceItem, PreservationSuggestion
from pkp.types.query import ComplexityLevel, TaskType
from pkp.types.retrieval import RetrievalResult


def _item(
    chunk_id: str,
    *,
    doc_id: str = "doc-1",
    text: str,
    score: float = 1.0,
    evidence_kind: str = "internal",
    special_chunk_type: str | None = None,
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
