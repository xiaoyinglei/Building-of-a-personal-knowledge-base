from __future__ import annotations

from pkp.eval.ragas_eval import (
    ChunkReference,
    EmbeddingCompatibilityAdapter,
    enrich_generated_samples,
    infer_question_type,
    select_generation_chunks,
    select_low_score_rows,
    summarize_metric_rows,
)


def _chunk(
    *,
    manifest_doc_id: str,
    runtime_doc_id: str,
    chunk_id: str,
    text: str,
    citation_anchor: str,
    section_path: list[str] | None = None,
) -> ChunkReference:
    return ChunkReference(
        manifest_doc_id=manifest_doc_id,
        runtime_doc_id=runtime_doc_id,
        location=f"data/test_corpus/{manifest_doc_id}.md",
        file_name=f"{manifest_doc_id}.md",
        source_type="markdown",
        chunk_id=chunk_id,
        text=text,
        citation_anchor=citation_anchor,
        section_path=section_path or [],
        chunk_role="child",
        special_chunk_type=None,
        order_index=1,
    )


def test_infer_question_type_uses_simple_heuristics() -> None:
    assert infer_question_type("Summarize the architecture tradeoffs.") == "summary"
    assert infer_question_type("Which section explains installation?") == "section_lookup"
    assert infer_question_type("What does Table 2 report for recall?") == "table_question"
    assert infer_question_type("What date appears in the scanned letter?") == "ocr_question"
    assert infer_question_type("What library is recommended in the quickstart?") == "factual"


def test_enrich_generated_samples_prefers_reference_ids_and_maps_to_manifest_doc() -> None:
    chunks = [
        _chunk(
            manifest_doc_id="doc_alpha",
            runtime_doc_id="runtime-alpha",
            chunk_id="chunk-a",
            text="Alpha systems prioritize reliability over stylistic fluency.",
            citation_anchor="alpha/reliability",
            section_path=["Alpha", "Reliability"],
        ),
        _chunk(
            manifest_doc_id="doc_beta",
            runtime_doc_id="runtime-beta",
            chunk_id="chunk-b",
            text="Beta systems emphasize experimentation speed.",
            citation_anchor="beta/speed",
            section_path=["Beta", "Speed"],
        ),
    ]

    rows = [
        {
            "user_input": "What do Alpha systems prioritize?",
            "reference": "They prioritize reliability over stylistic fluency.",
            "reference_contexts": ["Alpha systems prioritize reliability over stylistic fluency."],
            "reference_context_ids": ["chunk-a"],
            "query_style": "formal",
            "query_length": "short",
        }
    ]

    enriched = enrich_generated_samples(rows, chunks)

    assert len(enriched) == 1
    sample = enriched[0]
    assert sample.question_id == "synthetic_001"
    assert sample.expected_doc_id == "doc_alpha"
    assert sample.expected_runtime_doc_id == "runtime-alpha"
    assert sample.reference_context_ids == ["chunk-a"]
    assert sample.expected_section_hint == "Alpha > Reliability"
    assert sample.question_type == "factual"


def test_enrich_generated_samples_falls_back_to_reference_context_text_matching() -> None:
    chunks = [
        _chunk(
            manifest_doc_id="doc_alpha",
            runtime_doc_id="runtime-alpha",
            chunk_id="chunk-a",
            text="Alpha systems prioritize reliability over stylistic fluency.",
            citation_anchor="alpha/reliability",
            section_path=["Alpha", "Reliability"],
        ),
        _chunk(
            manifest_doc_id="doc_alpha",
            runtime_doc_id="runtime-alpha",
            chunk_id="chunk-a2",
            text="Alpha rollout requires approval from the operations lead.",
            citation_anchor="alpha/operations",
            section_path=["Alpha", "Operations"],
        ),
    ]

    rows = [
        {
            "user_input": "Summarize Alpha rollout requirements.",
            "reference": "Alpha requires approval from the operations lead.",
            "reference_contexts": [" Alpha rollout requires approval from the operations lead. "],
        }
    ]

    enriched = enrich_generated_samples(rows, chunks)

    assert len(enriched) == 1
    sample = enriched[0]
    assert sample.reference_context_ids == ["chunk-a2"]
    assert sample.expected_doc_id == "doc_alpha"
    assert sample.expected_section_hint == "Alpha > Operations"
    assert sample.question_type == "summary"


def test_summarize_metric_rows_computes_metric_averages_and_overall_score() -> None:
    rows = [
        {
            "question_id": "q1",
            "answer_relevancy": 0.9,
            "faithfulness": 0.6,
            "id_based_context_precision": 1.0,
            "id_based_context_recall": 0.5,
        },
        {
            "question_id": "q2",
            "answer_relevancy": 0.3,
            "faithfulness": None,
            "id_based_context_precision": 0.0,
            "id_based_context_recall": 0.5,
        },
    ]

    summary = summarize_metric_rows(
        rows,
        metric_names=[
            "answer_relevancy",
            "faithfulness",
            "id_based_context_precision",
            "id_based_context_recall",
        ],
    )

    assert summary["sample_count"] == 2
    assert summary["answer_relevancy"] == 0.6
    assert summary["faithfulness"] == 0.6
    assert summary["id_based_context_precision"] == 0.5
    assert summary["id_based_context_recall"] == 0.5
    assert summary["overall_score"] == 0.5083


def test_select_low_score_rows_flags_any_metric_or_overall_below_threshold() -> None:
    rows = [
        {
            "question_id": "q1",
            "answer_relevancy": 0.9,
            "faithfulness": 0.8,
            "id_based_context_precision": 1.0,
        },
        {
            "question_id": "q2",
            "answer_relevancy": 0.7,
            "faithfulness": 0.3,
            "id_based_context_precision": 0.9,
        },
        {
            "question_id": "q3",
            "answer_relevancy": 0.2,
            "faithfulness": 0.1,
            "id_based_context_precision": 0.0,
        },
    ]

    low_score_rows = select_low_score_rows(
        rows,
        metric_names=["answer_relevancy", "faithfulness", "id_based_context_precision"],
        threshold=0.5,
    )

    assert [row["question_id"] for row in low_score_rows] == ["q2", "q3"]
    assert low_score_rows[0]["low_score_reasons"] == ["faithfulness"]
    assert low_score_rows[1]["low_score_reasons"] == [
        "answer_relevancy",
        "faithfulness",
        "id_based_context_precision",
        "overall_score",
    ]


def test_select_generation_chunks_downsamples_dense_docs_but_keeps_structured_special_chunks() -> None:
    chunks = [
        *[
            _chunk(
                manifest_doc_id="doc_alpha",
                runtime_doc_id="runtime-alpha",
                chunk_id=f"child-{index}",
                text=f"Child chunk {index}",
                citation_anchor=f"alpha/child-{index}",
            )
            for index in range(6)
        ],
        ChunkReference(
            manifest_doc_id="doc_alpha",
            runtime_doc_id="runtime-alpha",
            location="data/test_corpus/doc_alpha.pdf",
            file_name="doc_alpha.pdf",
            source_type="pdf",
            chunk_id="table-1",
            text="Important table",
            citation_anchor="alpha/table-1",
            section_path=["Alpha", "Tables"],
            chunk_role="special",
            special_chunk_type="table",
            order_index=100,
        ),
        ChunkReference(
            manifest_doc_id="doc_alpha",
            runtime_doc_id="runtime-alpha",
            location="data/test_corpus/doc_alpha.pdf",
            file_name="doc_alpha.pdf",
            source_type="pdf",
            chunk_id="ocr-1",
            text="OCR region 1",
            citation_anchor="alpha/ocr-1",
            section_path=["Alpha", "OCR"],
            chunk_role="special",
            special_chunk_type="ocr_region",
            order_index=101,
        ),
        ChunkReference(
            manifest_doc_id="doc_alpha",
            runtime_doc_id="runtime-alpha",
            location="data/test_corpus/doc_alpha.pdf",
            file_name="doc_alpha.pdf",
            source_type="pdf",
            chunk_id="ocr-2",
            text="OCR region 2",
            citation_anchor="alpha/ocr-2",
            section_path=["Alpha", "OCR"],
            chunk_role="special",
            special_chunk_type="ocr_region",
            order_index=102,
        ),
    ]

    selected = select_generation_chunks(
        chunks,
        max_child_chunks_per_doc=2,
        max_special_chunks_per_doc=1,
        max_ocr_region_chunks_per_doc=1,
    )

    assert [chunk.chunk_id for chunk in selected] == ["child-0", "child-5", "table-1", "ocr-1"]


def test_embedding_compatibility_adapter_exposes_langchain_style_methods() -> None:
    class _FakeEmbedding:
        def embed_text(self, text: str, **_: object) -> list[float]:
            return [float(len(text))]

        def embed_texts(self, texts: list[str], **_: object) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

    adapter = EmbeddingCompatibilityAdapter(_FakeEmbedding())

    assert adapter.embed_query("alpha") == [5.0]
    assert adapter.embed_documents(["a", "beta"]) == [[1.0], [4.0]]
