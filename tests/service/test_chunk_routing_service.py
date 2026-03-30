from rag.ingest.chunk import ChunkRoutingService
from rag.schema._types.content import SourceType
from rag.schema._types.processing import ChunkingStrategy, DocumentFeatures


def test_pdf_defaults_to_hybrid_chunker() -> None:
    service = ChunkRoutingService()
    features = DocumentFeatures(
        source_type=SourceType.PDF,
        section_count=6,
        word_count=1800,
        heading_count=3,
        heading_quality_score=0.35,
        table_count=1,
        figure_count=0,
        caption_count=0,
        ocr_region_count=0,
        avg_section_words=300.0,
        structure_depth=2,
        has_dense_structure=False,
        metadata={},
    )

    decision = service.route(features)

    assert decision.selected_strategy is ChunkingStrategy.HYBRID
    assert decision.special_chunk_mode is True
    assert decision.local_refine is True
    assert decision.fallback is False


def test_markdown_prefers_hierarchical_chunker() -> None:
    service = ChunkRoutingService()
    features = DocumentFeatures(
        source_type=SourceType.MARKDOWN,
        section_count=9,
        word_count=1200,
        heading_count=8,
        heading_quality_score=0.94,
        table_count=0,
        figure_count=0,
        caption_count=0,
        ocr_region_count=0,
        avg_section_words=133.0,
        structure_depth=4,
        has_dense_structure=True,
        metadata={},
    )

    decision = service.route(features)

    assert decision.selected_strategy is ChunkingStrategy.HIERARCHICAL
    assert decision.special_chunk_mode is True
    assert decision.local_refine is False


def test_docx_heading_quality_controls_secondary_route() -> None:
    service = ChunkRoutingService()
    high_quality = DocumentFeatures(
        source_type=SourceType.DOCX,
        section_count=7,
        word_count=1400,
        heading_count=6,
        heading_quality_score=0.88,
        table_count=0,
        figure_count=0,
        caption_count=0,
        ocr_region_count=0,
        avg_section_words=200.0,
        structure_depth=3,
        has_dense_structure=True,
        metadata={},
    )
    low_quality = high_quality.model_copy(update={"heading_quality_score": 0.22, "heading_count": 1})

    high_decision = service.route(high_quality)
    low_decision = service.route(low_quality)

    assert high_decision.selected_strategy is ChunkingStrategy.HIERARCHICAL
    assert low_decision.selected_strategy is ChunkingStrategy.HYBRID
    assert low_decision.local_refine is True


def test_image_route_enables_special_chunks_and_fallback() -> None:
    service = ChunkRoutingService()
    features = DocumentFeatures(
        source_type=SourceType.IMAGE,
        section_count=1,
        word_count=24,
        heading_count=0,
        heading_quality_score=0.0,
        table_count=0,
        figure_count=1,
        caption_count=1,
        ocr_region_count=3,
        avg_section_words=24.0,
        structure_depth=1,
        has_dense_structure=False,
        metadata={},
    )

    decision = service.route(features)

    assert decision.selected_strategy is ChunkingStrategy.IMAGE
    assert decision.special_chunk_mode is True
    assert decision.local_refine is True
    assert decision.fallback is True
    assert any("image" in reason.lower() for reason in decision.reasons)
