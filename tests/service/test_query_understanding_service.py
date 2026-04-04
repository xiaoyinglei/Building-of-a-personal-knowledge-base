from __future__ import annotations

import pytest

from rag.query.understanding import QueryUnderstandingService
from rag.schema._types.query import ConfidenceBand, QueryIntent


@pytest.mark.parametrize(
    ("query", "intent", "query_type"),
    [
        ("这个项目做什么？", QueryIntent.FACTUAL_LOOKUP, "definition_lookup"),
        ("这个系统为什么要做图扩展？", QueryIntent.SEMANTIC_LOOKUP, "semantic_lookup"),
        ("第2页讲了什么风险？", QueryIntent.METADATA_CONSTRAINED_LOOKUP, "page_constrained_lookup"),
        ("讲系统分层的那一部分在哪一节？", QueryIntent.SECTION_LOOKUP, "heading_lookup"),
        ("系统架构分为哪几层？", QueryIntent.STRUCTURE_LOOKUP, "architecture_lookup"),
        ("总结一下这个系统的核心能力。", QueryIntent.SUMMARY_REQUEST, "document_summary_request"),
        ("这个系统的处理流程是怎样的？", QueryIntent.FLOW_PROCESS_REQUEST, "workflow_request"),
        ("比较 Alpha 和 Beta 的检索链路差异。", QueryIntent.COMPARISON_REQUEST, "comparative_lookup"),
        ("解释一下这个公式表达了什么", QueryIntent.SPECIAL_CONTENT_LOOKUP, "formula_lookup"),
        ("比较 pptx 和 xlsx 里的表格有什么区别", QueryIntent.COMPARISON_REQUEST, "comparative_lookup"),
    ],
)
def test_query_understanding_service_covers_formal_intents(
    query: str,
    intent: QueryIntent,
    query_type: str,
) -> None:
    service = QueryUnderstandingService()

    result = service.analyze(query)

    assert result.intent is intent
    assert result.query_type == query_type


def test_query_understanding_service_extracts_semantic_section_constraints() -> None:
    service = QueryUnderstandingService()

    result = service.analyze("讲系统分层的那一部分在哪一节？")

    assert result.needs_structure is True
    assert result.structure_constraints.requires_structure_match is True
    assert result.structure_constraints.prefer_heading_match is True
    assert "architecture" in result.structure_constraints.semantic_section_families
    assert "系统架构" in result.preferred_section_terms
    assert result.routing_hints.structure_priority >= 0.45


def test_query_understanding_service_extracts_metadata_and_special_constraints() -> None:
    service = QueryUnderstandingService()

    result = service.analyze("pdf 第2到4页的表格指标是什么？")

    assert result.needs_metadata is True
    assert result.needs_special is True
    assert result.metadata_filters.source_types == ["pdf"]
    assert [(item.start, item.end) for item in result.metadata_filters.page_ranges] == [(2, 4)]
    assert result.special_targets == ["table"]


def test_query_understanding_service_extracts_source_type_constraints() -> None:
    service = QueryUnderstandingService()

    result = service.analyze("比较 pptx 和 xlsx 里的表格有什么区别")

    assert result.metadata_filters.source_types == ["pptx", "xlsx"]
    assert result.special_targets == ["table"]
    assert result.needs_metadata is True


def test_query_understanding_service_produces_low_confidence_for_vague_query() -> None:
    service = QueryUnderstandingService()

    result = service.analyze("这个呢？")

    assert result.confidence_band is ConfidenceBand.LOW
    assert result.should_rewrite_query is True
