from __future__ import annotations

import pytest

from rag.query.analysis import QueryUnderstandingService


@pytest.mark.parametrize(
    ("query", "query_type"),
    [
        ("这个项目做什么？", "lookup"),
        ("第2页讲了什么风险？", "scoped_lookup"),
        ("讲系统分层的那一部分在哪一节？", "section_lookup"),
        ("系统架构分为哪几层？", "structure_lookup"),
        ("总结一下这个系统的核心能力。", "summary"),
        ("这个系统的处理流程是怎样的？", "process"),
        ("比较 Alpha 和 Beta 的检索链路差异。", "comparison"),
        ("解释一下这个公式表达了什么", "special_lookup"),
    ],
)
def test_query_understanding_service_classifies_coarse_query_types(query: str, query_type: str) -> None:
    result = QueryUnderstandingService().analyze(query)

    assert result.query_type == query_type


def test_query_understanding_service_extracts_structure_constraints_from_explicit_section_query() -> None:
    result = QueryUnderstandingService().analyze("讲系统分层的那一部分在哪一节？")

    assert result.needs_structure is True
    assert result.structure_constraints.requires_structure_match is True
    assert result.structure_constraints.prefer_heading_match is True
    assert "architecture" in result.structure_constraints.semantic_section_families
    assert "系统架构" in result.preferred_section_terms


def test_query_understanding_service_extracts_metadata_and_special_constraints() -> None:
    result = QueryUnderstandingService().analyze("pdf 第2到4页的表格指标是什么？")

    assert result.needs_metadata is True
    assert result.needs_special is True
    assert result.metadata_filters.source_types == ["pdf"]
    assert [(item.start, item.end) for item in result.metadata_filters.page_ranges] == [(2, 4)]
    assert result.special_targets == ["table"]


def test_query_understanding_service_extracts_multiple_source_types() -> None:
    result = QueryUnderstandingService().analyze("比较 pptx 和 xlsx 里的表格有什么区别")

    assert result.query_type == "comparison"
    assert result.metadata_filters.source_types == ["pptx", "xlsx"]
    assert result.special_targets == ["table"]
    assert result.needs_metadata is True


def test_query_understanding_service_marks_process_queries_for_graph_expansion() -> None:
    result = QueryUnderstandingService().analyze("这个系统的处理流程是怎样的？")

    assert result.query_type == "process"
    assert result.needs_graph_expansion is True
