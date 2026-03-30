from rag.query.context import QueryUnderstandingService


def test_query_understanding_service_detects_formula_special_queries() -> None:
    service = QueryUnderstandingService()

    result = service.analyze("解释一下这个公式表达了什么")

    assert result.intent == "special_lookup"
    assert result.query_type == "formula"
    assert result.special_targets == ["formula"]


def test_query_understanding_service_detects_pptx_and_xlsx_source_filters() -> None:
    service = QueryUnderstandingService()

    result = service.analyze("比较 pptx 和 xlsx 里的表格有什么区别")

    assert result.metadata_filters["source_types"] == ["pptx", "xlsx"]
    assert result.special_targets == ["table"]
