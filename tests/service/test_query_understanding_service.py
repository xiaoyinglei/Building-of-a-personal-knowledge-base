from pkp.query.context import QueryUnderstandingService


def test_query_understanding_service_detects_formula_special_queries() -> None:
    service = QueryUnderstandingService()

    result = service.analyze("解释一下这个公式表达了什么")

    assert result.intent == "special_lookup"
    assert result.query_type == "formula"
    assert result.special_targets == ["formula"]
