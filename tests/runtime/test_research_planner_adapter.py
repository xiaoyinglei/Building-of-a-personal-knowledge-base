from pkp.runtime.adapters import ResearchPlannerAdapter


def test_research_planner_adapter_detects_comparison_queries_in_english_and_chinese() -> None:
    planner = ResearchPlannerAdapter()

    english = planner.decompose("compare Fast Path and Deep Path")
    chinese = planner.decompose("比较 Fast Path 和 Deep Path")

    assert english == [
        "compare Fast Path and Deep Path",
        "compare Fast Path and Deep Path evidence",
    ]
    assert chinese == [
        "比较 Fast Path 和 Deep Path",
        "比较 Fast Path 和 Deep Path evidence",
    ]


def test_research_planner_adapter_uses_memory_hints_only_for_query_expansion() -> None:
    planner = ResearchPlannerAdapter()

    expanded = planner.expand(
        "agent reliability",
        evidence_matrix=[],
        round_index=1,
        memory_hints=["topic page", "conflict map"],
    )

    assert expanded == [
        "agent reliability details",
        "agent reliability topic page",
        "agent reliability conflict map",
    ]
