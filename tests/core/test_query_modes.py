from pkp.algorithms.retrieval.mode_planner import RetrievalPlanBuilder
from pkp.query.query import QueryMode
from pkp.types.query import QueryUnderstanding


def _understanding(
    *,
    needs_sparse: bool = True,
    needs_structure: bool = False,
    needs_special: bool = False,
    needs_metadata: bool = False,
) -> QueryUnderstanding:
    return QueryUnderstanding(
        intent="semantic_lookup",
        query_type="general",
        needs_dense=True,
        needs_sparse=needs_sparse,
        needs_structure=needs_structure,
        needs_special=needs_special,
        needs_metadata=needs_metadata,
    )


def test_retrieval_plan_builder_maps_explicit_modes_to_expected_branches() -> None:
    planner = RetrievalPlanBuilder()
    understanding = _understanding()

    assert planner.build(query_understanding=understanding, requested_mode=QueryMode.NAIVE).internal_branches == (
        "vector",
    )
    assert planner.build(query_understanding=understanding, requested_mode=QueryMode.LOCAL).internal_branches == (
        "local",
    )
    assert planner.build(query_understanding=understanding, requested_mode=QueryMode.GLOBAL).internal_branches == (
        "global",
    )
    assert planner.build(query_understanding=understanding, requested_mode=QueryMode.HYBRID).internal_branches == (
        "local",
        "global",
    )


def test_retrieval_plan_builder_extends_mix_mode_with_sparse_structure_special_and_metadata() -> None:
    planner = RetrievalPlanBuilder()

    plan = planner.build(
        query_understanding=_understanding(
            needs_sparse=True,
            needs_structure=True,
            needs_special=True,
            needs_metadata=True,
        ),
        requested_mode=QueryMode.MIX,
    )

    assert plan.internal_branches == ("local", "global", "vector", "full_text", "section", "special", "metadata")
    assert plan.allow_graph_expansion is True
    assert plan.allow_special is True
    assert plan.allow_structure is True
    assert plan.allow_metadata is True


def test_retrieval_plan_builder_disables_graph_expansion_for_naive_mode() -> None:
    planner = RetrievalPlanBuilder()

    plan = planner.build(
        query_understanding=_understanding(),
        requested_mode="naive",
    )

    assert plan.mode is QueryMode.NAIVE
    assert plan.allow_graph_expansion is False
