from pkp.schema._types.access import AccessPolicy, ExecutionLocationPreference, RuntimeMode
from pkp.schema._types.envelope import ExecutionPolicy
from pkp.schema._types.query import ComplexityLevel, TaskType


def test_execution_policy_carries_routing_contract_fields() -> None:
    policy = ExecutionPolicy(
        effective_access_policy=AccessPolicy.default(),
        task_type=TaskType.SYNTHESIS,
        complexity_level=ComplexityLevel.L4_RESEARCH,
        latency_budget=45,
        cost_budget=3.5,
        execution_location_preference=ExecutionLocationPreference.CLOUD_FIRST,
        fallback_allowed=True,
        source_scope=["doc-1", "artifact-1"],
        allowed_runtimes={RuntimeMode.DEEP},
    )

    assert policy.task_type is TaskType.SYNTHESIS
    assert policy.complexity_level is ComplexityLevel.L4_RESEARCH
    assert policy.source_scope == ["doc-1", "artifact-1"]
    assert policy.allowed_runtimes == {RuntimeMode.DEEP}
