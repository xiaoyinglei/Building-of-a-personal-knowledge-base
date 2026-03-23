from pkp.runtime.deep_research_runtime import resolve_execution_locations
from pkp.types import (
    AccessPolicy,
    ExecutionLocation,
    ExecutionLocationPreference,
    ExternalRetrievalPolicy,
    Residency,
    RuntimeMode,
)


def test_resolve_execution_locations_respects_local_required_policy() -> None:
    policy = AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes={RuntimeMode.FAST, RuntimeMode.DEEP},
        allowed_locations={ExecutionLocation.LOCAL},
        sensitivity_tags={"secret"},
    )

    locations = resolve_execution_locations(policy, ExecutionLocationPreference.CLOUD_FIRST)

    assert locations == [ExecutionLocation.LOCAL]


def test_resolve_execution_locations_prefers_requested_order_when_policy_allows_both() -> None:
    locations = resolve_execution_locations(
        AccessPolicy.default(),
        ExecutionLocationPreference.LOCAL_FIRST,
    )

    assert locations == [ExecutionLocation.LOCAL, ExecutionLocation.CLOUD]
