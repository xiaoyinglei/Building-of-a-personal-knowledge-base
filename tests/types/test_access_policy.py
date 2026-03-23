from pkp.types.access import (
    AccessPolicy,
    ExecutionLocation,
    ExternalRetrievalPolicy,
    Residency,
    RuntimeMode,
)


def test_access_policy_narrowing_restricts_runtime_location_and_external_access() -> None:
    baseline = AccessPolicy(
        residency=Residency.CLOUD_ALLOWED,
        external_retrieval=ExternalRetrievalPolicy.ALLOW,
        allowed_runtimes={RuntimeMode.FAST, RuntimeMode.DEEP},
        allowed_locations={ExecutionLocation.CLOUD, ExecutionLocation.LOCAL},
        sensitivity_tags={"general"},
    )

    narrowed = AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes={RuntimeMode.FAST},
        allowed_locations={ExecutionLocation.LOCAL},
        sensitivity_tags={"restricted"},
    )

    result = baseline.narrow(narrowed)

    assert result.residency is Residency.LOCAL_REQUIRED
    assert result.external_retrieval is ExternalRetrievalPolicy.DENY
    assert result.allowed_runtimes == {RuntimeMode.FAST}
    assert result.allowed_locations == {ExecutionLocation.LOCAL}
    assert result.sensitivity_tags == {"general", "restricted"}


def test_access_policy_narrowing_rejects_empty_runtime_or_location_sets() -> None:
    baseline = AccessPolicy.default()
    narrowed = AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes={RuntimeMode.DEEP},
        allowed_locations={ExecutionLocation.LOCAL},
        sensitivity_tags=set(),
    )

    incompatible = AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes={RuntimeMode.FAST},
        allowed_locations={ExecutionLocation.LOCAL},
        sensitivity_tags=set(),
    )

    result = baseline.narrow(narrowed)
    assert result.allowed_runtimes == {RuntimeMode.DEEP}

    try:
        result.narrow(incompatible)
    except ValueError as exc:
        assert "allowed_runtimes" in str(exc)
    else:
        raise AssertionError("expected narrow() to reject widening")
