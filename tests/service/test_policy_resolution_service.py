from pkp.ingest._policy.service import PolicyResolutionService
from pkp.schema._types.access import (
    AccessPolicy,
    ExecutionLocation,
    ExternalRetrievalPolicy,
    Residency,
    RuntimeMode,
)


def test_policy_resolution_narrows_source_policy_to_chunk_policy() -> None:
    service = PolicyResolutionService()
    source = AccessPolicy(
        residency=Residency.CLOUD_ALLOWED,
        external_retrieval=ExternalRetrievalPolicy.ALLOW,
        allowed_runtimes={RuntimeMode.FAST, RuntimeMode.DEEP},
        allowed_locations={ExecutionLocation.CLOUD, ExecutionLocation.LOCAL},
        sensitivity_tags={"general"},
    )
    document = AccessPolicy(
        residency=Residency.LOCAL_PREFERRED,
        external_retrieval=ExternalRetrievalPolicy.ALLOW,
        allowed_runtimes={RuntimeMode.FAST, RuntimeMode.DEEP},
        allowed_locations={ExecutionLocation.LOCAL},
        sensitivity_tags={"general"},
    )
    segment = AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes={RuntimeMode.FAST},
        allowed_locations={ExecutionLocation.LOCAL},
        sensitivity_tags={"restricted"},
    )

    resolved = service.resolve_effective_access_policy(
        source_policy=source,
        document_policy=document,
        segment_policy=segment,
    )

    assert resolved.residency is Residency.LOCAL_REQUIRED
    assert resolved.external_retrieval is ExternalRetrievalPolicy.DENY
    assert resolved.allowed_runtimes == {RuntimeMode.FAST}
    assert resolved.allowed_locations == {ExecutionLocation.LOCAL}
    assert resolved.sensitivity_tags == {"general", "restricted"}
