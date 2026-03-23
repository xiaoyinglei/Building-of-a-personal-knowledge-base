from pathlib import Path

from pkp.service.ingest_service import IngestService
from pkp.types.access import (
    AccessPolicy,
    ExecutionLocation,
    ExternalRetrievalPolicy,
    Residency,
    RuntimeMode,
)


def test_plain_text_ingest_infers_root_toc_and_inherits_policy(tmp_path: Path) -> None:
    service = IngestService.create_in_memory(tmp_path)
    policy = AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes={RuntimeMode.FAST},
        allowed_locations={ExecutionLocation.LOCAL},
        sensitivity_tags={"private"},
    )

    result = service.ingest_plain_text(
        location="notes/plain.txt",
        text="First paragraph.\n\nSecond paragraph.",
        owner="user",
        access_policy=policy,
    )

    assert result.segments[0].toc_path == ["plain"]
    assert result.segments[0].heading_level == 1
    assert result.chunks[0].effective_access_policy == policy
