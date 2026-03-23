from __future__ import annotations

from pkp.types.access import AccessPolicy


class PolicyResolutionService:
    def resolve_effective_access_policy(
        self,
        *,
        source_policy: AccessPolicy,
        document_policy: AccessPolicy | None = None,
        segment_policy: AccessPolicy | None = None,
        chunk_policy: AccessPolicy | None = None,
    ) -> AccessPolicy:
        resolved = source_policy
        if document_policy is not None:
            resolved = resolved.narrow(document_policy)
        if segment_policy is not None:
            resolved = resolved.narrow(segment_policy)
        if chunk_policy is not None:
            resolved = resolved.narrow(chunk_policy)
        return resolved
