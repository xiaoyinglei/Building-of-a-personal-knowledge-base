from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Residency(StrEnum):
    CLOUD_ALLOWED = "cloud_allowed"
    LOCAL_PREFERRED = "local_preferred"
    LOCAL_REQUIRED = "local_required"


class ExternalRetrievalPolicy(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class RuntimeMode(StrEnum):
    FAST = "fast"
    DEEP = "deep"


class ExecutionLocation(StrEnum):
    CLOUD = "cloud"
    LOCAL = "local"


class ExecutionLocationPreference(StrEnum):
    CLOUD_FIRST = "cloud_first"
    LOCAL_FIRST = "local_first"
    LOCAL_ONLY = "local_only"


_RESIDENCY_ORDER: dict[Residency, int] = {
    Residency.CLOUD_ALLOWED: 0,
    Residency.LOCAL_PREFERRED: 1,
    Residency.LOCAL_REQUIRED: 2,
}


class AccessPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    residency: Residency = Residency.CLOUD_ALLOWED
    external_retrieval: ExternalRetrievalPolicy = ExternalRetrievalPolicy.ALLOW
    allowed_runtimes: frozenset[RuntimeMode] = Field(
        default_factory=lambda: frozenset({RuntimeMode.FAST, RuntimeMode.DEEP})
    )
    allowed_locations: frozenset[ExecutionLocation] = Field(
        default_factory=lambda: frozenset({ExecutionLocation.CLOUD, ExecutionLocation.LOCAL})
    )
    sensitivity_tags: frozenset[str] = Field(default_factory=frozenset)

    @classmethod
    def default(cls) -> AccessPolicy:
        return cls()

    def narrow(self, other: AccessPolicy) -> AccessPolicy:
        allowed_runtimes = self.allowed_runtimes & other.allowed_runtimes
        if not allowed_runtimes:
            raise ValueError("allowed_runtimes cannot become empty during narrowing")

        allowed_locations = self.allowed_locations & other.allowed_locations
        if not allowed_locations:
            raise ValueError("allowed_locations cannot become empty during narrowing")

        residency = max((self.residency, other.residency), key=_RESIDENCY_ORDER.__getitem__)
        external_retrieval = (
            ExternalRetrievalPolicy.DENY
            if ExternalRetrievalPolicy.DENY in {self.external_retrieval, other.external_retrieval}
            else ExternalRetrievalPolicy.ALLOW
        )
        return AccessPolicy(
            residency=residency,
            external_retrieval=external_retrieval,
            allowed_runtimes=allowed_runtimes,
            allowed_locations=allowed_locations,
            sensitivity_tags=self.sensitivity_tags | other.sensitivity_tags,
        )

    @property
    def local_only(self) -> bool:
        return self.residency is Residency.LOCAL_REQUIRED

    def allows_runtime(self, mode: RuntimeMode) -> bool:
        return mode in self.allowed_runtimes

    def allows_location(self, location: ExecutionLocation) -> bool:
        return location in self.allowed_locations
