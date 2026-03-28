from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pkp.types.access import AccessPolicy, ExecutionLocationPreference


@dataclass(frozen=True, slots=True)
class QueryOptions:
    mode: Literal["naive", "local", "global", "hybrid", "mix"] = "mix"
    source_scope: tuple[str, ...] = ()
    access_policy: AccessPolicy = field(default_factory=AccessPolicy.default)
    execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST
    max_context_tokens: int = 1200
    max_evidence_chunks: int = 8
