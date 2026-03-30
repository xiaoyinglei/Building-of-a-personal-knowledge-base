from __future__ import annotations

from dataclasses import dataclass

from pkp.interfaces._runtime.artifact_promotion_runtime import ArtifactPromotionRuntime
from pkp.interfaces._runtime.deep_research_runtime import DeepResearchRuntime
from pkp.interfaces._runtime.fast_query_runtime import FastQueryRuntime
from pkp.interfaces._runtime.ingest_runtime import IngestRuntime
from pkp.interfaces._runtime.session_runtime import SessionRuntime
from pkp.utils._telemetry import TelemetryService


@dataclass(slots=True)
class RuntimeContainer:
    ingest_runtime: IngestRuntime
    fast_query_runtime: FastQueryRuntime
    deep_research_runtime: DeepResearchRuntime
    artifact_promotion_runtime: ArtifactPromotionRuntime
    session_runtime: SessionRuntime
    diagnostics_runtime: object | None = None
    metadata_repo: object | None = None
    telemetry_service: TelemetryService | None = None
