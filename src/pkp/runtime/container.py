from __future__ import annotations

from dataclasses import dataclass

from pkp.runtime.artifact_promotion_runtime import ArtifactPromotionRuntime
from pkp.runtime.deep_research_runtime import DeepResearchRuntime
from pkp.runtime.fast_query_runtime import FastQueryRuntime
from pkp.runtime.ingest_runtime import IngestRuntime
from pkp.runtime.session_runtime import SessionRuntime
from pkp.service.telemetry_service import TelemetryService


@dataclass(slots=True)
class RuntimeContainer:
    ingest_runtime: IngestRuntime
    fast_query_runtime: FastQueryRuntime
    deep_research_runtime: DeepResearchRuntime
    artifact_promotion_runtime: ArtifactPromotionRuntime
    session_runtime: SessionRuntime
    metadata_repo: object | None = None
    telemetry_service: TelemetryService | None = None
