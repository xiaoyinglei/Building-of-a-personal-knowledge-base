from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from pkp.types.access import AccessPolicy, ExecutionLocationPreference
from pkp.types.envelope import ExecutionPolicy
from pkp.types.query import ComplexityLevel, TaskType


class RoutingThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    fast_min_evidence_chunks: int = 2
    fast_min_sections: int = 1
    deep_min_evidence_chunks: int = 4
    deep_min_supporting_units: int = 2
    max_retrieval_rounds: int = 4
    max_recursive_depth: int = 2
    default_wall_clock_budget_seconds: int = 180
    default_synthesis_retry_count: int = 1


def default_access_policy() -> AccessPolicy:
    return AccessPolicy.default()


def build_execution_policy(
    *,
    task_type: TaskType,
    complexity_level: ComplexityLevel,
    access_policy: AccessPolicy | None = None,
    source_scope: list[str] | None = None,
    latency_budget: int = 30,
    cost_budget: float = 1.0,
    execution_location_preference: ExecutionLocationPreference = (ExecutionLocationPreference.CLOUD_FIRST),
    fallback_allowed: bool = True,
) -> ExecutionPolicy:
    return ExecutionPolicy(
        effective_access_policy=access_policy or default_access_policy(),
        task_type=task_type,
        complexity_level=complexity_level,
        latency_budget=latency_budget,
        cost_budget=cost_budget,
        execution_location_preference=execution_location_preference,
        fallback_allowed=fallback_allowed,
        source_scope=source_scope or [],
    )
