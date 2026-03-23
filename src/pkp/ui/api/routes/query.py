from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pkp.bootstrap import load_settings
from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.dependencies import get_request_container

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    mode: str = "fast"
    source_scope: list[str] = Field(default_factory=list)
    latency_budget: int | None = None
    token_budget: int | None = None
    execution_location_preference: ExecutionLocationPreference | None = None
    fallback_allowed: bool | None = None
    cost_budget: float | None = None


@router.post("/query")
def query(payload: QueryRequest, request: Request) -> dict[str, object]:
    container = get_request_container(request)
    settings = load_settings()
    runtime = container.deep_research_runtime if payload.mode == "deep" else container.fast_query_runtime
    is_deep = payload.mode == "deep"
    latency_budget = (
        payload.latency_budget
        if payload.latency_budget is not None
        else settings.runtime.default_wall_clock_budget_seconds
    )
    cost_budget = payload.cost_budget if payload.cost_budget is not None else 1.0
    token_budget = payload.token_budget if payload.token_budget is not None else settings.runtime.max_token_budget
    execution_location_preference = (
        payload.execution_location_preference or settings.runtime.execution_location_preference
    )
    fallback_allowed = (
        payload.fallback_allowed if payload.fallback_allowed is not None else settings.runtime.fallback_allowed
    )
    response = runtime.run(
        payload.query,
        build_execution_policy(
            task_type=TaskType.RESEARCH if is_deep else TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L4_RESEARCH if is_deep else ComplexityLevel.L1_DIRECT,
            access_policy=default_access_policy(),
            source_scope=payload.source_scope,
            latency_budget=latency_budget,
            cost_budget=cost_budget,
            token_budget=token_budget,
            execution_location_preference=execution_location_preference,
            fallback_allowed=fallback_allowed,
        ),
    )
    return cast(dict[str, object], response.model_dump(mode="json"))
