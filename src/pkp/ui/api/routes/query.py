from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from pkp.bootstrap import load_settings
from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, TaskType
from pkp.ui.dependencies import get_request_container

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    mode: str = "fast"


@router.post("/query")
def query(payload: QueryRequest, request: Request) -> dict[str, object]:
    container = get_request_container(request)
    settings = load_settings()
    runtime = container.deep_research_runtime if payload.mode == "deep" else container.fast_query_runtime
    response = runtime.run(
        payload.query,
        build_execution_policy(
            task_type=TaskType.RESEARCH if payload.mode == "deep" else TaskType.LOOKUP,
            complexity_level=(ComplexityLevel.L4_RESEARCH if payload.mode == "deep" else ComplexityLevel.L1_DIRECT),
            access_policy=default_access_policy(),
            latency_budget=settings.runtime.default_wall_clock_budget_seconds,
            token_budget=settings.runtime.max_token_budget,
            execution_location_preference=settings.runtime.execution_location_preference,
            fallback_allowed=settings.runtime.fallback_allowed,
        ),
    )
    return cast(dict[str, object], response.model_dump(mode="json"))
