from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.dependencies import get_request_container

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    mode: str = "fast"


@router.post("/query")
def query(payload: QueryRequest, request: Request) -> dict[str, object]:
    container = get_request_container(request)
    runtime = container.deep_research_runtime if payload.mode == "deep" else container.fast_query_runtime
    response = runtime.run(
        payload.query,
        build_execution_policy(
            task_type=TaskType.RESEARCH if payload.mode == "deep" else TaskType.LOOKUP,
            complexity_level=(ComplexityLevel.L4_RESEARCH if payload.mode == "deep" else ComplexityLevel.L1_DIRECT),
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.CLOUD_FIRST,
        ),
    )
    return cast(dict[str, object], response.model_dump(mode="json"))
