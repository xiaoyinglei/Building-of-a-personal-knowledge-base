from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from pkp.ui.dependencies import get_request_container

router = APIRouter()


class ApproveArtifactRequest(BaseModel):
    artifact_id: str


@router.post("/artifacts/approve")
def approve(payload: ApproveArtifactRequest, request: Request) -> dict[str, object]:
    container = get_request_container(request)
    result = container.artifact_promotion_runtime.approve(payload.artifact_id)
    payload_dict = result if isinstance(result, dict) else result.model_dump(mode="json")
    return cast(dict[str, object], payload_dict)
