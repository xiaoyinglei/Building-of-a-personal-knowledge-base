from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pkp.ui.dependencies import get_request_container

router = APIRouter()


class ApproveArtifactRequest(BaseModel):
    artifact_id: str


def _artifact_payload(result: Any) -> dict[str, object]:
    payload = result if isinstance(result, dict) else result.model_dump(mode="json")
    return cast(dict[str, object], payload)


@router.get("/artifacts")
def list_artifacts(request: Request) -> list[dict[str, object]]:
    container = get_request_container(request)
    return [_artifact_payload(artifact) for artifact in container.artifact_promotion_runtime.list_artifacts()]


@router.get("/artifacts/{artifact_id}")
def show_artifact(artifact_id: str, request: Request) -> dict[str, object]:
    container = get_request_container(request)
    try:
        artifact = container.artifact_promotion_runtime.get_artifact(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _artifact_payload(artifact)


@router.post("/artifacts/approve")
def approve(payload: ApproveArtifactRequest, request: Request) -> dict[str, object]:
    container = get_request_container(request)
    result = container.artifact_promotion_runtime.approve(payload.artifact_id)
    return _artifact_payload(result)
