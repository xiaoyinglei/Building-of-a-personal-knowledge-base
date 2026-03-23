from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from pkp.ui.dependencies import get_request_container

router = APIRouter()


class IngestRequest(BaseModel):
    source_type: str
    location: str


@router.post("/ingest")
def ingest(payload: IngestRequest, request: Request) -> dict[str, int | str]:
    container = get_request_container(request)
    return container.ingest_runtime.ingest_source(
        source_type=payload.source_type,
        location=payload.location,
    )
