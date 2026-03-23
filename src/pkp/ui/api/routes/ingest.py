from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, model_validator

from pkp.ui.dependencies import get_request_container

router = APIRouter()


class IngestRequest(BaseModel):
    source_type: str
    location: str | None = None
    content: str | None = None
    title: str | None = None

    @model_validator(mode="after")
    def validate_source_input(self) -> IngestRequest:
        if self.location is None and self.content is None:
            raise ValueError("either location or content is required")
        return self


@router.post("/ingest")
def ingest(payload: IngestRequest, request: Request) -> dict[str, int | str]:
    container = get_request_container(request)
    return container.ingest_runtime.ingest_source(
        source_type=payload.source_type,
        location=payload.location,
        content=payload.content,
        title=payload.title,
    )
