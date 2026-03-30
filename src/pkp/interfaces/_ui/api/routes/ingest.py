from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, model_validator

from pkp.schema._types import AccessPolicy
from pkp.interfaces._ui.dependencies import get_request_container

router = APIRouter()
UPLOAD_FILE = File(...)
UPLOAD_TITLE = Form(default=None)


class IngestRequest(BaseModel):
    source_type: str
    location: str | None = None
    content: str | None = None
    title: str | None = None
    access_policy: AccessPolicy | None = None

    @model_validator(mode="after")
    def validate_source_input(self) -> IngestRequest:
        if self.location is None and self.content is None:
            raise ValueError("either location or content is required")
        return self


@router.post("/ingest")
def ingest(payload: IngestRequest, request: Request) -> dict[str, int | str]:
    container = get_request_container(request)
    try:
        return container.ingest_runtime.ingest_source(
            source_type=payload.source_type,
            location=payload.location,
            content=payload.content,
            title=payload.title,
            access_policy=payload.access_policy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _infer_source_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if suffix in {".html", ".htm"}:
        return "browser_clip"
    if suffix in {".txt", ".text"}:
        return "plain_text"
    return "plain_text"


def _unique_upload_path(upload_dir: Path, filename: str) -> Path:
    candidate = upload_dir / Path(filename).name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        retry = upload_dir / f"{stem}-{counter}{suffix}"
        if not retry.exists():
            return retry
        counter += 1


@router.post("/ingest/upload")
async def ingest_upload(
    request: Request,
    file: UploadFile = UPLOAD_FILE,
    title: str | None = UPLOAD_TITLE,
) -> dict[str, int | str]:
    container = get_request_container(request)
    upload_dir = Path(request.app.state.workbench_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "upload.bin").name
    destination = _unique_upload_path(upload_dir, filename)
    content = await file.read()
    destination.write_bytes(content)
    await file.close()
    try:
        return container.ingest_runtime.ingest_source(
            source_type=_infer_source_type(filename),
            location=str(destination),
            title=title,
            access_policy=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
