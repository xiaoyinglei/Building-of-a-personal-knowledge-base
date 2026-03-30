from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request

from pkp.interfaces._ui.dependencies import get_request_container

router = APIRouter()


@router.get("/sessions/{session_id}")
def show_session(session_id: str, request: Request) -> dict[str, object]:
    container = get_request_container(request)
    snapshot = container.session_runtime.get(session_id)
    return asdict(snapshot)
