from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from pkp.ui.dependencies import get_request_container

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    container = get_request_container(request)
    diagnostics_runtime = getattr(container, "diagnostics_runtime", None)
    if diagnostics_runtime is None:
        return {
            "status": "ok",
            "providers": [],
            "indices": {
                "documents": 0,
                "chunks": 0,
                "vectors": 0,
                "missing_vectors": 0,
            },
        }
    report = diagnostics_runtime.report()
    payload = report if isinstance(report, dict) else report.model_dump(mode="json")
    return cast(dict[str, object], payload)
