from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pkp.ui.api.routes.artifacts import router as artifacts_router
from pkp.ui.api.routes.health import router as health_router
from pkp.ui.api.routes.ingest import router as ingest_router
from pkp.ui.api.routes.query import router as query_router
from pkp.ui.api.routes.sessions import router as sessions_router
from pkp.ui.api.routes.workbench import router as workbench_router
from pkp.ui.dependencies import build_container


def create_app(
    container_factory: Callable[[], object] | None = None,
    *,
    workbench_upload_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Personal Knowledge Platform")
    app.state.container = None
    app.state.container_factory = container_factory or build_container
    app.state.workbench_upload_dir = workbench_upload_dir or Path("data/runtime/uploads")
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(artifacts_router)
    app.include_router(sessions_router)
    app.include_router(workbench_router)
    app.mount(
        "/workbench/assets",
        StaticFiles(directory=Path(__file__).resolve().parents[1] / "web" / "assets"),
        name="workbench-assets",
    )
    return app
