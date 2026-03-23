from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from pkp.ui.api.routes.artifacts import router as artifacts_router
from pkp.ui.api.routes.health import router as health_router
from pkp.ui.api.routes.ingest import router as ingest_router
from pkp.ui.api.routes.query import router as query_router
from pkp.ui.dependencies import build_container


def create_app(container_factory: Callable[[], object] | None = None) -> FastAPI:
    app = FastAPI(title="Personal Knowledge Platform")
    app.state.container = None
    app.state.container_factory = container_factory or build_container
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(artifacts_router)
    return app
