from __future__ import annotations

from collections.abc import Callable
from typing import cast

from fastapi import Request

from pkp.bootstrap import build_runtime_container, load_settings
from pkp.runtime.container import RuntimeContainer

_container_factory: Callable[[], RuntimeContainer] | None = None


def set_container_factory(factory: Callable[[], RuntimeContainer]) -> None:
    global _container_factory
    _container_factory = factory


def clear_container_factory() -> None:
    global _container_factory
    _container_factory = None


def build_container() -> RuntimeContainer:
    if _container_factory is not None:
        return _container_factory()

    settings = load_settings()
    return build_runtime_container(settings)


def get_request_container(request: Request) -> RuntimeContainer:
    container = cast(RuntimeContainer | None, request.app.state.container)
    if container is None:
        factory = cast(Callable[[], RuntimeContainer], request.app.state.container_factory)
        container = factory()
        request.app.state.container = container
    return container
