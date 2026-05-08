from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class DummyInput(BaseModel):
    text: str


class DummyOutput(BaseModel):
    result: str


_dummy_spec = ToolSpec(
    name="dummy",
    description="A dummy tool",
    input_model=DummyInput,
    output_model=DummyOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    timeout_seconds=1.0,
)


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        assert registry.get("dummy") is _dummy_spec

    def test_get_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_all(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        another = ToolSpec(
            name="another",
            description="x",
            input_model=DummyInput,
            output_model=DummyOutput,
            error_model=ToolError,
            permissions=ToolPermissions(),
            timeout_seconds=2.0,
        )
        registry.register(another)
        names = [spec.name for spec in registry.list_all()]
        assert "dummy" in names
        assert "another" in names

    def test_register_duplicate_overwrites(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        updated = ToolSpec(
            name="dummy",
            description="updated",
            input_model=DummyInput,
            output_model=DummyOutput,
            error_model=ToolError,
            permissions=ToolPermissions(),
            timeout_seconds=3.0,
        )
        registry.register(updated)
        assert registry.get("dummy").timeout_seconds == 3.0
