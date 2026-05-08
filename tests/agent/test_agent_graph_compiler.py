from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.compiler import AgentGraphCompiler
from rag.agent.core.definition import AgentDefinition
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class SearchInput(BaseModel):
    query: str


class SearchOutput(BaseModel):
    items: list[str]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="vector_search",
            description="Vector search",
            input_model=SearchInput,
            output_model=SearchOutput,
            error_model=ToolError,
            permissions=ToolPermissions(read_db=True, embed=True),
            timeout_seconds=5.0,
        )
    )
    return registry


def _definition(*, allowed_tools: list[str]) -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="Research agent",
        system_prompt="Use grounded evidence.",
        allowed_tools=allowed_tools,
    )


def test_compiler_builds_graph_for_registered_agent_tools() -> None:
    compiler = AgentGraphCompiler(tool_registry=_registry())

    graph = compiler.compile(_definition(allowed_tools=["vector_search"]))

    assert hasattr(graph, "ainvoke")


def test_compiler_rejects_unregistered_agent_tools() -> None:
    compiler = AgentGraphCompiler(tool_registry=_registry())

    with pytest.raises(ValueError, match="unregistered tools: missing_tool"):
        compiler.compile(_definition(allowed_tools=["vector_search", "missing_tool"]))
