from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition, ToolPolicy
from rag.agent.graphs.base import build_agent_graph
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class EchoInput(BaseModel):
    message: str


class EchoOutput(BaseModel):
    message: str


_echo_spec = ToolSpec(
    name="echo",
    description="Echo back the message",
    input_model=EchoInput,
    output_model=EchoOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    timeout_seconds=1.0,
)


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_echo_spec)
    return registry


def _make_config(*, max_parallel_calls: int = 4) -> AgentRunConfig:
    return AgentRunConfig(
        run_id="graph-test",
        thread_id="graph-test",
        budget_total=10000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
        execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        tool_policy=ToolPolicy(max_parallel_calls=max_parallel_calls),
    )


def _make_definition(*, max_iterations: int = 3) -> AgentDefinition:
    return AgentDefinition(
        agent_type="echo_agent",
        description="Test echo agent",
        system_prompt="You have an echo tool.",
        allowed_tools=["echo"],
        max_iterations=max_iterations,
    )


def _initial_state(
    *,
    pending_tool_calls: list[ToolCallPlan] | None = None,
    config: AgentRunConfig | None = None,
) -> AgentState:
    return {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "research task requiring the agent loop",
        "run_config": config or _make_config(),
        "plan": None,
        "iteration": 0,
        "status": "running",
        "pending_tool_calls": pending_tool_calls or [],
        "confirmed_tool_call_ids": set(),
        "user_decision": None,
        "next_subtasks": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "subtask_results": {},
        "terminal_subtasks": set(),
        "successful_subtasks": set(),
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }


class TestBaseGraph:
    def test_builds_without_errors(self) -> None:
        graph = build_agent_graph(definition=_make_definition(), tool_registry=_make_registry())
        assert graph is not None

    @pytest.mark.anyio
    async def test_direct_route_without_tools_synthesizes_final_answer(self) -> None:
        graph = build_agent_graph(definition=_make_definition(), tool_registry=_make_registry())
        result = await graph.ainvoke(_initial_state(), config={"configurable": {"thread_id": "graph-test"}})
        assert result["status"] == "done"
        assert result["final_answer"] is not None

    @pytest.mark.anyio
    async def test_registered_tool_without_runner_fails_closed(self) -> None:
        graph = build_agent_graph(definition=_make_definition(), tool_registry=_make_registry())
        call = ToolCallPlan.create("echo", {"message": "hello"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "error"
        assert tool_result.error.code == "tool_not_implemented"
        assert result["insufficient_evidence_flag"] is True

    @pytest.mark.anyio
    async def test_unregistered_tool_records_failure_result(self) -> None:
        graph = build_agent_graph(definition=_make_definition(), tool_registry=_make_registry())
        call = ToolCallPlan.create("missing_tool", {"message": "hello"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "error"
        assert tool_result.error.code == "tool_not_registered"

    @pytest.mark.anyio
    async def test_invalid_tool_arguments_record_failure_result(self) -> None:
        graph = build_agent_graph(definition=_make_definition(), tool_registry=_make_registry())
        call = ToolCallPlan.create("echo", {"unexpected": "value"})
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=[call]),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        [tool_result] = result["tool_results"]
        assert tool_result.status == "error"
        assert tool_result.error.code == "invalid_arguments"

    @pytest.mark.anyio
    async def test_max_parallel_calls_batches_pending_tools(self) -> None:
        graph = build_agent_graph(definition=_make_definition(max_iterations=3), tool_registry=_make_registry())
        calls = [
            ToolCallPlan.create("echo", {"message": "one"}),
            ToolCallPlan.create("echo", {"message": "two"}),
        ]
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=calls, config=_make_config(max_parallel_calls=1)),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        assert len(result["tool_results"]) == 2
        assert result["pending_tool_calls"] == []
        assert result["iteration"] == 2

    @pytest.mark.anyio
    async def test_max_iterations_fails_closed_with_pending_tools(self) -> None:
        graph = build_agent_graph(definition=_make_definition(max_iterations=1), tool_registry=_make_registry())
        calls = [
            ToolCallPlan.create("echo", {"message": "one"}),
            ToolCallPlan.create("echo", {"message": "two"}),
        ]
        result = await graph.ainvoke(
            _initial_state(pending_tool_calls=calls, config=_make_config(max_parallel_calls=1)),
            config={"configurable": {"thread_id": "graph-test"}},
        )
        assert result["status"] == "failed"
        assert len(result["tool_results"]) == 1
        assert len(result["pending_tool_calls"]) == 1
        assert result["stop_reason"] == "max_iterations"
