from __future__ import annotations

from rag.agent.core.definition import AgentDefinition
from rag.agent.state import AgentState


async def evaluate_node(state: AgentState, *, definition: AgentDefinition) -> dict:
    iteration = state.get("iteration", 0)

    try:
        from rag.agent.core.context import RuntimeRegistry

        handles = RuntimeRegistry.get(state["run_config"].run_id)
        if await handles.budget_ledger.remaining() <= 0:
            return {"status": "failed", "stop_reason": "budget_exhausted"}
    except KeyError:
        pass

    pending = state.get("pending_tool_calls", [])
    executed_batch = bool(state.get("tool_results"))
    next_iteration = iteration + 1 if executed_batch else iteration
    if not pending:
        return {"status": "done", "stop_reason": "no_pending_tools", "iteration": next_iteration}

    if next_iteration >= definition.max_iterations:
        return {"status": "failed", "stop_reason": "max_iterations", "iteration": next_iteration}

    return {"status": "running", "iteration": next_iteration}


def route_after_evaluate(state: AgentState) -> str:
    if state.get("status") in {"done", "failed", "paused"}:
        return "synthesize"
    return "execute"
