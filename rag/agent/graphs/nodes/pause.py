from __future__ import annotations

from rag.agent.state import AgentState


def pause_node(state: AgentState) -> dict:
    return {
        "status": "paused",
        "needs_user_input": state.get("needs_user_input"),
    }
