from __future__ import annotations

from rag.agent.state import AgentState


def observe_node(state: AgentState) -> dict:
    results = state.get("tool_results", [])
    if not results:
        return {}
    return {"insufficient_evidence_flag": any(result.status == "error" for result in results)}
