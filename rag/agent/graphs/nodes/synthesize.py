from __future__ import annotations

from rag.agent.state import AgentState


def synthesize_node(state: AgentState) -> dict:
    tool_results = state.get("tool_results", [])
    ok_count = sum(1 for result in tool_results if result.status == "ok")
    error_count = sum(1 for result in tool_results if result.status == "error")
    status = state.get("status")
    final_status = "failed" if status == "failed" else "done"
    return {
        "status": final_status,
        "final_answer": f"Agent run complete. {ok_count} tools succeeded, {error_count} failed.",
        "groundedness_flag": ok_count > 0,
        "insufficient_evidence_flag": error_count > 0,
    }
