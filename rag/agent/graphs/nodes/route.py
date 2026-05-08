from __future__ import annotations

from rag.agent.state import AgentState


def route_node(state: AgentState) -> dict:
    task = state.get("task", "")
    complexity = _classify_complexity(task)
    if complexity == "simple":
        return {"status": "fast_path", "route_reason": "simple_lookup"}
    if complexity == "decompose":
        return {"status": "decompose", "route_reason": "multi_hop_or_compare"}
    return {"status": "direct", "route_reason": "single_agent_research"}


def _classify_complexity(task: str) -> str:
    normalized = task.lower()
    compare_keywords = ("compare", "对比", "diff", "区别", "vs", "versus")
    multi_hop_keywords = ("timeline", "时间线", "history", "how did", "why did", "为什么")
    if any(keyword in normalized for keyword in compare_keywords):
        return "decompose"
    if any(keyword in normalized for keyword in multi_hop_keywords):
        return "decompose"
    if len(normalized.split()) <= 3:
        return "simple"
    return "direct"


def route_after_route(state: AgentState) -> str:
    if state.get("status") == "fast_path":
        return "synthesize"
    return "execute"
