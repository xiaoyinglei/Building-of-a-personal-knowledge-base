from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes.evaluate import evaluate_node, route_after_evaluate
from rag.agent.graphs.nodes.execute import execute_node
from rag.agent.graphs.nodes.observe import observe_node
from rag.agent.graphs.nodes.route import route_after_route, route_node
from rag.agent.graphs.nodes.synthesize import synthesize_node
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry


def build_agent_graph(
    *,
    definition: AgentDefinition,
    tool_registry: ToolRegistry,
):
    graph = StateGraph(AgentState)

    async def bound_execute_node(state: AgentState) -> dict:
        return await execute_node(state, tool_registry=tool_registry)

    async def bound_evaluate_node(state: AgentState) -> dict:
        return await evaluate_node(state, definition=definition)

    graph.add_node("route", route_node)
    graph.add_node("execute", bound_execute_node)
    graph.add_node("observe", observe_node)
    graph.add_node("evaluate", bound_evaluate_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_after_route,
        {
            "execute": "execute",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("execute", "observe")
    graph.add_edge("observe", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "execute": "execute",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("synthesize", END)

    return graph.compile()
