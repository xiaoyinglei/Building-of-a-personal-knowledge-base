from __future__ import annotations

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.core.compiler import AgentGraphCompiler
from rag.agent.tools.llm_tools import ALL_LLM_TOOLS
from rag.agent.tools.rag_tools import ALL_RAG_TOOLS
from rag.agent.tools.registry import ToolRegistry


def _registry_with_builtin_tools() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [*ALL_RAG_TOOLS, *ALL_LLM_TOOLS]:
        registry.register(tool)
    return registry


def test_research_agent_uses_spec_tool_allowlist() -> None:
    assert RESEARCH_AGENT.agent_type == "research"
    assert RESEARCH_AGENT.allowed_tools == [
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "llm_summarize",
    ]


def test_research_agent_prompt_requires_grounded_citations() -> None:
    prompt = RESEARCH_AGENT.system_prompt

    assert "retrieved evidence" in prompt
    assert "citations" in prompt
    assert "insufficient evidence" in prompt


def test_research_agent_compiles_when_builtin_tools_registered() -> None:
    compiler = AgentGraphCompiler(tool_registry=_registry_with_builtin_tools())

    graph = compiler.compile(RESEARCH_AGENT)

    assert hasattr(graph, "ainvoke")
