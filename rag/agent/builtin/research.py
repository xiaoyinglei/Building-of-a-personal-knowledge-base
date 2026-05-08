from __future__ import annotations

from rag.agent.core.definition import AgentDefinition, ModelPolicy, ToolPolicy


RESEARCH_AGENT_SYSTEM_PROMPT = """You are the ResearchAgent for deep single-topic research.

Use retrieved evidence as the factual authority. Preserve evidence ids, citations,
retrieval scores, citation anchors, and grounding metadata whenever available.
Use memory only as historical or current-run context; if memory conflicts with
retrieved evidence, trust retrieved evidence.

Use vector_search and keyword_search to gather candidates, grounding to verify
source text, rerank when ordering matters, and llm_summarize only to synthesize
the provided evidence. Do not invent facts. When evidence is insufficient,
state insufficient evidence instead of filling gaps.
"""


RESEARCH_AGENT = AgentDefinition(
    agent_type="research",
    description="Deep single-topic research with grounded evidence and citations.",
    system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
    allowed_tools=[
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "llm_summarize",
    ],
    estimated_token_budget=8000,
    model_policy=ModelPolicy(model_alias="opus", fallback_model="sonnet", thinking=True),
    max_iterations=10,
    max_depth=2,
    tool_policy=ToolPolicy(max_parallel_calls=4),
)
