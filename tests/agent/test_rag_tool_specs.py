from __future__ import annotations

from rag.agent.tools.rag_tools import ALL_RAG_TOOLS


def test_all_rag_tools_contains_expected_specs() -> None:
    by_name = {tool.name: tool for tool in ALL_RAG_TOOLS}
    assert set(by_name) == {
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "graph_expand",
    }


def test_rag_tool_permissions_match_contract() -> None:
    by_name = {tool.name: tool for tool in ALL_RAG_TOOLS}
    assert by_name["vector_search"].permissions.read_db is True
    assert by_name["vector_search"].permissions.embed is True
    assert by_name["keyword_search"].permissions.read_db is True
    assert by_name["grounding"].permissions.read_object_store is True
    assert by_name["rerank"].permissions.generate is True
    assert by_name["graph_expand"].permissions.read_db is True
