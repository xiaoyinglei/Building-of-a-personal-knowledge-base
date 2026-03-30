from __future__ import annotations

from pathlib import Path

from rag import RAG, StorageConfig
from rag.ingest.ingest import DirectContentItem
from rag.query.query import QueryOptions
from rag.schema._types.content import GraphEdge, GraphNode


def test_rag_exposes_insert_query_delete_rebuild() -> None:
    core = RAG(storage=StorageConfig.in_memory())

    assert hasattr(core, "insert")
    assert hasattr(core, "query")
    assert hasattr(core, "delete")
    assert hasattr(core, "rebuild")


def test_rag_exposes_batch_ingest_and_custom_kg_operations() -> None:
    core = RAG(storage=StorageConfig.in_memory())

    assert hasattr(core, "insert_many")
    assert hasattr(core, "insert_content_list")
    assert hasattr(core, "insert_custom_kg")
    assert hasattr(core, "upsert_node")
    assert hasattr(core, "upsert_edge")
    assert hasattr(core, "delete_node")
    assert hasattr(core, "delete_edge")


def test_query_options_defaults_to_mix_mode() -> None:
    options = QueryOptions()

    assert options.mode == "mix"


def test_query_options_accepts_bypass_mode() -> None:
    options = QueryOptions(mode="bypass")

    assert options.mode == "bypass"


def test_rag_bypass_mode_uses_direct_llm_path_without_retrieval() -> None:
    core = RAG(storage=StorageConfig.in_memory())
    try:
        result = core.query(
            "直接回答：Alpha Engine 是什么？",
            options=QueryOptions(
                mode="bypass",
                response_type="Single Paragraph",
                user_prompt="简洁回答，不要引用。",
            ),
        )

        assert result.mode == "bypass"
        assert result.retrieval.evidence.internal == []
        assert result.retrieval.reranked_chunk_ids == []
        assert "Alpha Engine" in result.context.prompt
        assert result.context.evidence == []
    finally:
        core.stores.close()


def test_storage_config_accepts_string_root(tmp_path: Path) -> None:
    root = tmp_path / ".rag"

    core = RAG(storage=StorageConfig(root=str(root)))

    assert core.stores.root == root


def test_rag_custom_kg_operations_round_trip() -> None:
    core = RAG(storage=StorageConfig.in_memory())
    try:
        node = GraphNode(node_id="entity-alpha", node_type="entity", label="Alpha Engine")
        edge = GraphEdge(
            edge_id="edge-supports",
            from_node_id="entity-alpha",
            to_node_id="entity-beta",
            relation_type="supports",
            confidence=0.9,
            evidence_chunk_ids=["chunk-1"],
        )

        core.upsert_node(node, evidence_chunk_ids=["chunk-1"])
        core.upsert_edge(edge)

        assert core.list_nodes(node_type="entity") == [node]
        assert core.get_node(node.node_id) == node
        assert core.get_edge(edge.edge_id) == edge

        result = core.insert_custom_kg(
            nodes=[GraphNode(node_id="entity-beta", node_type="entity", label="Beta Service")],
            edges=[edge],
        )

        assert result["node_count"] == 1
        assert result["edge_count"] == 1

        core.delete_edge(edge.edge_id)
        core.delete_node(node.node_id)

        assert core.get_edge(edge.edge_id) is None
        assert core.get_node(node.node_id) is None
    finally:
        core.stores.close()


def test_rag_insert_content_list_normalizes_mixed_content_inputs(tmp_path: Path) -> None:
    core = RAG(storage=StorageConfig.in_memory())
    try:
        markdown_path = tmp_path / "note.md"
        markdown_path.write_text("# Alpha Note\n\nAlpha Engine supports Beta Service.\n", encoding="utf-8")

        result = core.insert_content_list(
            [
                DirectContentItem(
                    location="memory://alpha.txt",
                    source_type="plain_text",
                    content="Alpha Engine supports Beta Service.",
                ),
                DirectContentItem(
                    location=str(markdown_path),
                    source_type="markdown",
                    content=markdown_path,
                ),
                DirectContentItem(
                    location="https://example.com/article",
                    source_type="web",
                    content=(
                        "<html><body><article><h1>Alpha Web</h1>"
                        "<p>Alpha Engine overview.</p></article></body></html>"
                    ),
                ),
            ]
        )

        assert result.success_count == 3
        assert result.failure_count == 0
        assert len(result.results) == 3
        assert {item.source.source_type.value for item in result.results} == {"plain_text", "markdown", "web"}
    finally:
        core.stores.close()
