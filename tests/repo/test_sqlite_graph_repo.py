from pathlib import Path

from rag.schema.core import GraphEdge, GraphNode
from rag.storage.graph_backends.sqlite_graph_repo import SQLiteGraphRepo


def test_sqlite_graph_repo_tracks_candidates_and_promoted_edges(tmp_path: Path) -> None:
    repo = SQLiteGraphRepo(tmp_path / "graph.sqlite3")
    edge = GraphEdge(
        edge_id="edge-1",
        from_node_id="node-1",
        to_node_id="node-2",
        relation_type="supports",
        confidence=0.8,
        evidence_chunk_ids=["chunk-1"],
    )

    repo.save_candidate_edge(edge)
    assert repo.list_candidate_edges() == [edge]

    repo.promote_candidate_edge(edge.edge_id)
    assert repo.list_candidate_edges() == []
    assert repo.list_edges() == [edge]


def test_sqlite_graph_repo_keeps_node_and_edge_provenance_indexes(tmp_path: Path) -> None:
    repo = SQLiteGraphRepo(tmp_path / "graph.sqlite3")
    node = GraphNode(node_id="entity-1", node_type="entity", label="Revenue")
    edge = GraphEdge(
        edge_id="edge-1",
        from_node_id="entity-1",
        to_node_id="entity-2",
        relation_type="supports",
        confidence=0.91,
        evidence_chunk_ids=["chunk-1", "chunk-2"],
    )

    repo.save_node(node)
    repo.bind_node_evidence(node.node_id, ["chunk-2", "chunk-1"])
    repo.save_candidate_edge(edge)

    assert repo.list_node_evidence_chunk_ids(node.node_id) == ["chunk-1", "chunk-2"]
    assert repo.list_edges_for_node("entity-1", include_candidates=True) == [edge]
    assert repo.list_edges_for_chunk("chunk-1", include_candidates=True) == [edge]


def test_sqlite_graph_repo_accumulates_node_and_edge_evidence_across_writes(tmp_path: Path) -> None:
    repo = SQLiteGraphRepo(tmp_path / "graph.sqlite3")
    node = GraphNode(node_id="entity-alpha", node_type="entity", label="Alpha Engine")
    first_edge = GraphEdge(
        edge_id="edge-supports",
        from_node_id="entity-alpha",
        to_node_id="entity-beta",
        relation_type="supports",
        confidence=0.9,
        evidence_chunk_ids=["chunk-1"],
    )
    second_edge = first_edge.model_copy(update={"evidence_chunk_ids": ["chunk-2"]})

    repo.save_node(node)
    repo.merge_node_evidence(node.node_id, ["chunk-1"])
    repo.merge_node_evidence(node.node_id, ["chunk-2", "chunk-1"])
    repo.save_edge(first_edge)
    repo.save_edge(second_edge)

    merged_edge = repo.get_edge("edge-supports")

    assert repo.list_node_evidence_chunk_ids(node.node_id) == ["chunk-1", "chunk-2"]
    assert merged_edge is not None
    assert merged_edge.evidence_chunk_ids == ["chunk-1", "chunk-2"]


def test_sqlite_graph_repo_indexes_entity_aliases_for_lookup_and_cleanup(tmp_path: Path) -> None:
    repo = SQLiteGraphRepo(tmp_path / "graph.sqlite3")
    node = GraphNode(
        node_id="entity-alpha",
        node_type="entity",
        label="Alpha Engine",
        metadata={"aliases": "AE||Alpha Engine"},
    )

    repo.save_node(node)
    repo.merge_node_evidence(node.node_id, ["chunk-1"])

    assert [item.node_id for item in repo.list_nodes_by_alias("AE", node_type="entity")] == ["entity-alpha"]
    assert [item.node_id for item in repo.list_nodes_by_alias("alpha engine", node_type="entity")] == ["entity-alpha"]

    repo.delete_by_chunk_ids(["chunk-1"])

    assert repo.list_nodes_by_alias("AE", node_type="entity") == []


def test_sqlite_graph_repo_deletes_nodes_and_edges_explicitly(tmp_path: Path) -> None:
    repo = SQLiteGraphRepo(tmp_path / "graph.sqlite3")
    node = GraphNode(node_id="entity-alpha", node_type="entity", label="Alpha Engine")
    edge = GraphEdge(
        edge_id="edge-supports",
        from_node_id="entity-alpha",
        to_node_id="entity-beta",
        relation_type="supports",
        confidence=0.9,
        evidence_chunk_ids=["chunk-1"],
    )

    repo.save_node(node)
    repo.bind_node_evidence(node.node_id, ["chunk-1"])
    repo.save_edge(edge)

    repo.delete_edge(edge.edge_id)
    repo.delete_node(node.node_id)

    assert repo.get_edge(edge.edge_id) is None
    assert repo.get_node(node.node_id) is None
