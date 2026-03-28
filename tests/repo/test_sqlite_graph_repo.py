from pathlib import Path

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.types.content import GraphEdge, GraphNode


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
