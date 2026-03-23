from pathlib import Path

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.types.content import GraphEdge


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
