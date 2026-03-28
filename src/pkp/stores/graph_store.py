from __future__ import annotations

from dataclasses import dataclass

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.types.content import GraphEdge, GraphNode


@dataclass(slots=True)
class GraphStore:
    graph_repo: SQLiteGraphRepo

    def save_node(self, node: GraphNode, *, evidence_chunk_ids: list[str] | None = None) -> None:
        self.graph_repo.save_node(node)
        if evidence_chunk_ids is not None:
            self.graph_repo.bind_node_evidence(node.node_id, evidence_chunk_ids)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.graph_repo.get_node(node_id)

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]:
        return self.graph_repo.list_nodes(node_type=node_type)

    def list_node_evidence_chunk_ids(self, node_id: str) -> list[str]:
        return self.graph_repo.list_node_evidence_chunk_ids(node_id)

    def save_candidate_edge(self, edge: GraphEdge) -> None:
        self.graph_repo.save_candidate_edge(edge)

    def save_edge(self, edge: GraphEdge) -> None:
        self.graph_repo.save_edge(edge)

    def promote_candidate_edge(self, edge_id: str) -> None:
        self.graph_repo.promote_candidate_edge(edge_id)

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None:
        return self.graph_repo.get_edge(edge_id, include_candidates=include_candidates)

    def list_candidate_edges(self) -> list[GraphEdge]:
        return self.graph_repo.list_candidate_edges()

    def list_edges(self) -> list[GraphEdge]:
        return self.graph_repo.list_edges()

    def list_edges_for_node(self, node_id: str, *, include_candidates: bool = False) -> list[GraphEdge]:
        return self.graph_repo.list_edges_for_node(node_id, include_candidates=include_candidates)

    def list_edges_for_chunk(self, chunk_id: str, *, include_candidates: bool = False) -> list[GraphEdge]:
        return self.graph_repo.list_edges_for_chunk(chunk_id, include_candidates=include_candidates)
