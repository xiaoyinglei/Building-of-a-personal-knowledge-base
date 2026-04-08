from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, cast

from rag.schema.core import GraphEdge, GraphNode


class Neo4jGraphRepo:
    def __init__(
        self,
        dsn: str,
        *,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self._dsn = dsn
        self._username = username
        self._password = password
        self._database = database
        self._driver: Any = self._connect()
        self._ensure_schema()

    def save_node(self, node: GraphNode) -> None:
        existing = self.get_node(node.node_id)
        merged = (
            node
            if existing is None
            else node.model_copy(update={"metadata": self._merge_metadata(existing.metadata, node.metadata)})
        )
        aliases = self._node_alias_values(merged)
        normalized_aliases = [normalized for alias in aliases if (normalized := self._normalize_alias(alias))]
        self._execute_write(
            """
            MERGE (n:RAGNode {node_id: $node_id})
            SET n.node_type = $node_type,
                n.label = $label,
                n.metadata_json = $metadata_json,
                n.aliases = $aliases,
                n.normalized_aliases = $normalized_aliases,
                n.saved_at = $saved_at
            """,
            {
                "node_id": merged.node_id,
                "node_type": merged.node_type,
                "label": merged.label,
                "metadata_json": self._dump(merged),
                "aliases": aliases,
                "normalized_aliases": normalized_aliases,
                "saved_at": datetime.now(UTC).isoformat(),
            },
        )

    def merge_node_evidence(self, node_id: str, chunk_ids: list[str] | tuple[str, ...]) -> None:
        merged = [*self.list_node_evidence_chunk_ids(node_id), *list(chunk_ids)]
        self.bind_node_evidence(node_id, list(dict.fromkeys(merged)))

    def bind_node_evidence(self, node_id: str, chunk_ids: list[str] | tuple[str, ...]) -> None:
        normalized = list(dict.fromkeys(chunk_ids))
        self._execute_write(
            """
            MERGE (n:RAGNode {node_id: $node_id})
            ON CREATE SET n.node_type = coalesce(n.node_type, 'unknown'),
                          n.label = coalesce(n.label, $node_id),
                          n.metadata_json = coalesce(n.metadata_json, '{}'),
                          n.aliases = coalesce(n.aliases, [$node_id]),
                          n.normalized_aliases = coalesce(n.normalized_aliases, [$normalized_node_id])
            SET n.evidence_chunk_ids = $chunk_ids,
                n.saved_at = $saved_at
            """,
            {
                "node_id": node_id,
                "normalized_node_id": self._normalize_alias(node_id) or node_id,
                "chunk_ids": normalized,
                "saved_at": datetime.now(UTC).isoformat(),
            },
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        row = self._execute_read_one(
            """
            MATCH (n:RAGNode {node_id: $node_id})
            RETURN n.node_id AS node_id, n.node_type AS node_type, n.label AS label, n.metadata_json AS metadata_json
            """,
            {"node_id": node_id},
        )
        return None if row is None else self._load_node(row)

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]:
        if node_type is None:
            rows = self._execute_read(
                """
                MATCH (n:RAGNode)
                RETURN
                    n.node_id AS node_id,
                    n.node_type AS node_type,
                    n.label AS label,
                    n.metadata_json AS metadata_json
                ORDER BY n.saved_at, n.node_id
                """,
                {},
            )
        else:
            rows = self._execute_read(
                """
                MATCH (n:RAGNode)
                WHERE n.node_type = $node_type
                RETURN
                    n.node_id AS node_id,
                    n.node_type AS node_type,
                    n.label AS label,
                    n.metadata_json AS metadata_json
                ORDER BY n.saved_at, n.node_id
                """,
                {"node_type": node_type},
            )
        return [self._load_node(row) for row in rows]

    def list_nodes_by_alias(self, alias: str, *, node_type: str | None = None) -> list[GraphNode]:
        normalized = self._normalize_alias(alias)
        if not normalized:
            return []
        params: dict[str, object] = {"alias": normalized}
        where = "$alias IN coalesce(n.normalized_aliases, [])"
        if node_type is not None:
            where += " AND n.node_type = $node_type"
            params["node_type"] = node_type
        rows = self._execute_read(
            f"""
            MATCH (n:RAGNode)
            WHERE {where}
            RETURN n.node_id AS node_id, n.node_type AS node_type, n.label AS label, n.metadata_json AS metadata_json
            ORDER BY n.saved_at, n.node_id
            """,
            params,
        )
        return [self._load_node(row) for row in rows]

    def list_nodes_for_chunk(self, chunk_id: str) -> list[GraphNode]:
        rows = self._execute_read(
            """
            MATCH (n:RAGNode)
            WHERE $chunk_id IN coalesce(n.evidence_chunk_ids, [])
            RETURN n.node_id AS node_id, n.node_type AS node_type, n.label AS label, n.metadata_json AS metadata_json
            ORDER BY n.saved_at, n.node_id
            """,
            {"chunk_id": chunk_id},
        )
        return [self._load_node(row) for row in rows]

    def list_node_evidence_chunk_ids(self, node_id: str) -> list[str]:
        row = self._execute_read_one(
            "MATCH (n:RAGNode {node_id: $node_id}) RETURN coalesce(n.evidence_chunk_ids, []) AS chunk_ids",
            {"node_id": node_id},
        )
        if row is None:
            return []
        return [str(item) for item in cast(list[str], row["chunk_ids"])]

    def save_candidate_edge(self, edge: GraphEdge) -> None:
        self._save_edge(edge, candidate=True)

    def save_edge(self, edge: GraphEdge) -> None:
        self._save_edge(edge, candidate=False)

    def promote_candidate_edge(self, edge_id: str) -> None:
        self._execute_write(
            """
            MATCH ()-[r:RAG_EDGE {edge_id: $edge_id}]->()
            SET r.candidate = FALSE, r.saved_at = $saved_at
            """,
            {"edge_id": edge_id, "saved_at": datetime.now(UTC).isoformat()},
        )

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None:
        rows = self._execute_read(
            """
            MATCH (from:RAGNode)-[r:RAG_EDGE {edge_id: $edge_id}]->(to:RAGNode)
            WHERE $include_candidates OR coalesce(r.candidate, FALSE) = FALSE
            RETURN
                r.edge_id AS edge_id,
                from.node_id AS from_node_id,
                to.node_id AS to_node_id,
                r.relation_type AS relation_type,
                r.confidence AS confidence,
                coalesce(r.evidence_chunk_ids, []) AS evidence_chunk_ids,
                r.metadata_json AS metadata_json
            """,
            {"edge_id": edge_id, "include_candidates": include_candidates},
        )
        if not rows:
            return None
        return self._load_edge(rows[0])

    def list_candidate_edges(self) -> list[GraphEdge]:
        return self._list_edges(candidate=True)

    def list_edges(self) -> list[GraphEdge]:
        return self._list_edges(candidate=False)

    def list_edges_for_node(self, node_id: str, *, include_candidates: bool = False) -> list[GraphEdge]:
        rows = self._execute_read(
            """
            MATCH (from:RAGNode)-[r:RAG_EDGE]->(to:RAGNode)
            WHERE (from.node_id = $node_id OR to.node_id = $node_id)
              AND ($include_candidates OR coalesce(r.candidate, FALSE) = FALSE)
            RETURN
                r.edge_id AS edge_id,
                from.node_id AS from_node_id,
                to.node_id AS to_node_id,
                r.relation_type AS relation_type,
                r.confidence AS confidence,
                coalesce(r.evidence_chunk_ids, []) AS evidence_chunk_ids,
                r.metadata_json AS metadata_json
            ORDER BY r.saved_at, r.edge_id
            """,
            {"node_id": node_id, "include_candidates": include_candidates},
        )
        return [self._load_edge(row) for row in rows]

    def list_edges_for_chunk(self, chunk_id: str, *, include_candidates: bool = False) -> list[GraphEdge]:
        rows = self._execute_read(
            """
            MATCH (from:RAGNode)-[r:RAG_EDGE]->(to:RAGNode)
            WHERE $chunk_id IN coalesce(r.evidence_chunk_ids, [])
              AND ($include_candidates OR coalesce(r.candidate, FALSE) = FALSE)
            RETURN
                r.edge_id AS edge_id,
                from.node_id AS from_node_id,
                to.node_id AS to_node_id,
                r.relation_type AS relation_type,
                r.confidence AS confidence,
                coalesce(r.evidence_chunk_ids, []) AS evidence_chunk_ids,
                r.metadata_json AS metadata_json
            ORDER BY r.saved_at, r.edge_id
            """,
            {"chunk_id": chunk_id, "include_candidates": include_candidates},
        )
        return [self._load_edge(row) for row in rows]

    def delete_node(self, node_id: str) -> None:
        self._execute_write("MATCH (n:RAGNode {node_id: $node_id}) DETACH DELETE n", {"node_id": node_id})

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None:
        self._execute_write(
            """
            MATCH ()-[r:RAG_EDGE {edge_id: $edge_id}]->()
            WHERE $include_candidates OR coalesce(r.candidate, FALSE) = FALSE
            DELETE r
            """,
            {"edge_id": edge_id, "include_candidates": include_candidates},
        )

    def delete_by_chunk_ids(self, chunk_ids: list[str] | tuple[str, ...]) -> tuple[list[str], list[str]]:
        normalized = list(dict.fromkeys(chunk_ids))
        if not normalized:
            return ([], [])

        affected_edges = self._execute_read(
            """
            MATCH (from:RAGNode)-[r:RAG_EDGE]->(to:RAGNode)
            WHERE any(chunk_id IN coalesce(r.evidence_chunk_ids, []) WHERE chunk_id IN $chunk_ids)
            RETURN
                r.edge_id AS edge_id,
                from.node_id AS from_node_id,
                to.node_id AS to_node_id,
                r.relation_type AS relation_type,
                r.confidence AS confidence,
                coalesce(r.evidence_chunk_ids, []) AS evidence_chunk_ids,
                r.metadata_json AS metadata_json,
                coalesce(r.candidate, FALSE) AS candidate
            """,
            {"chunk_ids": normalized},
        )
        deleted_edge_ids: list[str] = []
        for row in affected_edges:
            remaining = [
                chunk_id
                for chunk_id in cast(list[str], row["evidence_chunk_ids"])
                if chunk_id not in normalized
            ]
            edge_id = str(row["edge_id"])
            if remaining:
                self._execute_write(
                    """
                    MATCH ()-[r:RAG_EDGE {edge_id: $edge_id}]->()
                    SET r.evidence_chunk_ids = $evidence_chunk_ids, r.saved_at = $saved_at
                    """,
                    {
                        "edge_id": edge_id,
                        "evidence_chunk_ids": remaining,
                        "saved_at": datetime.now(UTC).isoformat(),
                    },
                )
                continue
            self.delete_edge(edge_id, include_candidates=True)
            deleted_edge_ids.append(edge_id)

        affected_nodes = self._execute_read(
            """
            MATCH (n:RAGNode)
            WHERE any(chunk_id IN coalesce(n.evidence_chunk_ids, []) WHERE chunk_id IN $chunk_ids)
            RETURN
                n.node_id AS node_id,
                coalesce(n.evidence_chunk_ids, []) AS evidence_chunk_ids
            """,
            {"chunk_ids": normalized},
        )
        deleted_node_ids: list[str] = []
        for row in affected_nodes:
            node_id = str(row["node_id"])
            remaining = [
                chunk_id
                for chunk_id in cast(list[str], row["evidence_chunk_ids"])
                if chunk_id not in normalized
            ]
            if remaining:
                self._execute_write(
                    """
                    MATCH (n:RAGNode {node_id: $node_id})
                    SET n.evidence_chunk_ids = $evidence_chunk_ids, n.saved_at = $saved_at
                    """,
                    {
                        "node_id": node_id,
                        "evidence_chunk_ids": remaining,
                        "saved_at": datetime.now(UTC).isoformat(),
                    },
                )
                continue
            dangling_edges = self.list_edges_for_node(node_id, include_candidates=True)
            deleted_edge_ids.extend(edge.edge_id for edge in dangling_edges)
            self.delete_node(node_id)
            deleted_node_ids.append(node_id)

        return (list(dict.fromkeys(deleted_node_ids)), list(dict.fromkeys(deleted_edge_ids)))

    def close(self) -> None:
        self._driver.close()

    def _save_edge(self, edge: GraphEdge, *, candidate: bool) -> None:
        existing = self.get_edge(edge.edge_id, include_candidates=True)
        merged = (
            edge
            if existing is None
            else edge.model_copy(
                update={
                    "confidence": max(existing.confidence, edge.confidence),
                    "evidence_chunk_ids": list(dict.fromkeys([*existing.evidence_chunk_ids, *edge.evidence_chunk_ids])),
                    "metadata": self._merge_metadata(existing.metadata, edge.metadata),
                }
            )
        )
        self._execute_write(
            """
            MERGE (from:RAGNode {node_id: $from_node_id})
            ON CREATE SET from.node_type = coalesce(from.node_type, 'unknown'),
                          from.label = coalesce(from.label, $from_node_id),
                          from.metadata_json = coalesce(from.metadata_json, '{}'),
                          from.aliases = coalesce(from.aliases, [$from_node_id]),
                          from.normalized_aliases = coalesce(from.normalized_aliases, [$normalized_from_node_id])
            MERGE (to:RAGNode {node_id: $to_node_id})
            ON CREATE SET to.node_type = coalesce(to.node_type, 'unknown'),
                          to.label = coalesce(to.label, $to_node_id),
                          to.metadata_json = coalesce(to.metadata_json, '{}'),
                          to.aliases = coalesce(to.aliases, [$to_node_id]),
                          to.normalized_aliases = coalesce(to.normalized_aliases, [$normalized_to_node_id])
            MERGE (from)-[r:RAG_EDGE {edge_id: $edge_id}]->(to)
            SET r.relation_type = $relation_type,
                r.confidence = $confidence,
                r.evidence_chunk_ids = $evidence_chunk_ids,
                r.metadata_json = $metadata_json,
                r.candidate = $candidate,
                r.saved_at = $saved_at
            """,
            {
                "edge_id": merged.edge_id,
                "from_node_id": merged.from_node_id,
                "to_node_id": merged.to_node_id,
                "normalized_from_node_id": self._normalize_alias(merged.from_node_id) or merged.from_node_id,
                "normalized_to_node_id": self._normalize_alias(merged.to_node_id) or merged.to_node_id,
                "relation_type": merged.relation_type,
                "confidence": float(merged.confidence),
                "evidence_chunk_ids": list(dict.fromkeys(merged.evidence_chunk_ids)),
                "metadata_json": self._dump(merged),
                "candidate": candidate,
                "saved_at": datetime.now(UTC).isoformat(),
            },
        )

    def _list_edges(self, *, candidate: bool) -> list[GraphEdge]:
        rows = self._execute_read(
            """
            MATCH (from:RAGNode)-[r:RAG_EDGE]->(to:RAGNode)
            WHERE coalesce(r.candidate, FALSE) = $candidate
            RETURN
                r.edge_id AS edge_id,
                from.node_id AS from_node_id,
                to.node_id AS to_node_id,
                r.relation_type AS relation_type,
                r.confidence AS confidence,
                coalesce(r.evidence_chunk_ids, []) AS evidence_chunk_ids,
                r.metadata_json AS metadata_json
            ORDER BY r.saved_at, r.edge_id
            """,
            {"candidate": candidate},
        )
        return [self._load_edge(row) for row in rows]

    def _ensure_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT rag_node_id IF NOT EXISTS FOR (n:RAGNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE INDEX rag_node_aliases IF NOT EXISTS FOR (n:RAGNode) ON (n.normalized_aliases)",
            "CREATE INDEX rag_edge_id IF NOT EXISTS FOR ()-[r:RAG_EDGE]-() ON (r.edge_id)",
        )
        for statement in statements:
            self._execute_write(statement, {})

    def _connect(self) -> Any:
        from neo4j import GraphDatabase

        auth = None
        if self._username is not None and self._password is not None:
            auth = (self._username, self._password)
        return GraphDatabase.driver(self._dsn, auth=auth)

    def _execute_write(self, query: str, parameters: dict[str, object]) -> None:
        with cast(Any, self._driver).session(database=self._database) as session:
            session.execute_write(lambda tx: tx.run(query, parameters).consume())

    def _execute_read(self, query: str, parameters: dict[str, object]) -> list[dict[str, Any]]:
        with cast(Any, self._driver).session(database=self._database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query, parameters)))
        return [cast(dict[str, Any], record.data()) for record in cast(list[Any], result)]

    def _execute_read_one(self, query: str, parameters: dict[str, object]) -> dict[str, Any] | None:
        rows = self._execute_read(query, parameters)
        return rows[0] if rows else None

    @staticmethod
    def _dump(model: GraphNode | GraphEdge) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=True)

    @staticmethod
    def _load_node(row: dict[str, Any]) -> GraphNode:
        metadata_json = str(row.get("metadata_json", "{}"))
        payload = json.loads(metadata_json)
        if isinstance(payload, dict) and {"node_id", "node_type", "label"} <= payload.keys():
            return GraphNode.model_validate(payload)
        return GraphNode(
            node_id=str(row["node_id"]),
            node_type=str(row["node_type"]),
            label=str(row["label"]),
            metadata=cast(dict[str, str], payload if isinstance(payload, dict) else {}),
        )

    @staticmethod
    def _load_edge(row: dict[str, Any]) -> GraphEdge:
        metadata_json = str(row.get("metadata_json", "{}"))
        payload = json.loads(metadata_json)
        if isinstance(payload, dict) and {"edge_id", "from_node_id", "to_node_id", "relation_type"} <= payload.keys():
            return GraphEdge.model_validate(payload)
        return GraphEdge(
            edge_id=str(row["edge_id"]),
            from_node_id=str(row["from_node_id"]),
            to_node_id=str(row["to_node_id"]),
            relation_type=str(row["relation_type"]),
            confidence=float(row["confidence"]),
            evidence_chunk_ids=[str(item) for item in cast(list[str], row["evidence_chunk_ids"])],
            metadata=cast(dict[str, str], payload if isinstance(payload, dict) else {}),
        )

    @staticmethod
    def _merge_metadata(existing: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
        merged = dict(existing)
        merged.update(incoming)
        for scalar_key, plural_key in (("doc_id", "doc_ids"), ("source_id", "source_ids")):
            values: list[str] = []
            for candidate in (existing.get(scalar_key), incoming.get(scalar_key)):
                if candidate:
                    values.append(candidate)
            for candidate in (existing.get(plural_key), incoming.get(plural_key)):
                if candidate:
                    values.extend(item.strip() for item in candidate.split(",") if item.strip())
            if values:
                merged[plural_key] = ",".join(sorted(dict.fromkeys(values)))
        aliases: list[str] = []
        for container in (existing, incoming):
            alias_blob = container.get("aliases")
            if alias_blob:
                aliases.extend(alias for alias in alias_blob.split("||") if alias)
        if aliases:
            merged["aliases"] = "||".join(sorted(dict.fromkeys(aliases)))
        return merged

    @staticmethod
    def _node_alias_values(node: GraphNode) -> list[str]:
        aliases = [node.label]
        alias_blob = node.metadata.get("aliases", "")
        if alias_blob:
            aliases.extend(alias.strip() for alias in alias_blob.split("||") if alias.strip())
        return list(dict.fromkeys(alias for alias in aliases if alias.strip()))

    @staticmethod
    def _normalize_alias(alias: str) -> str:
        normalized = re.sub(r"\s+", " ", alias.strip().lower())
        normalized = re.sub(r"[^\w\s\-./:]+", "", normalized)
        return normalized.strip()
