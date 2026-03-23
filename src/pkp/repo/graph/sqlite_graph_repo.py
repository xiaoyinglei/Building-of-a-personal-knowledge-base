from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pkp.types.content import GraphEdge, GraphNode


class SQLiteGraphRepo:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_edges (
                edge_id TEXT PRIMARY KEY,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    @staticmethod
    def _dump(model: GraphNode | GraphEdge) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=True)

    @staticmethod
    def _load_node(payload: str) -> GraphNode:
        return GraphNode.model_validate(json.loads(payload))

    @staticmethod
    def _load_edge(payload: str) -> GraphEdge:
        return GraphEdge.model_validate(json.loads(payload))

    def save_node(self, node: GraphNode) -> None:
        self._conn.execute(
            """
            INSERT INTO nodes (node_id, saved_at, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (node.node_id, datetime.now(UTC).isoformat(), self._dump(node)),
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> GraphNode | None:
        row = self._conn.execute(
            "SELECT payload FROM nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return None if row is None else self._load_node(row["payload"])

    def list_nodes(self) -> list[GraphNode]:
        rows = self._conn.execute(
            "SELECT payload FROM nodes ORDER BY saved_at, node_id",
        ).fetchall()
        return [self._load_node(row["payload"]) for row in rows]

    def save_candidate_edge(self, edge: GraphEdge) -> None:
        self._conn.execute(
            """
            INSERT INTO candidate_edges (edge_id, saved_at, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (edge.edge_id, datetime.now(UTC).isoformat(), self._dump(edge)),
        )
        self._conn.commit()

    def promote_candidate_edge(self, edge_id: str) -> None:
        row = self._conn.execute(
            "SELECT payload FROM candidate_edges WHERE edge_id = ?",
            (edge_id,),
        ).fetchone()
        if row is None:
            return
        self._conn.execute("DELETE FROM candidate_edges WHERE edge_id = ?", (edge_id,))
        self._conn.execute(
            """
            INSERT INTO edges (edge_id, saved_at, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (edge_id, datetime.now(UTC).isoformat(), row["payload"]),
        )
        self._conn.commit()

    def list_candidate_edges(self) -> list[GraphEdge]:
        rows = self._conn.execute(
            "SELECT payload FROM candidate_edges ORDER BY saved_at, edge_id",
        ).fetchall()
        return [self._load_edge(row["payload"]) for row in rows]

    def list_edges(self) -> list[GraphEdge]:
        rows = self._conn.execute(
            "SELECT payload FROM edges ORDER BY saved_at, edge_id",
        ).fetchall()
        return [self._load_edge(row["payload"]) for row in rows]
