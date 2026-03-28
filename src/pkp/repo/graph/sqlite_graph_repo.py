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
        self._migrate_legacy_nodes()
        self._migrate_legacy_edges("candidate_edges")
        self._migrate_legacy_edges("edges")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_edges (
                edge_id TEXT PRIMARY KEY,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type_label
            ON nodes(node_type, label);

            CREATE INDEX IF NOT EXISTS idx_candidate_edges_from_to
            ON candidate_edges(from_node_id, to_node_id, relation_type);

            CREATE INDEX IF NOT EXISTS idx_edges_from_to
            ON edges(from_node_id, to_node_id, relation_type);

            CREATE TABLE IF NOT EXISTS node_evidence (
                node_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                PRIMARY KEY(node_id, chunk_id)
            );

            CREATE INDEX IF NOT EXISTS idx_node_evidence_chunk
            ON node_evidence(chunk_id, node_id);

            CREATE TABLE IF NOT EXISTS edge_evidence (
                edge_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                PRIMARY KEY(edge_id, chunk_id)
            );

            CREATE INDEX IF NOT EXISTS idx_edge_evidence_chunk
            ON edge_evidence(chunk_id, edge_id);
            """
        )
        self._backfill_edge_evidence("candidate_edges")
        self._backfill_edge_evidence("edges")
        self._conn.commit()

    def _migrate_legacy_nodes(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if columns and ("node_type" not in columns or "label" not in columns):
            self._conn.execute("ALTER TABLE nodes RENAME TO nodes_legacy")
            self._conn.executescript(
                """
                CREATE TABLE nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )
            rows = self._conn.execute(
                "SELECT node_id, saved_at, payload FROM nodes_legacy ORDER BY saved_at, node_id"
            ).fetchall()
            for row in rows:
                node = self._load_node(row["payload"])
                self._conn.execute(
                    """
                    INSERT INTO nodes (node_id, node_type, label, saved_at, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (node.node_id, node.node_type, node.label, row["saved_at"], row["payload"]),
                )
            self._conn.execute("DROP TABLE nodes_legacy")

    def _migrate_legacy_edges(self, table_name: str) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if columns and ("from_node_id" not in columns or "to_node_id" not in columns or "relation_type" not in columns):
            legacy_name = f"{table_name}_legacy"
            self._conn.execute(f"ALTER TABLE {table_name} RENAME TO {legacy_name}")
            self._conn.execute(
                f"""
                CREATE TABLE {table_name} (
                    edge_id TEXT PRIMARY KEY,
                    from_node_id TEXT NOT NULL,
                    to_node_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            rows = self._conn.execute(
                f"SELECT edge_id, saved_at, payload FROM {legacy_name} ORDER BY saved_at, edge_id"
            ).fetchall()
            for row in rows:
                edge = self._load_edge(row["payload"])
                self._conn.execute(
                    f"""
                    INSERT INTO {table_name} (
                        edge_id,
                        from_node_id,
                        to_node_id,
                        relation_type,
                        saved_at,
                        payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.edge_id,
                        edge.from_node_id,
                        edge.to_node_id,
                        edge.relation_type,
                        row["saved_at"],
                        row["payload"],
                    ),
                )
            self._conn.execute(f"DROP TABLE {legacy_name}")

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
            INSERT INTO nodes (node_id, node_type, label, saved_at, payload)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                node_type=excluded.node_type,
                label=excluded.label,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (
                node.node_id,
                node.node_type,
                node.label,
                datetime.now(UTC).isoformat(),
                self._dump(node),
            ),
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> GraphNode | None:
        row = self._conn.execute(
            "SELECT payload FROM nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return None if row is None else self._load_node(row["payload"])

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]:
        if node_type is None:
            rows = self._conn.execute(
                "SELECT payload FROM nodes ORDER BY saved_at, node_id",
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload FROM nodes WHERE node_type = ? ORDER BY saved_at, node_id",
                (node_type,),
            ).fetchall()
        return [self._load_node(row["payload"]) for row in rows]

    def bind_node_evidence(self, node_id: str, chunk_ids: list[str] | tuple[str, ...]) -> None:
        saved_at = datetime.now(UTC).isoformat()
        normalized_ids = tuple(dict.fromkeys(chunk_ids))
        self._conn.execute("DELETE FROM node_evidence WHERE node_id = ?", (node_id,))
        for chunk_id in normalized_ids:
            self._conn.execute(
                """
                INSERT INTO node_evidence (node_id, chunk_id, saved_at)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id, chunk_id) DO UPDATE SET
                    saved_at=excluded.saved_at
                """,
                (node_id, chunk_id, saved_at),
            )
        self._conn.commit()

    def list_node_evidence_chunk_ids(self, node_id: str) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT chunk_id
            FROM node_evidence
            WHERE node_id = ?
            ORDER BY chunk_id
            """,
            (node_id,),
        ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def save_candidate_edge(self, edge: GraphEdge) -> None:
        self._conn.execute(
            """
            INSERT INTO candidate_edges (
                edge_id,
                from_node_id,
                to_node_id,
                relation_type,
                saved_at,
                payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                from_node_id=excluded.from_node_id,
                to_node_id=excluded.to_node_id,
                relation_type=excluded.relation_type,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (
                edge.edge_id,
                edge.from_node_id,
                edge.to_node_id,
                edge.relation_type,
                datetime.now(UTC).isoformat(),
                self._dump(edge),
            ),
        )
        self._replace_edge_evidence(edge.edge_id, edge.evidence_chunk_ids)
        self._conn.commit()

    def promote_candidate_edge(self, edge_id: str) -> None:
        row = self._conn.execute(
            "SELECT payload FROM candidate_edges WHERE edge_id = ?",
            (edge_id,),
        ).fetchone()
        if row is None:
            return
        self._conn.execute("DELETE FROM candidate_edges WHERE edge_id = ?", (edge_id,))
        edge = self._load_edge(row["payload"])
        self._conn.execute(
            """
            INSERT INTO edges (
                edge_id,
                from_node_id,
                to_node_id,
                relation_type,
                saved_at,
                payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                from_node_id=excluded.from_node_id,
                to_node_id=excluded.to_node_id,
                relation_type=excluded.relation_type,
                saved_at=excluded.saved_at,
                payload=excluded.payload
            """,
            (
                edge_id,
                edge.from_node_id,
                edge.to_node_id,
                edge.relation_type,
                datetime.now(UTC).isoformat(),
                row["payload"],
            ),
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

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None:
        tables = ["edges", "candidate_edges"] if include_candidates else ["edges"]
        for table_name in tables:
            row = self._conn.execute(
                f"SELECT payload FROM {table_name} WHERE edge_id = ?",
                (edge_id,),
            ).fetchone()
            if row is not None:
                return self._load_edge(row["payload"])
        return None

    def list_edges_for_node(self, node_id: str, *, include_candidates: bool = False) -> list[GraphEdge]:
        table_names = ["edges", "candidate_edges"] if include_candidates else ["edges"]
        loaded: list[GraphEdge] = []
        for table_name in table_names:
            rows = self._conn.execute(
                f"""
                SELECT payload
                FROM {table_name}
                WHERE from_node_id = ? OR to_node_id = ?
                ORDER BY saved_at, edge_id
                """,
                (node_id, node_id),
            ).fetchall()
            loaded.extend(self._load_edge(row["payload"]) for row in rows)
        return loaded

    def list_edges_for_chunk(self, chunk_id: str, *, include_candidates: bool = False) -> list[GraphEdge]:
        table_names = ["edges", "candidate_edges"] if include_candidates else ["edges"]
        loaded: list[GraphEdge] = []
        for table_name in table_names:
            rows = self._conn.execute(
                f"""
                SELECT {table_name}.payload
                FROM {table_name}
                INNER JOIN edge_evidence
                  ON edge_evidence.edge_id = {table_name}.edge_id
                WHERE edge_evidence.chunk_id = ?
                ORDER BY {table_name}.saved_at, {table_name}.edge_id
                """,
                (chunk_id,),
            ).fetchall()
            loaded.extend(self._load_edge(row["payload"]) for row in rows)
        return loaded

    def _replace_edge_evidence(self, edge_id: str, chunk_ids: list[str]) -> None:
        saved_at = datetime.now(UTC).isoformat()
        normalized_ids = tuple(dict.fromkeys(chunk_ids))
        self._conn.execute("DELETE FROM edge_evidence WHERE edge_id = ?", (edge_id,))
        for chunk_id in normalized_ids:
            self._conn.execute(
                """
                INSERT INTO edge_evidence (edge_id, chunk_id, saved_at)
                VALUES (?, ?, ?)
                ON CONFLICT(edge_id, chunk_id) DO UPDATE SET
                    saved_at=excluded.saved_at
                """,
                (edge_id, chunk_id, saved_at),
            )

    def _backfill_edge_evidence(self, table_name: str) -> None:
        rows = self._conn.execute(
            f"SELECT edge_id, payload FROM {table_name} ORDER BY edge_id",
        ).fetchall()
        for row in rows:
            existing = self._conn.execute(
                "SELECT 1 FROM edge_evidence WHERE edge_id = ? LIMIT 1",
                (row["edge_id"],),
            ).fetchone()
            if existing is not None:
                continue
            edge = self._load_edge(row["payload"])
            self._replace_edge_evidence(edge.edge_id, edge.evidence_chunk_ids)

    def close(self) -> None:
        self._conn.close()
