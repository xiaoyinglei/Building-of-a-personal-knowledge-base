from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pkp.schema._types.memory import EpisodicMemory, UserMemory


class SQLiteMemoryRepo:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_user_memories_user_key
            ON user_memories(user_id, preference_key, updated_at DESC);

            CREATE TABLE IF NOT EXISTS episodic_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodic_memories_user_updated
            ON episodic_memories(user_id, updated_at DESC);
            """
        )
        self._conn.commit()

    @staticmethod
    def _dump(model: UserMemory | EpisodicMemory) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=True)

    def save_user_memory(self, memory: UserMemory) -> None:
        self._conn.execute(
            """
            INSERT INTO user_memories (
                memory_id,
                user_id,
                preference_key,
                updated_at,
                payload
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                user_id=excluded.user_id,
                preference_key=excluded.preference_key,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                memory.memory_id,
                memory.user_id,
                memory.preference_key,
                memory.updated_at.isoformat(),
                self._dump(memory),
            ),
        )
        self._conn.commit()

    def save_episodic_memory(self, memory: EpisodicMemory) -> None:
        self._conn.execute(
            """
            INSERT INTO episodic_memories (
                memory_id,
                user_id,
                session_id,
                query,
                updated_at,
                payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                user_id=excluded.user_id,
                session_id=excluded.session_id,
                query=excluded.query,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                memory.memory_id,
                memory.user_id,
                memory.session_id,
                memory.query,
                memory.updated_at.isoformat(),
                self._dump(memory),
            ),
        )
        self._conn.commit()

    def get_user_memory(self, memory_id: str) -> UserMemory | None:
        row = self._conn.execute(
            "SELECT payload FROM user_memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        return None if row is None else UserMemory.model_validate(json.loads(row["payload"]))

    def get_episodic_memory(self, memory_id: str) -> EpisodicMemory | None:
        row = self._conn.execute(
            "SELECT payload FROM episodic_memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        return None if row is None else EpisodicMemory.model_validate(json.loads(row["payload"]))

    def list_user_memories(self, user_id: str) -> list[UserMemory]:
        rows = self._conn.execute(
            """
            SELECT payload
            FROM user_memories
            WHERE user_id = ?
            ORDER BY updated_at DESC, memory_id DESC
            """,
            (user_id,),
        ).fetchall()
        return [UserMemory.model_validate(json.loads(row["payload"])) for row in rows]

    def list_episodic_memories(
        self,
        user_id: str,
        *,
        source_scope: list[str] | None = None,
    ) -> list[EpisodicMemory]:
        rows = self._conn.execute(
            """
            SELECT payload
            FROM episodic_memories
            WHERE user_id = ?
            ORDER BY updated_at DESC, memory_id DESC
            """,
            (user_id,),
        ).fetchall()
        memories = [EpisodicMemory.model_validate(json.loads(row["payload"])) for row in rows]
        if not source_scope:
            return memories
        allowed = set(source_scope)
        return [memory for memory in memories if allowed.intersection(memory.source_scope)]
