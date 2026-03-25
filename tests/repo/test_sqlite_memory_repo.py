from datetime import UTC, datetime
from pathlib import Path

from pkp.repo.storage.sqlite_memory_repo import SQLiteMemoryRepo
from pkp.types.memory import EpisodicMemory, MemoryEvidenceLink, UserMemory


def _evidence(chunk_id: str, doc_id: str) -> MemoryEvidenceLink:
    return MemoryEvidenceLink(
        chunk_id=chunk_id,
        doc_id=doc_id,
        citation_anchor=f"{doc_id}#{chunk_id}",
    )


def _timestamp(hour: int) -> datetime:
    return datetime(2026, 3, 25, hour, 0, tzinfo=UTC)


def _user_memory(updated_at: datetime) -> UserMemory:
    return UserMemory(
        memory_id="pref-answer-style",
        user_id="user-1",
        preference_key="answer_style",
        preference_summary="Prefer terse bullet answers.",
        evidence=[_evidence("chunk-a", "doc-a")],
        source_scope=["doc-a"],
        reliability=0.9,
        created_at=_timestamp(9),
        updated_at=updated_at,
    )


def _episode(memory_id: str, doc_id: str, updated_at: datetime) -> EpisodicMemory:
    return EpisodicMemory(
        memory_id=memory_id,
        user_id="user-1",
        session_id=f"session-{memory_id}",
        query="compare docs",
        episode_summary=f"Episode for {doc_id}",
        evidence=[_evidence(f"chunk-{doc_id}", doc_id)],
        source_scope=[doc_id],
        reliability=0.85,
        created_at=updated_at,
        updated_at=updated_at,
    )


def test_sqlite_memory_repo_persists_user_and_episodic_memory_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    repo = SQLiteMemoryRepo(db_path)
    user_memory = _user_memory(_timestamp(10))
    episode = _episode("episode-1", "doc-b", _timestamp(11))

    repo.save_user_memory(user_memory)
    repo.save_episodic_memory(episode)

    reloaded = SQLiteMemoryRepo(db_path)

    assert reloaded.get_user_memory(user_memory.memory_id) == user_memory
    assert reloaded.get_episodic_memory(episode.memory_id) == episode
    assert reloaded.list_user_memories("user-1") == [user_memory]
    assert reloaded.list_episodic_memories("user-1") == [episode]


def test_sqlite_memory_repo_replaces_existing_memory_and_filters_episode_scope(tmp_path: Path) -> None:
    repo = SQLiteMemoryRepo(tmp_path / "memory.sqlite3")
    original = _user_memory(_timestamp(10))
    updated = original.model_copy(
        update={
            "preference_summary": "Prefer numbered lists for action items.",
            "updated_at": _timestamp(12),
        }
    )
    relevant_episode = _episode("episode-1", "doc-a", _timestamp(11))
    irrelevant_episode = _episode("episode-2", "doc-z", _timestamp(13))

    repo.save_user_memory(original)
    repo.save_user_memory(updated)
    repo.save_episodic_memory(relevant_episode)
    repo.save_episodic_memory(irrelevant_episode)

    stored_preferences = repo.list_user_memories("user-1")
    scoped_episodes = repo.list_episodic_memories("user-1", source_scope=["doc-a"])

    assert stored_preferences == [updated]
    assert scoped_episodes == [relevant_episode]
