from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pkp.types.memory import EpisodicMemory, MemoryEvidenceLink, MemoryKind, UserMemory


def _evidence(chunk_id: str, doc_id: str) -> MemoryEvidenceLink:
    return MemoryEvidenceLink(
        chunk_id=chunk_id,
        doc_id=doc_id,
        citation_anchor=f"{doc_id}#{chunk_id}",
    )


def _timestamp() -> datetime:
    return datetime(2026, 3, 25, 9, 0, tzinfo=UTC)


def test_memory_contracts_capture_user_preferences_and_episode_summaries() -> None:
    user_memory = UserMemory(
        memory_id="mem-user-1",
        user_id="user-1",
        preference_key="answer_style",
        preference_summary="Prefer terse bullet answers.",
        evidence=[_evidence("chunk-a", "doc-a"), _evidence("chunk-b", "doc-b")],
        source_scope=["doc-b", "doc-a", "doc-a"],
        reliability=0.9,
        created_at=_timestamp(),
        updated_at=_timestamp(),
    )
    episodic_memory = EpisodicMemory(
        memory_id="mem-episode-1",
        user_id="user-1",
        session_id="session-1",
        query="compare docs",
        episode_summary="Alpha and Beta differ on retention.",
        evidence=[_evidence("chunk-c", "doc-c")],
        source_scope=["doc-c"],
        reliability=0.85,
        created_at=_timestamp(),
        updated_at=_timestamp(),
    )

    assert user_memory.memory_kind is MemoryKind.USER_PREFERENCE
    assert user_memory.source_scope == ["doc-a", "doc-b"]
    assert user_memory.recall_hint == "Preference [answer_style]: Prefer terse bullet answers."
    assert episodic_memory.memory_kind is MemoryKind.EPISODIC_SUMMARY
    assert episodic_memory.recall_hint == "Episode [compare docs]: Alpha and Beta differ on retention."


def test_memory_contracts_require_evidence_backed_source_scope() -> None:
    with pytest.raises(ValidationError, match="source_scope must be backed by evidence doc_ids"):
        UserMemory(
            memory_id="mem-user-1",
            user_id="user-1",
            preference_key="answer_style",
            preference_summary="Prefer terse bullet answers.",
            evidence=[_evidence("chunk-a", "doc-a")],
            source_scope=["doc-missing"],
            reliability=0.9,
            created_at=_timestamp(),
            updated_at=_timestamp(),
        )


def test_memory_contracts_require_supporting_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence must not be empty"):
        EpisodicMemory(
            memory_id="mem-episode-1",
            user_id="user-1",
            session_id="session-1",
            query="compare docs",
            episode_summary="Alpha and Beta differ on retention.",
            evidence=[],
            source_scope=["doc-c"],
            reliability=0.85,
            created_at=_timestamp(),
            updated_at=_timestamp(),
        )


def test_memory_contracts_reject_blank_summaries() -> None:
    with pytest.raises(ValidationError, match="preference_summary must not be blank"):
        UserMemory(
            memory_id="mem-user-1",
            user_id="user-1",
            preference_key="answer_style",
            preference_summary="  ",
            evidence=[_evidence("chunk-a", "doc-a")],
            source_scope=["doc-a"],
            reliability=0.9,
            created_at=_timestamp(),
            updated_at=_timestamp(),
        )
