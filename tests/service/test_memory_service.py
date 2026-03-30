from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.query._memory.service import MemoryService
from rag.schema._types.access import RuntimeMode
from rag.schema._types.envelope import EvidenceItem, PreservationSuggestion, QueryResponse
from rag.schema._types.memory import EpisodicMemory, MemoryEvidenceLink, UserMemory


class FakeMemoryRepo:
    def __init__(self) -> None:
        self.user_memories: dict[str, UserMemory] = {}
        self.episodic_memories: dict[str, EpisodicMemory] = {}

    def save_user_memory(self, memory: UserMemory) -> None:
        self.user_memories[memory.memory_id] = memory

    def save_episodic_memory(self, memory: EpisodicMemory) -> None:
        self.episodic_memories[memory.memory_id] = memory

    def list_user_memories(self, user_id: str) -> list[UserMemory]:
        return [memory for memory in self.user_memories.values() if memory.user_id == user_id]

    def list_episodic_memories(
        self,
        user_id: str,
        *,
        source_scope: list[str] | None = None,
    ) -> list[EpisodicMemory]:
        memories = [memory for memory in self.episodic_memories.values() if memory.user_id == user_id]
        if not source_scope:
            return memories
        allowed = set(source_scope)
        return [memory for memory in memories if allowed & set(memory.source_scope)]


def _evidence(chunk_id: str, doc_id: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        citation_anchor=f"{doc_id}#{chunk_id}",
        text=f"evidence text for {chunk_id}",
        score=0.9,
    )


def _response(runtime_mode: RuntimeMode) -> QueryResponse:
    return QueryResponse(
        conclusion="Alpha and Beta differ on retention.",
        evidence=[_evidence("chunk-a", "doc-a"), _evidence("chunk-b", "doc-b")],
        differences_or_conflicts=["Retention diverges between Alpha and Beta."],
        uncertainty="medium",
        preservation_suggestion=PreservationSuggestion(suggested=False),
        runtime_mode=runtime_mode,
    )


def _memory_evidence(chunk_id: str, doc_id: str) -> MemoryEvidenceLink:
    return MemoryEvidenceLink(
        chunk_id=chunk_id,
        doc_id=doc_id,
        citation_anchor=f"{doc_id}#{chunk_id}",
    )


def test_memory_service_upserts_evidence_linked_user_preferences() -> None:
    repo = FakeMemoryRepo()
    service = MemoryService(repo=repo, user_id="user-1")

    first = service.remember_user_preference(
        preference_key="answer_style",
        preference_summary="Prefer terse bullet answers.",
        evidence=[_evidence("chunk-a", "doc-a")],
    )
    second = service.remember_user_preference(
        preference_key="answer_style",
        preference_summary="Prefer numbered lists for action items.",
        evidence=[_evidence("chunk-b", "doc-b")],
    )

    assert first.memory_id == second.memory_id
    assert len(repo.user_memories) == 1
    stored = next(iter(repo.user_memories.values()))
    assert stored.preference_summary == "Prefer numbered lists for action items."
    assert stored.evidence[0].chunk_id == "chunk-b"


def test_memory_service_recall_returns_preferences_and_scope_relevant_episodes() -> None:
    repo = FakeMemoryRepo()
    service = MemoryService(repo=repo, user_id="user-1")
    service.remember_user_preference(
        preference_key="answer_style",
        preference_summary="Prefer terse bullet answers.",
        evidence=[_evidence("chunk-a", "doc-a")],
    )
    repo.save_episodic_memory(
        EpisodicMemory(
            memory_id="episode-1",
            user_id="user-1",
            session_id="session-1",
            query="compare docs",
            episode_summary="Alpha and Beta differ on retention.",
            evidence=[_memory_evidence("chunk-a", "doc-a")],
            source_scope=["doc-a"],
            reliability=0.9,
            created_at=datetime(2026, 3, 25, 9, 0, tzinfo=UTC),
            updated_at=datetime(2026, 3, 25, 9, 0, tzinfo=UTC),
        )
    )
    repo.save_episodic_memory(
        EpisodicMemory(
            memory_id="episode-2",
            user_id="user-1",
            session_id="session-2",
            query="calendar plan",
            episode_summary="Unrelated planning summary.",
            evidence=[_memory_evidence("chunk-z", "doc-z")],
            source_scope=["doc-z"],
            reliability=0.9,
            created_at=datetime(2026, 3, 25, 8, 0, tzinfo=UTC),
            updated_at=datetime(2026, 3, 25, 8, 0, tzinfo=UTC),
        )
    )

    hints = service.recall("compare docs", ["doc-a"])

    assert hints == [
        "Preference [answer_style]: Prefer terse bullet answers.",
        "Episode [compare docs]: Alpha and Beta differ on retention.",
    ]


def test_memory_service_records_deep_evidence_backed_episode_summaries() -> None:
    repo = FakeMemoryRepo()
    service = MemoryService(repo=repo, user_id="user-1")

    episode_id = service.record_episode(
        session_id="session-1",
        query="compare docs",
        response=_response(RuntimeMode.DEEP),
        evidence_matrix=[{"claim": "ignored derived claim"}],
        source_scope=["doc-a", "doc-b"],
    )

    assert episode_id in repo.episodic_memories
    stored = repo.episodic_memories[episode_id]
    assert stored.episode_summary == "Alpha and Beta differ on retention."
    assert stored.source_scope == ["doc-a", "doc-b"]
    assert [item.chunk_id for item in stored.evidence] == ["chunk-a", "chunk-b"]


def test_memory_service_rejects_non_deep_episode_capture() -> None:
    service = MemoryService(repo=FakeMemoryRepo(), user_id="user-1")

    with pytest.raises(ValueError, match="deep research responses"):
        service.record_episode(
            session_id="session-1",
            query="compare docs",
            response=_response(RuntimeMode.FAST),
            evidence_matrix=[{"claim": "ignored derived claim"}],
            source_scope=["doc-a", "doc-b"],
        )
