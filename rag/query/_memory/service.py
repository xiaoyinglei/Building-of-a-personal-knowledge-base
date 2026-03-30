from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha1
from typing import Protocol

from rag.schema._types import EvidenceItem, RuntimeMode
from rag.schema._types.envelope import QueryResponse
from rag.schema._types.memory import EpisodicMemory, MemoryEvidenceLink, UserMemory
from rag.schema._types.text import keyword_overlap, search_terms


class MemoryRepo(Protocol):
    def save_user_memory(self, memory: UserMemory) -> None: ...

    def save_episodic_memory(self, memory: EpisodicMemory) -> None: ...

    def list_user_memories(self, user_id: str) -> list[UserMemory]: ...

    def list_episodic_memories(
        self,
        user_id: str,
        *,
        source_scope: list[str] | None = None,
    ) -> list[EpisodicMemory]: ...


class MemoryService:
    def __init__(self, repo: MemoryRepo, user_id: str = "default") -> None:
        self._repo = repo
        self._user_id = user_id

    def remember_user_preference(
        self,
        *,
        preference_key: str,
        preference_summary: str,
        evidence: list[EvidenceItem],
    ) -> UserMemory:
        existing = self._find_user_memory(preference_key)
        now = datetime.now(UTC)
        memory = UserMemory(
            memory_id=existing.memory_id if existing is not None else self._user_memory_id(preference_key),
            user_id=self._user_id,
            preference_key=preference_key,
            preference_summary=preference_summary,
            evidence=self._to_memory_evidence(evidence),
            source_scope=self._source_scope_from_evidence(evidence),
            reliability=self._average_score(evidence),
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        self._repo.save_user_memory(memory)
        return memory

    def recall(self, query: str, source_scope: list[str]) -> list[str]:
        hints = [memory.recall_hint for memory in self._repo.list_user_memories(self._user_id)]
        query_terms = search_terms(query)
        for memory in self._repo.list_episodic_memories(self._user_id, source_scope=source_scope):
            haystack = " ".join([memory.query, memory.episode_summary])
            if keyword_overlap(query_terms, haystack) <= 0:
                continue
            hints.append(memory.recall_hint)
        return hints

    def record_episode(
        self,
        *,
        session_id: str,
        query: str,
        response: QueryResponse,
        evidence_matrix: list[dict[str, object]],
        source_scope: list[str],
    ) -> str:
        del evidence_matrix
        if response.runtime_mode is not RuntimeMode.DEEP:
            raise ValueError("episode capture only supports deep research responses")
        now = datetime.now(UTC)
        memory_id = self._episode_memory_id(session_id, query, response.conclusion)
        memory = EpisodicMemory(
            memory_id=memory_id,
            user_id=self._user_id,
            session_id=session_id,
            query=query,
            episode_summary=response.conclusion,
            evidence=self._to_memory_evidence(response.evidence),
            source_scope=sorted(set(source_scope)),
            reliability=self._average_score(response.evidence),
            created_at=now,
            updated_at=now,
        )
        self._repo.save_episodic_memory(memory)
        return memory_id

    def _find_user_memory(self, preference_key: str) -> UserMemory | None:
        for memory in self._repo.list_user_memories(self._user_id):
            if memory.preference_key == preference_key:
                return memory
        return None

    @staticmethod
    def _to_memory_evidence(evidence: list[EvidenceItem]) -> list[MemoryEvidenceLink]:
        return [
            MemoryEvidenceLink(
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                citation_anchor=item.citation_anchor,
            )
            for item in evidence
        ]

    @staticmethod
    def _source_scope_from_evidence(evidence: list[EvidenceItem]) -> list[str]:
        return sorted({item.doc_id for item in evidence})

    @staticmethod
    def _average_score(evidence: list[EvidenceItem]) -> float:
        if not evidence:
            return 0.0
        return sum(item.score for item in evidence) / len(evidence)

    @staticmethod
    def _user_memory_id(preference_key: str) -> str:
        return f"user-memory-{preference_key}"

    @staticmethod
    def _episode_memory_id(session_id: str, query: str, conclusion: str) -> str:
        digest = sha1(f"{session_id}:{query}:{conclusion}".encode()).hexdigest()[:12]
        return f"episode-memory-{digest}"
