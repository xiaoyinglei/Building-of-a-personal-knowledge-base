from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SessionSnapshot:
    sub_questions: list[str] = field(default_factory=list)
    evidence_matrix: list[dict[str, object]] = field(default_factory=list)


class SessionRuntime:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionSnapshot] = {}

    def get(self, session_id: str) -> SessionSnapshot:
        return self._sessions.setdefault(session_id, SessionSnapshot())

    def store_sub_questions(self, session_id: str, sub_questions: list[str]) -> None:
        snapshot = self.get(session_id)
        snapshot.sub_questions = list(sub_questions)

    def store_evidence_matrix(
        self,
        session_id: str,
        evidence_matrix: list[dict[str, object]],
    ) -> None:
        snapshot = self.get(session_id)
        snapshot.evidence_matrix = list(evidence_matrix)
