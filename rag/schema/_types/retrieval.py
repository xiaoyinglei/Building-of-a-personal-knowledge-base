from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag.query.context import EvidenceBundle, SelfCheckResult
from rag.query.analysis import RoutingDecision
from rag.schema._types.diagnostics import RetrievalDiagnostics
from rag.schema._types.envelope import PreservationSuggestion


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: RoutingDecision
    evidence: EvidenceBundle
    self_check: SelfCheckResult
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    graph_expanded: bool = False
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    preservation_suggestion: PreservationSuggestion = Field(
        default_factory=lambda: PreservationSuggestion(suggested=False)
    )
