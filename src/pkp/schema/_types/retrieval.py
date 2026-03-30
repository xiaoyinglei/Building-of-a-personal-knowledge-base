from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pkp.query.context import EvidenceBundle, RoutingDecision, SelfCheckResult
from pkp.schema._types.diagnostics import RetrievalDiagnostics
from pkp.schema._types.envelope import PreservationSuggestion


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
