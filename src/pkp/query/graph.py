from __future__ import annotations

from collections.abc import Sequence

from pkp.algorithms.retrieval.search_backed_factory import (
    HybridSpecialRetriever,
    MultiProviderBackedVectorRetriever,
    RetrievedCandidate,
    SearchBackedRetrievalFactory,
)
from pkp.service.evidence_service import CandidateLike, EvidenceBundle
from pkp.schema.document import AccessPolicy


class GraphExpansionService:
    @staticmethod
    def _candidate_allowed(candidate: CandidateLike, source_scope: Sequence[str]) -> bool:
        if not source_scope:
            return True
        scope = {candidate.doc_id}
        if candidate.source_id:
            scope.add(candidate.source_id)
        return bool(scope & set(source_scope))

    def expand(
        self,
        *,
        query: str,
        source_scope: Sequence[str],
        evidence: EvidenceBundle,
        graph_candidates: Sequence[CandidateLike],
        access_policy: AccessPolicy,
    ) -> list[CandidateLike]:
        if not evidence.internal:
            return []
        return [candidate for candidate in graph_candidates if self._candidate_allowed(candidate, source_scope)]

__all__ = [
    "GraphExpansionService",
    "HybridSpecialRetriever",
    "MultiProviderBackedVectorRetriever",
    "RetrievedCandidate",
    "SearchBackedRetrievalFactory",
]
