from __future__ import annotations

from dataclasses import dataclass

from pkp.types.envelope import EvidenceItem
from pkp.types.retrieval import RetrievalResult


@dataclass(slots=True)
class ContextEvidenceMerger:
    def merge(self, retrieval: RetrievalResult) -> list[EvidenceItem]:
        internal_by_id = {item.chunk_id: item for item in retrieval.evidence.internal}
        ordered_internal = [
            internal_by_id[chunk_id]
            for chunk_id in retrieval.reranked_chunk_ids
            if chunk_id in internal_by_id
        ]
        seen_internal = {item.chunk_id for item in ordered_internal}
        ordered_internal.extend(item for item in retrieval.evidence.internal if item.chunk_id not in seen_internal)

        merged: list[EvidenceItem] = []
        seen: set[tuple[str, str]] = set()
        for item in [*ordered_internal, *retrieval.evidence.graph, *retrieval.evidence.external]:
            key = (item.evidence_kind, item.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged
