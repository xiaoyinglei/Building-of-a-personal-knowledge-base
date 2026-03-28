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
        merged_by_chunk_id: dict[str, EvidenceItem] = {}
        ordered_chunk_ids: list[str] = []

        for item in [*ordered_internal, *retrieval.evidence.graph]:
            existing = merged_by_chunk_id.get(item.chunk_id)
            if existing is None:
                merged_by_chunk_id[item.chunk_id] = item
                ordered_chunk_ids.append(item.chunk_id)
                continue
            merged_by_chunk_id[item.chunk_id] = self._merge_duplicate_item(existing, item)

        merged.extend(merged_by_chunk_id[chunk_id] for chunk_id in ordered_chunk_ids)

        seen_external: set[str] = set()
        for item in retrieval.evidence.external:
            if item.chunk_id in seen_external:
                continue
            seen_external.add(item.chunk_id)
            merged.append(item)
        return merged

    @staticmethod
    def _merge_duplicate_item(existing: EvidenceItem, incoming: EvidenceItem) -> EvidenceItem:
        preferred = existing
        secondary = incoming
        if existing.evidence_kind != "internal" and incoming.evidence_kind == "internal":
            preferred = incoming
            secondary = existing

        merged_kind = "internal" if "internal" in {existing.evidence_kind, incoming.evidence_kind} else preferred.evidence_kind
        merged_text = preferred.text if len(preferred.text) >= len(secondary.text) else secondary.text

        return preferred.model_copy(
            update={
                "evidence_kind": merged_kind,
                "score": max(float(existing.score), float(incoming.score)),
                "text": merged_text,
                "section_path": preferred.section_path or secondary.section_path,
                "file_name": preferred.file_name or secondary.file_name,
                "source_id": preferred.source_id or secondary.source_id,
                "chunk_type": preferred.chunk_type or secondary.chunk_type,
                "source_type": preferred.source_type or secondary.source_type,
                "special_chunk_type": preferred.special_chunk_type or secondary.special_chunk_type,
                "parent_chunk_id": preferred.parent_chunk_id or secondary.parent_chunk_id,
                "page_start": preferred.page_start if preferred.page_start is not None else secondary.page_start,
                "page_end": preferred.page_end if preferred.page_end is not None else secondary.page_end,
            }
        )
