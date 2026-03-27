from __future__ import annotations

from hashlib import sha256

from pkp.repo.parse._util import normalize_whitespace
from pkp.types.access import AccessPolicy
from pkp.types.content import Chunk, Segment


class ChunkingService:
    def __init__(self, *, max_words: int = 160) -> None:
        self._max_words = max_words

    def chunk_section(
        self,
        *,
        location: str,
        doc_id: str,
        segment: Segment,
        text: str,
        access_policy: AccessPolicy,
    ) -> list[Chunk]:
        normalized = normalize_whitespace(text)
        if not normalized:
            return []

        words = normalized.split(" ")
        if len(words) <= self._max_words:
            chunks = [normalized]
        else:
            chunks = []
            for start in range(0, len(words), self._max_words):
                chunks.append(" ".join(words[start : start + self._max_words]))

        results: list[Chunk] = []
        cursor = 0
        for index, chunk_text in enumerate(chunks):
            span_start = normalized.find(chunk_text, cursor)
            if span_start < 0:
                span_start = cursor
            span_end = span_start + len(chunk_text)
            cursor = span_end
            chunk_id = self._chunk_id(location, segment.anchor or segment.segment_id, index)
            results.append(
                Chunk(
                    chunk_id=chunk_id,
                    segment_id=segment.segment_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    token_count=len(chunk_text.split()),
                    citation_anchor=segment.anchor or segment.segment_id,
                    citation_span=(span_start, span_end),
                    effective_access_policy=access_policy,
                    extraction_quality=1.0 if len(chunks) == 1 else 0.95,
                    embedding_ref=chunk_id,
                    order_index=index,
                    metadata={"order_index": str(index)},
                )
            )
        return results

    @staticmethod
    def _chunk_id(location: str, anchor: str, index: int) -> str:
        digest = sha256(f"{location}|{anchor}|{index}".encode()).hexdigest()
        return f"chunk-{digest[:16]}"
