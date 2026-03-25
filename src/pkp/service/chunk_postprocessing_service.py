from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from pkp.repo.parse._util import normalize_whitespace
from pkp.types.content import Chunk


@dataclass(frozen=True)
class PostprocessedChunks:
    parent_chunks: list[Chunk]
    child_chunks: list[Chunk]
    special_chunks: list[Chunk]
    merged_chunks: int
    deduplicated_chunks: int


class ChunkPostprocessingService:
    def __init__(self, *, min_words: int = 24) -> None:
        self._min_words = min_words

    def postprocess(
        self,
        *,
        parent_chunks: list[Chunk],
        child_chunks: list[Chunk],
        special_chunks: list[Chunk],
    ) -> PostprocessedChunks:
        normalized_parents = [
            self._clean_chunk(chunk) for chunk in sorted(parent_chunks, key=lambda item: item.order_index)
        ]
        normalized_children = [
            self._clean_chunk(chunk) for chunk in sorted(child_chunks, key=lambda item: item.order_index)
        ]
        normalized_special = [
            self._clean_chunk(chunk) for chunk in sorted(special_chunks, key=lambda item: item.order_index)
        ]

        merged_children, merged_count = self._merge_short_chunks(normalized_children)
        deduped_children, child_dedup = self._deduplicate(merged_children)
        deduped_special, special_dedup = self._deduplicate(normalized_special)
        linked_children = self._link_neighbors(deduped_children)
        linked_special = self._link_neighbors(deduped_special)
        return PostprocessedChunks(
            parent_chunks=self._assign_hashes(normalized_parents),
            child_chunks=self._assign_hashes(linked_children),
            special_chunks=self._assign_hashes(linked_special),
            merged_chunks=merged_count,
            deduplicated_chunks=child_dedup + special_dedup,
        )

    def _merge_short_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        if not chunks:
            return [], 0
        merged: list[Chunk] = []
        merge_count = 0
        index = 0
        while index < len(chunks):
            current = chunks[index]
            word_count = len(current.text.split())
            if word_count >= self._min_words:
                merged.append(current)
                index += 1
                continue

            if index + 1 < len(chunks) and chunks[index + 1].parent_chunk_id == current.parent_chunk_id:
                next_chunk = chunks[index + 1]
                merged_text = normalize_whitespace(f"{current.text} {next_chunk.text}")
                merged.append(
                    next_chunk.model_copy(
                        update={
                            "text": merged_text,
                            "token_count": len(merged_text.split()),
                            "citation_span": (0, len(merged_text)),
                            "order_index": current.order_index,
                        }
                    )
                )
                merge_count += 1
                index += 2
                continue

            if merged and merged[-1].parent_chunk_id == current.parent_chunk_id:
                previous = merged[-1]
                merged_text = normalize_whitespace(f"{previous.text} {current.text}")
                merged[-1] = previous.model_copy(
                    update={
                        "text": merged_text,
                        "token_count": len(merged_text.split()),
                        "citation_span": (0, len(merged_text)),
                    }
                )
                merge_count += 1
            else:
                merged.append(current)
            index += 1
        return merged, merge_count

    @staticmethod
    def _deduplicate(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        seen: set[tuple[str | None, str, str]] = set()
        results: list[Chunk] = []
        deduped = 0
        for chunk in chunks:
            key = (chunk.parent_chunk_id, chunk.special_chunk_type or "", chunk.text)
            if key in seen:
                deduped += 1
                continue
            seen.add(key)
            results.append(chunk)
        return results, deduped

    @staticmethod
    def _link_neighbors(chunks: list[Chunk]) -> list[Chunk]:
        linked: list[Chunk] = []
        for index, chunk in enumerate(chunks):
            previous = chunks[index - 1].chunk_id if index > 0 else None
            next_chunk = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
            linked.append(
                chunk.model_copy(
                    update={
                        "prev_chunk_id": previous,
                        "next_chunk_id": next_chunk,
                    }
                )
            )
        return linked

    @staticmethod
    def _assign_hashes(chunks: list[Chunk]) -> list[Chunk]:
        results: list[Chunk] = []
        for chunk in chunks:
            content_hash = sha256(chunk.text.encode("utf-8")).hexdigest()
            results.append(chunk.model_copy(update={"content_hash": content_hash}))
        return results

    @staticmethod
    def _clean_chunk(chunk: Chunk) -> Chunk:
        cleaned = normalize_whitespace(chunk.text)
        return chunk.model_copy(
            update={
                "text": cleaned,
                "token_count": len(cleaned.split()),
                "citation_span": (0, len(cleaned)),
            }
        )
