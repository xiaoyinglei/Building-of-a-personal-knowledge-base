from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from typing import Protocol

from pkp.utils._contracts import VectorSearchResult


class VectorChunkRecord(Protocol):
    chunk_id: str
    doc_id: str
    segment_id: str
    text: str


@dataclass(frozen=True)
class _VectorRecord:
    item_id: str
    item_kind: str
    embedding_space: str
    vector: tuple[float, ...]
    metadata: dict[str, str]
    text: str = ""


class InMemoryVectorRepo:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], _VectorRecord] = {}

    def upsert(
        self,
        item_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> None:
        self._records[(embedding_space, item_kind, item_id)] = _VectorRecord(
            item_id=item_id,
            item_kind=item_kind,
            embedding_space=embedding_space,
            vector=tuple(float(value) for value in vector),
            metadata=dict(metadata or {}),
            text=dict(metadata or {}).get("text", ""),
        )

    def index_chunks(self, chunks: list[VectorChunkRecord]) -> None:
        vocabulary = self._build_vocabulary(chunk.text for chunk in chunks)
        for chunk in chunks:
            text = chunk.text
            vector = self._text_to_vector(text, vocabulary)
            metadata = {"doc_id": chunk.doc_id, "segment_id": chunk.segment_id}
            self._records[("default", "chunk", chunk.chunk_id)] = _VectorRecord(
                item_id=chunk.chunk_id,
                item_kind="chunk",
                embedding_space="default",
                vector=vector,
                metadata=metadata,
                text=text,
            )

    def search(
        self,
        query: str | Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "chunk",
    ) -> list[VectorSearchResult]:
        records = [
            record
            for record in self._records.values()
            if (embedding_space is None or record.embedding_space == embedding_space) and record.item_kind == item_kind
        ]
        if not records:
            return []

        allowed_doc_ids = set(doc_ids or [])
        if isinstance(query, str):
            vocabulary = self._build_vocabulary([query, *(record.text for record in records)])
            query_vector = self._text_to_vector(query, vocabulary)
            scored = []
            for record in records:
                if allowed_doc_ids and record.metadata.get("doc_id") not in allowed_doc_ids:
                    continue
                record_vector = self._text_to_vector(record.text, vocabulary)
                scored.append(
                    VectorSearchResult(
                        item_id=record.item_id,
                        score=self._cosine_similarity(query_vector, record_vector),
                        metadata=dict(record.metadata),
                    )
                )
            scored.sort(key=lambda result: (-result.score, result.item_id))
            return scored[:limit]

        query_vector = tuple(float(value) for value in query)
        if not query_vector:
            return []

        scored = []
        for record in records:
            if allowed_doc_ids and record.metadata.get("doc_id") not in allowed_doc_ids:
                continue
            scored.append(
                VectorSearchResult(
                    item_id=record.item_id,
                    score=self._cosine_similarity(query_vector, record.vector),
                    metadata=dict(record.metadata),
                )
            )
        scored.sort(key=lambda result: (-result.score, result.item_id))
        return scored[:limit]

    def existing_item_ids(
        self,
        item_ids: tuple[str, ...] | list[str],
        *,
        embedding_space: str | None = None,
        item_kind: str | None = "chunk",
    ) -> set[str]:
        return {
            item_id
            for item_id in item_ids
            if any(
                record.item_id == item_id
                and (embedding_space is None or record.embedding_space == embedding_space)
                and (item_kind is None or record.item_kind == item_kind)
                for record in self._records.values()
            )
        }

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_chunks: bool = False,
    ) -> int:
        records = [
            record
            for record in self._records.values()
            if (embedding_space is None or record.embedding_space == embedding_space)
            and (item_kind is None or record.item_kind == item_kind)
        ]
        if not distinct_chunks:
            return len(records)
        return len({record.item_id for record in records})

    @staticmethod
    def _build_vocabulary(texts: Iterable[str]) -> tuple[str, ...]:
        tokens = sorted({token for text in texts for token in text.lower().split()})
        return tuple(tokens)

    @staticmethod
    def _text_to_vector(text: str, vocabulary: tuple[str, ...]) -> tuple[float, ...]:
        counts = {token: 0.0 for token in vocabulary}
        for token in text.lower().split():
            if token in counts:
                counts[token] += 1.0
        return tuple(counts[token] for token in vocabulary)

    @staticmethod
    def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            raise ValueError("query vector and stored vector dimensions must match")

        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        dot = sum(lv * rv for lv, rv in zip(left, right, strict=True))
        return dot / (left_norm * right_norm)
