from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from pkp.repo.interfaces import ModelProviderRepo
from pkp.types.content import Chunk, Document

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
_UPPER_CASE_RE = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_STOPWORDS = {
    "about",
    "after",
    "before",
    "between",
    "because",
    "chunk",
    "context",
    "detail",
    "document",
    "evidence",
    "from",
    "have",
    "into",
    "more",
    "note",
    "page",
    "query",
    "reliable",
    "retrieval",
    "section",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
}

_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\b(supports?|supported by)\b", re.IGNORECASE), "supports", 0.88),
    (re.compile(r"\b(includes?|contains?)\b", re.IGNORECASE), "contains", 0.84),
    (re.compile(r"\b(enables?|enabled by)\b", re.IGNORECASE), "enables", 0.83),
    (re.compile(r"\b(requires?|required by)\b", re.IGNORECASE), "requires", 0.82),
    (re.compile(r"\b(improves?|improved by)\b", re.IGNORECASE), "improves", 0.8),
    (re.compile(r"\b(compare[sd]?|contrasts?)\b", re.IGNORECASE), "compares", 0.75),
    (re.compile(r"\b(uses?|used by)\b", re.IGNORECASE), "uses", 0.78),
)


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    entity_type: str = "concept"
    description: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_key: str
    target_key: str
    relation_type: str
    description: str
    confidence: float
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class EntityRelationExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class EntityRelationExtractor(Protocol):
    def extract(
        self,
        *,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> EntityRelationExtractionResult: ...


class HeuristicEntityRelationExtractor:
    def extract(
        self,
        *,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> EntityRelationExtractionResult:
        del document
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []
        for chunk in chunks:
            chunk_entities = self._extract_entities_from_chunk(chunk)
            entities.extend(chunk_entities)
            relations.extend(self._extract_relations_from_chunk(chunk, chunk_entities))
        return EntityRelationExtractionResult(entities=entities, relations=relations)

    def _extract_entities_from_chunk(self, chunk: Chunk) -> list[ExtractedEntity]:
        labels = self._candidate_labels(chunk)
        if not labels:
            return []
        description = self._description_for(chunk.text)
        entities: list[ExtractedEntity] = []
        seen_keys: set[str] = set()
        for label in labels:
            key = self._normalize_key(label)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            entities.append(
                ExtractedEntity(
                    key=key,
                    label=label,
                    entity_type="concept",
                    description=description,
                    source_chunk_ids=[chunk.chunk_id],
                    metadata={
                        "chunk_role": chunk.chunk_role.value,
                        "source_chunk_id": chunk.chunk_id,
                    },
                )
            )
        return entities

    def _extract_relations_from_chunk(
        self,
        chunk: Chunk,
        entities: Sequence[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        if len(entities) < 2:
            return []
        relations: list[ExtractedRelation] = []
        seen_keys: set[tuple[str, str, str]] = set()
        sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(chunk.text) if sentence.strip()]
        for sentence in sentences or [chunk.text]:
            sentence_entities = [entity for entity in entities if entity.label.lower() in sentence.lower()]
            if len(sentence_entities) < 2:
                continue
            relation_type, confidence = self._infer_relation(sentence)
            pairs = zip(sentence_entities, sentence_entities[1:], strict=False)
            for left, right in pairs:
                source_key, target_key = self._canonicalize_pair(
                    left.key,
                    right.key,
                    relation_type,
                )
                relation_key = (source_key, target_key, relation_type)
                if source_key == target_key or relation_key in seen_keys:
                    continue
                seen_keys.add(relation_key)
                relations.append(
                    ExtractedRelation(
                        source_key=source_key,
                        target_key=target_key,
                        relation_type=relation_type,
                        description=self._description_for(sentence),
                        confidence=confidence,
                        source_chunk_ids=[chunk.chunk_id],
                        metadata={"source_chunk_id": chunk.chunk_id},
                    )
                )
        if relations:
            return relations
        fallback_relations: list[ExtractedRelation] = []
        for left, right in zip(entities, entities[1:], strict=False):
            source_key, target_key = self._canonicalize_pair(left.key, right.key, "related_to")
            relation_key = (source_key, target_key, "related_to")
            if relation_key in seen_keys or source_key == target_key:
                continue
            seen_keys.add(relation_key)
            fallback_relations.append(
                ExtractedRelation(
                    source_key=source_key,
                    target_key=target_key,
                    relation_type="related_to",
                    description=self._description_for(chunk.text),
                    confidence=0.45,
                    source_chunk_ids=[chunk.chunk_id],
                    metadata={"source_chunk_id": chunk.chunk_id},
                )
            )
        return fallback_relations

    def _candidate_labels(self, chunk: Chunk) -> list[str]:
        labels: list[str] = []
        for pattern in (_TITLE_CASE_RE, _UPPER_CASE_RE):
            for match in pattern.findall(chunk.text):
                normalized = match.strip(".,:;()[]{} ")
                if normalized and normalized not in labels:
                    labels.append(normalized)
        toc_tail = chunk.metadata.get("toc_path", "").split(" > ")[-1].strip()
        if toc_tail and len(toc_tail) > 3 and toc_tail.lower() not in _STOPWORDS and toc_tail not in labels:
            labels.append(toc_tail)
        if labels:
            return labels[:6]

        tokens = [
            token.lower()
            for token in _TOKEN_RE.findall(chunk.text)
            if token.lower() not in _STOPWORDS and not token.isdigit()
        ]
        counts = Counter(tokens)
        for token, _count in counts.most_common(4):
            label = token.replace("_", " ")
            if label not in labels:
                labels.append(label)
        return labels[:4]

    @staticmethod
    def _description_for(text: str) -> str:
        normalized = " ".join(text.split())
        return normalized[:280]

    @staticmethod
    def _infer_relation(text: str) -> tuple[str, float]:
        for pattern, relation_type, confidence in _RELATION_PATTERNS:
            if pattern.search(text):
                return relation_type, confidence
        return "related_to", 0.55

    @staticmethod
    def _normalize_key(label: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return normalized

    @staticmethod
    def _canonicalize_pair(left_key: str, right_key: str, relation_type: str) -> tuple[str, str]:
        if relation_type in {"related_to", "compares"}:
            return tuple(sorted((left_key, right_key), key=str))  # type: ignore[return-value]
        return left_key, right_key


class PromptedEntityRelationExtractor:
    def __init__(
        self,
        *,
        model_provider: ModelProviderRepo | None = None,
        fallback: EntityRelationExtractor | None = None,
    ) -> None:
        self._model_provider = model_provider
        self._fallback = fallback or HeuristicEntityRelationExtractor()

    def extract(
        self,
        *,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> EntityRelationExtractionResult:
        fallback_result = self._fallback.extract(document=document, chunks=chunks)
        if self._model_provider is None:
            return fallback_result

        prompt = self._build_prompt(document=document, chunks=chunks)
        try:
            response = self._model_provider.chat(prompt)
        except Exception:
            return fallback_result
        parsed = self._parse_response(response)
        if parsed is None:
            return fallback_result

        if not parsed.entities and fallback_result.entities:
            return fallback_result
        return self._merge_llm_and_fallback(parsed, fallback_result)

    @staticmethod
    def _build_prompt(*, document: Document, chunks: Sequence[Chunk]) -> str:
        chunk_payload = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "toc_path": chunk.metadata.get("toc_path", ""),
            }
            for chunk in chunks
        ]
        return (
            "Extract grounded entities and relations from the following chunks. "
            "Return strict JSON with keys `entities` and `relations`.\n"
            "Each entity requires: key, label, entity_type, description, source_chunk_ids.\n"
            "Each relation requires: source_key, target_key, relation_type, description, confidence, source_chunk_ids.\n"
            f"Document title: {document.title}\n"
            f"Chunks: {json.dumps(chunk_payload, ensure_ascii=True)}"
        )

    @staticmethod
    def _parse_response(response: str) -> EntityRelationExtractionResult | None:
        match = _JSON_BLOCK_RE.search(response)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
            return EntityRelationExtractionResult.model_validate(payload)
        except Exception:
            return None

    @staticmethod
    def _merge_llm_and_fallback(
        llm_result: EntityRelationExtractionResult,
        fallback_result: EntityRelationExtractionResult,
    ) -> EntityRelationExtractionResult:
        entity_by_key = {entity.key: entity for entity in fallback_result.entities}
        for entity in llm_result.entities:
            entity_by_key[entity.key] = entity
        relation_by_key = {
            (relation.source_key, relation.target_key, relation.relation_type): relation
            for relation in fallback_result.relations
        }
        for relation in llm_result.relations:
            relation_by_key[(relation.source_key, relation.target_key, relation.relation_type)] = relation
        return EntityRelationExtractionResult(
            entities=list(entity_by_key.values()),
            relations=list(relation_by_key.values()),
        )
