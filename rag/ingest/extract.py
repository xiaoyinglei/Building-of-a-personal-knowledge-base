from __future__ import annotations

import json
import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.schema.chunk import Chunk
from rag.schema.document import Document


class ChatModelLike(Protocol):
    def chat(self, prompt: str) -> str: ...

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

_CANONICAL_RELATION_MAP = {
    "support": "supports",
    "supports": "supports",
    "supported_by": "supports",
    "contains": "contains",
    "contain": "contains",
    "includes": "contains",
    "included_in": "contains",
    "enables": "enables",
    "enabled_by": "enables",
    "requires": "requires",
    "required_by": "requires",
    "uses": "uses",
    "used_by": "uses",
    "depends_on": "depends_on",
    "dependency_on": "depends_on",
    "dependent_on": "depends_on",
    "part_of": "part_of",
    "belongs_to": "part_of",
    "composed_of": "contains",
    "integrates_with": "integrates_with",
    "connects_to": "integrates_with",
    "linked_to": "integrates_with",
    "compares": "compares",
    "contrasts": "compares",
    "related_to": "related_to",
}

_UNDIRECTED_RELATIONS = {"related_to", "compares", "integrates_with"}

_ENTITY_TYPE_SUFFIXES = {
    "service": "service",
    "engine": "system",
    "index": "index",
    "graph": "graph",
    "pipeline": "pipeline",
    "model": "model",
    "table": "table",
    "database": "database",
    "vector": "vector_store",
    "document": "document",
}


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


def normalize_entity_key(label: str) -> str:
    normalized = label.strip().lower().replace("&", " and ")
    normalized = re.sub(r"[`'’]", "", normalized)
    normalized = re.sub(r"^\s*(?:a|an|the)\s+", "", normalized)
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    singularized = [_singularize_token(token) for token in tokens if token not in _STOPWORDS]
    return "_".join(singularized)


def canonicalize_relation_type(relation_type: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", relation_type.strip().lower()).strip("_")
    return _CANONICAL_RELATION_MAP.get(normalized, normalized or "related_to")


def canonicalize_relation_pair(left_key: str, right_key: str, relation_type: str) -> tuple[str, str]:
    normalized_relation = canonicalize_relation_type(relation_type)
    if normalized_relation in _UNDIRECTED_RELATIONS:
        return tuple(sorted((left_key, right_key), key=str))  # type: ignore[return-value]
    return left_key, right_key


def choose_preferred_label(labels: Sequence[str]) -> str:
    normalized = [label.strip() for label in labels if label and label.strip()]
    if not normalized:
        return ""
    non_acronyms = [label for label in normalized if not _is_acronym(label)]
    pool = non_acronyms or list(normalized)
    return max(pool, key=lambda label: (len(label.split()), len(label), label))


class EntityRelationExtractor(Protocol):
    def extract(
        self,
        *,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> EntityRelationExtractionResult: ...


class EmptyEntityRelationExtractor:
    def extract(
        self,
        *,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> EntityRelationExtractionResult:
        del document, chunks
        return EntityRelationExtractionResult()


class PromptedEntityRelationExtractor:
    def __init__(
        self,
        *,
        model_provider: ChatModelLike | None = None,
        fallback: EntityRelationExtractor | None = None,
    ) -> None:
        self._model_provider = model_provider
        self._fallback = fallback or EmptyEntityRelationExtractor()

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
            "Each relation requires: source_key, target_key, relation_type, "
            "description, confidence, source_chunk_ids.\n"
            "Only emit relations that are explicitly supported by the chunk text.\n"
            "Do not estimate probabilistic confidence. Use confidence=1.0 for supported relations.\n"
            "If a relation is uncertain, omit it.\n"
            f"Document title: {document.title}\n"
            f"Chunks: {json.dumps(chunk_payload, ensure_ascii=True)}"
        )

    @staticmethod
    def _parse_response(response: str) -> EntityRelationExtractionResult | None:
        candidate = _extract_json_object(response)
        if candidate is None:
            return None
        try:
            payload = json.loads(candidate)
            return EntityRelationExtractionResult.model_validate(payload)
        except Exception:
            return None

    @staticmethod
    def _merge_llm_and_fallback(
        llm_result: EntityRelationExtractionResult,
        fallback_result: EntityRelationExtractionResult,
    ) -> EntityRelationExtractionResult:
        entity_by_key: dict[str, ExtractedEntity] = {}
        for entity in [*fallback_result.entities, *llm_result.entities]:
            canonical_key = normalize_entity_key(entity.key or entity.label)
            if not canonical_key:
                continue
            existing_entity = entity_by_key.get(canonical_key)
            aliases: list[str] = []
            if existing_entity is not None:
                aliases.extend(existing_entity.metadata.get("aliases", "").split("||"))
            aliases.extend(entity.metadata.get("aliases", "").split("||"))
            aliases.append(entity.label)
            preferred_label = choose_preferred_label([alias for alias in aliases if alias])
            entity_by_key[canonical_key] = ExtractedEntity(
                key=canonical_key,
                label=preferred_label or entity.label,
                entity_type=entity.entity_type
                if entity.entity_type != "concept"
                else (existing_entity.entity_type if existing_entity else "concept"),
                description=_choose_longer(existing_entity.description if existing_entity else "", entity.description),
                source_chunk_ids=list(
                    dict.fromkeys(
                        [*(existing_entity.source_chunk_ids if existing_entity else []), *entity.source_chunk_ids]
                    )
                ),
                metadata={
                    **({} if existing_entity is None else existing_entity.metadata),
                    **entity.metadata,
                    "aliases": "||".join(sorted(dict.fromkeys(alias for alias in aliases if alias))),
                },
            )

        relation_by_key: dict[tuple[str, str, str], ExtractedRelation] = {}
        for relation in [*fallback_result.relations, *llm_result.relations]:
            source_key = normalize_entity_key(relation.source_key)
            target_key = normalize_entity_key(relation.target_key)
            relation_type = canonicalize_relation_type(relation.relation_type)
            source_key, target_key = canonicalize_relation_pair(source_key, target_key, relation_type)
            relation_key = (source_key, target_key, relation_type)
            existing_relation = relation_by_key.get(relation_key)
            relation_by_key[relation_key] = ExtractedRelation(
                source_key=source_key,
                target_key=target_key,
                relation_type=relation_type,
                description=_choose_longer(
                    existing_relation.description if existing_relation else "",
                    relation.description,
                ),
                confidence=max(relation.confidence, 0.0 if existing_relation is None else existing_relation.confidence),
                source_chunk_ids=list(
                    dict.fromkeys(
                        [*(existing_relation.source_chunk_ids if existing_relation else []), *relation.source_chunk_ids]
                    )
                ),
                metadata={**({} if existing_relation is None else existing_relation.metadata), **relation.metadata},
            )
        return EntityRelationExtractionResult(
            entities=list(entity_by_key.values()),
            relations=list(relation_by_key.values()),
        )


def _extract_json_object(response: str) -> str | None:
    stripped = response.strip()
    if not stripped:
        return None
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return stripped[start : end + 1]


def _singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith(("sses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _label_acronym(label: str) -> str | None:
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", label) if token]
    if len(tokens) < 2:
        return None
    return "".join(token[0].upper() for token in tokens)


def _is_acronym(label: str) -> bool:
    normalized = label.strip()
    return normalized.isupper() and len(normalized) <= 6 and " " not in normalized


def _choose_longer(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return left if len(left) >= len(right) else right


class MergedEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    key: str
    label: str
    entity_type: str
    description: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class MergedRelation(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    from_node_id: str
    to_node_id: str
    relation_type: str
    description: str
    confidence: float
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class MergedGraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: list[MergedEntity] = Field(default_factory=list)
    relations: list[MergedRelation] = Field(default_factory=list)


class EntityRelationMerger:
    def merge(
        self,
        *,
        document: Document,
        extraction: EntityRelationExtractionResult,
    ) -> MergedGraph:
        merged_entities: dict[str, MergedEntity] = {}
        for entity in extraction.entities:
            entity_key = normalize_entity_key(entity.key or entity.label)
            if not entity_key:
                continue
            node_id = self._deterministic_id(entity_key, "entity")
            existing_entity = merged_entities.get(entity_key)
            evidence_chunk_ids = self._merge_chunk_ids(
                [] if existing_entity is None else existing_entity.evidence_chunk_ids,
                entity.source_chunk_ids,
            )
            description = (
                entity.description
                if existing_entity is None
                else self._choose_description(
                    existing_entity.description,
                    entity.description,
                )
            )
            aliases: list[str] = []
            if existing_entity is not None:
                aliases.extend(alias for alias in existing_entity.metadata.get("aliases", "").split("||") if alias)
            aliases.extend(alias for alias in entity.metadata.get("aliases", "").split("||") if alias)
            aliases.append(entity.label)
            preferred_label = choose_preferred_label(aliases) or entity.label
            merged_entities[entity_key] = MergedEntity(
                node_id=node_id,
                key=entity_key,
                label=preferred_label,
                entity_type=entity.entity_type
                if existing_entity is None or existing_entity.entity_type == "concept"
                else existing_entity.entity_type,
                description=description,
                evidence_chunk_ids=evidence_chunk_ids,
                metadata={
                    "doc_id": document.doc_id,
                    "doc_ids": document.doc_id,
                    "entity_key": entity_key,
                    "evidence_count": str(len(evidence_chunk_ids)),
                    "aliases": "||".join(sorted(dict.fromkeys(alias for alias in aliases if alias))),
                },
            )

        merged_relations: dict[tuple[str, str, str], MergedRelation] = {}
        for relation in extraction.relations:
            source_key = normalize_entity_key(relation.source_key)
            target_key = normalize_entity_key(relation.target_key)
            relation_type = canonicalize_relation_type(relation.relation_type)
            source_key, target_key = canonicalize_relation_pair(source_key, target_key, relation_type)
            source = merged_entities.get(source_key)
            target = merged_entities.get(target_key)
            if source is None or target is None:
                continue
            relation_key = (source.node_id, target.node_id, relation_type)
            existing_relation = merged_relations.get(relation_key)
            evidence_chunk_ids = self._merge_chunk_ids(
                [] if existing_relation is None else existing_relation.evidence_chunk_ids,
                relation.source_chunk_ids,
            )
            description = (
                relation.description
                if existing_relation is None
                else self._choose_description(
                    existing_relation.description,
                    relation.description,
                )
            )
            merged_relations[relation_key] = MergedRelation(
                edge_id=self._deterministic_id(
                    source.node_id,
                    target.node_id,
                    relation_type,
                    "edge",
                ),
                from_node_id=source.node_id,
                to_node_id=target.node_id,
                relation_type=relation_type,
                description=description,
                confidence=max(
                    relation.confidence,
                    0.0 if existing_relation is None else existing_relation.confidence,
                ),
                evidence_chunk_ids=evidence_chunk_ids,
                metadata={
                    "doc_id": document.doc_id,
                    "doc_ids": document.doc_id,
                    "from_label": source.label,
                    "to_label": target.label,
                    "evidence_count": str(len(evidence_chunk_ids)),
                },
            )

        return MergedGraph(
            entities=list(merged_entities.values()),
            relations=list(merged_relations.values()),
        )

    @staticmethod
    def _merge_chunk_ids(left: list[str], right: list[str]) -> list[str]:
        return list(dict.fromkeys([*left, *right]))

    @staticmethod
    def _choose_description(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return left if len(left) >= len(right) else right

    @staticmethod
    def _deterministic_id(*parts: str) -> str:
        digest = sha256("\0".join(parts).encode("utf-8")).hexdigest()
        return f"{parts[-1]}-{digest[:16]}"


__all__ = [
    "EntityRelationExtractionResult",
    "EntityRelationMerger",
    "EmptyEntityRelationExtractor",
    "ExtractedEntity",
    "ExtractedRelation",
    "HeuristicEntityRelationExtractor",
    "MergedEntity",
    "MergedGraph",
    "MergedRelation",
    "PromptedEntityRelationExtractor",
    "canonicalize_relation_pair",
    "canonicalize_relation_type",
    "choose_preferred_label",
    "normalize_entity_key",
]

HeuristicEntityRelationExtractor = EmptyEntityRelationExtractor
