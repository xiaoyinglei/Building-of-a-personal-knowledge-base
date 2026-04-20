from __future__ import annotations

import json
import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.schema.core import Chunk, Document


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

_RELATION_PHRASES = (
    "depends on",
    "integrates with",
    "part of",
    "supports",
    "contains",
    "enables",
    "requires",
    "uses",
    "compares",
    "relates to",
)
_PASSIVE_RELATION_PHRASES = (
    "supported by",
    "used by",
    "enabled by",
    "required by",
    "included in",
    "part of",
)
_ENTITY_LABEL_PATTERN = r"(?:[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)+|[A-Z]{2,6})"
_EXPLICIT_RELATION_RE = re.compile(
    rf"(?P<left>{_ENTITY_LABEL_PATTERN})\s+"
    r"(?P<relation>"
    + "|".join(re.escape(phrase) for phrase in _RELATION_PHRASES)
    + r")\s+"
    rf"(?P<right>{_ENTITY_LABEL_PATTERN})",
)
_PASSIVE_RELATION_RE = re.compile(
    rf"(?P<target>{_ENTITY_LABEL_PATTERN})\s+is\s+"
    r"(?P<relation>"
    + "|".join(re.escape(phrase) for phrase in _PASSIVE_RELATION_PHRASES)
    + r")\s+"
    rf"(?P<source>{_ENTITY_LABEL_PATTERN})",
)
_ALIAS_RE = re.compile(r"(?P<label>[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)+)\s+\((?P<alias>[A-Z]{2,6})\)")
_ENTITY_MENTION_RE = re.compile(_ENTITY_LABEL_PATTERN)


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


class HeuristicEntityRelationExtractor:
    def extract(
        self,
        *,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> EntityRelationExtractionResult:
        del document
        entities: dict[str, ExtractedEntity] = {}
        relations: dict[tuple[str, str, str], ExtractedRelation] = {}
        alias_map = self._alias_map(chunks)

        for chunk in chunks:
            for sentence in self._sentences(chunk.text):
                surface_sentence = re.sub(_ALIAS_RE, lambda match: match.group("label"), sentence)

                for match in _EXPLICIT_RELATION_RE.finditer(surface_sentence):
                    left_label = self._resolve_alias(match.group("left"), alias_map)
                    right_label = self._resolve_alias(match.group("right"), alias_map)
                    relation_phrase = match.group("relation").strip().replace(" ", "_")
                    relation_type = canonicalize_relation_type(relation_phrase)

                    self._record_relation(
                        entities,
                        relations,
                        left_label=left_label,
                        right_label=right_label,
                        relation_type=relation_type,
                        chunk_id=chunk.chunk_id,
                        sentence=surface_sentence,
                    )

                for match in _PASSIVE_RELATION_RE.finditer(surface_sentence):
                    source_label = self._resolve_alias(match.group("source"), alias_map)
                    target_label = self._resolve_alias(match.group("target"), alias_map)
                    relation_phrase = match.group("relation").strip().replace(" ", "_")
                    relation_type = canonicalize_relation_type(relation_phrase)
                    self._record_relation(
                        entities,
                        relations,
                        left_label=source_label,
                        right_label=target_label,
                        relation_type=relation_type,
                        chunk_id=chunk.chunk_id,
                        sentence=surface_sentence,
                    )

                seen_mentions: set[str] = set()
                for match in _ENTITY_MENTION_RE.finditer(surface_sentence):
                    label = self._resolve_alias(match.group(0), alias_map)
                    normalized_label = re.sub(r"\s+", " ", label).strip()
                    if normalized_label in seen_mentions:
                        continue
                    seen_mentions.add(normalized_label)
                    if _infer_entity_type(normalized_label) == "concept" and normalized_label not in alias_map.values():
                        continue
                    entity = self._build_entity(normalized_label, chunk_id=chunk.chunk_id, sentence=surface_sentence)
                    existing = entities.get(entity.key)
                    entities[entity.key] = self._merge_entity(existing, entity)

        return EntityRelationExtractionResult(
            entities=list(entities.values()),
            relations=list(relations.values()),
        )

    @staticmethod
    def _sentences(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", normalized) if part.strip()]

    @staticmethod
    def _build_entity(label: str, *, chunk_id: str, sentence: str) -> ExtractedEntity:
        normalized_label = re.sub(r"\s+", " ", label).strip()
        entity_key = normalize_entity_key(normalized_label)
        acronym = _label_acronym(normalized_label)
        aliases = [normalized_label]
        if acronym is not None:
            aliases.append(acronym)
        return ExtractedEntity(
            key=entity_key,
            label=normalized_label,
            entity_type=_infer_entity_type(normalized_label),
            description=sentence.strip(),
            source_chunk_ids=[chunk_id],
            metadata={"aliases": "||".join(aliases)},
        )

    @staticmethod
    def _alias_map(chunks: Sequence[Chunk]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for chunk in chunks:
            for match in _ALIAS_RE.finditer(chunk.text):
                label = match.group("label").strip()
                alias = match.group("alias").strip()
                aliases[alias] = label
        return aliases

    @staticmethod
    def _resolve_alias(label: str, alias_map: dict[str, str]) -> str:
        normalized = re.sub(r"\s+", " ", label).strip()
        return alias_map.get(normalized, normalized)

    @classmethod
    def _record_relation(
        cls,
        entities: dict[str, ExtractedEntity],
        relations: dict[tuple[str, str, str], ExtractedRelation],
        *,
        left_label: str,
        right_label: str,
        relation_type: str,
        chunk_id: str,
        sentence: str,
    ) -> None:
        left_entity = cls._build_entity(left_label, chunk_id=chunk_id, sentence=sentence)
        right_entity = cls._build_entity(right_label, chunk_id=chunk_id, sentence=sentence)
        for entity in (left_entity, right_entity):
            existing = entities.get(entity.key)
            entities[entity.key] = cls._merge_entity(existing, entity)

        relation_pair = canonicalize_relation_pair(left_entity.key, right_entity.key, relation_type)
        relation_key = (*relation_pair, relation_type)
        existing_relation = relations.get(relation_key)
        relation = ExtractedRelation(
            source_key=relation_key[0],
            target_key=relation_key[1],
            relation_type=relation_type,
            description=sentence.strip(),
            confidence=1.0,
            source_chunk_ids=[chunk_id],
        )
        relations[relation_key] = cls._merge_relation(existing_relation, relation)

    @staticmethod
    def _merge_entity(existing: ExtractedEntity | None, current: ExtractedEntity) -> ExtractedEntity:
        if existing is None:
            return current
        aliases = [
            *[alias for alias in existing.metadata.get("aliases", "").split("||") if alias],
            *[alias for alias in current.metadata.get("aliases", "").split("||") if alias],
        ]
        label = choose_preferred_label(aliases) or current.label or existing.label
        return ExtractedEntity(
            key=current.key,
            label=label,
            entity_type=existing.entity_type if existing.entity_type != "concept" else current.entity_type,
            description=_choose_longer(existing.description, current.description),
            source_chunk_ids=list(dict.fromkeys([*existing.source_chunk_ids, *current.source_chunk_ids])),
            metadata={"aliases": "||".join(sorted(dict.fromkeys(aliases)))},
        )

    @staticmethod
    def _merge_relation(existing: ExtractedRelation | None, current: ExtractedRelation) -> ExtractedRelation:
        if existing is None:
            return current
        return ExtractedRelation(
            source_key=current.source_key,
            target_key=current.target_key,
            relation_type=current.relation_type,
            description=_choose_longer(existing.description, current.description),
            confidence=max(existing.confidence, current.confidence),
            source_chunk_ids=list(dict.fromkeys([*existing.source_chunk_ids, *current.source_chunk_ids])),
            metadata={**existing.metadata, **current.metadata},
        )


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


def _infer_entity_type(label: str) -> str:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", label)]
    for token in reversed(tokens):
        entity_type = _ENTITY_TYPE_SUFFIXES.get(token)
        if entity_type is not None:
            return entity_type
    return "concept"


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
                    "doc_id": str(document.doc_id),
                    "doc_ids": str(document.doc_id),
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
                    "doc_id": str(document.doc_id),
                    "doc_ids": str(document.doc_id),
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
