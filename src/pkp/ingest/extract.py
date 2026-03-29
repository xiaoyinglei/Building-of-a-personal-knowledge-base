from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from pkp.repo.interfaces import ModelProviderRepo
from pkp.schema.chunk import Chunk
from pkp.schema.document import Document

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
_UPPER_CASE_RE = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
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

_GENERIC_SECTION_TERMS = {
    "overview",
    "summary",
    "introduction",
    "conclusion",
    "appendix",
    "background",
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


@dataclass(frozen=True)
class RelationRule:
    pattern: re.Pattern[str]
    relation_type: str
    confidence: float
    direction: str = "forward"


@dataclass(frozen=True)
class EntityMention:
    start: int
    end: int
    entity: ExtractedEntity


_RELATION_RULES: tuple[RelationRule, ...] = (
    RelationRule(re.compile(r"\bsupports?\b", re.IGNORECASE), "supports", 0.88, "forward"),
    RelationRule(re.compile(r"\bsupported by\b", re.IGNORECASE), "supports", 0.88, "reverse"),
    RelationRule(re.compile(r"\bdepends on\b", re.IGNORECASE), "depends_on", 0.9, "forward"),
    RelationRule(re.compile(r"\bdependency on\b", re.IGNORECASE), "depends_on", 0.86, "forward"),
    RelationRule(re.compile(r"\bdependent on\b", re.IGNORECASE), "depends_on", 0.84, "forward"),
    RelationRule(re.compile(r"\buses?\b", re.IGNORECASE), "uses", 0.78, "forward"),
    RelationRule(re.compile(r"\bused by\b", re.IGNORECASE), "uses", 0.78, "reverse"),
    RelationRule(re.compile(r"\bcontains?\b", re.IGNORECASE), "contains", 0.84, "forward"),
    RelationRule(re.compile(r"\bincludes?\b", re.IGNORECASE), "contains", 0.84, "forward"),
    RelationRule(re.compile(r"\bincluded in\b", re.IGNORECASE), "contains", 0.82, "reverse"),
    RelationRule(re.compile(r"\benables?\b", re.IGNORECASE), "enables", 0.83, "forward"),
    RelationRule(re.compile(r"\benabled by\b", re.IGNORECASE), "enables", 0.83, "reverse"),
    RelationRule(re.compile(r"\brequires?\b", re.IGNORECASE), "requires", 0.82, "forward"),
    RelationRule(re.compile(r"\brequired by\b", re.IGNORECASE), "requires", 0.82, "reverse"),
    RelationRule(re.compile(r"\bpart of\b", re.IGNORECASE), "part_of", 0.82, "forward"),
    RelationRule(re.compile(r"\bbelongs to\b", re.IGNORECASE), "part_of", 0.82, "forward"),
    RelationRule(re.compile(r"\bintegrates with\b", re.IGNORECASE), "integrates_with", 0.78, "undirected"),
    RelationRule(re.compile(r"\bconnects? to\b", re.IGNORECASE), "integrates_with", 0.76, "undirected"),
    RelationRule(re.compile(r"\blinked to\b", re.IGNORECASE), "integrates_with", 0.74, "undirected"),
    RelationRule(re.compile(r"\bcompare[sd]?\b", re.IGNORECASE), "compares", 0.75, "undirected"),
    RelationRule(re.compile(r"\bcontrasts?\b", re.IGNORECASE), "compares", 0.75, "undirected"),
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

        label_groups = self._group_labels(labels)
        description = self._description_for(chunk.text)
        entities: list[ExtractedEntity] = []
        for key, aliases in label_groups.items():
            preferred_label = choose_preferred_label(aliases)
            if not key or not preferred_label:
                continue
            entities.append(
                ExtractedEntity(
                    key=key,
                    label=preferred_label,
                    entity_type=self._infer_entity_type(preferred_label),
                    description=description,
                    source_chunk_ids=[chunk.chunk_id],
                    metadata={
                        "aliases": "||".join(sorted(dict.fromkeys(aliases))),
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
            mentions = self._entity_mentions(sentence, entities)
            if len(mentions) < 2:
                continue

            sentence_relations = self._extract_sentence_relations(sentence, mentions, chunk.chunk_id)
            if not sentence_relations:
                sentence_relations = self._fallback_relations(sentence, mentions, chunk.chunk_id)

            for relation in sentence_relations:
                relation_key = (relation.source_key, relation.target_key, relation.relation_type)
                if relation.source_key == relation.target_key or relation_key in seen_keys:
                    continue
                seen_keys.add(relation_key)
                relations.append(relation)
        return relations

    def _candidate_labels(self, chunk: Chunk) -> list[str]:
        labels: list[str] = []
        for pattern in (_TITLE_CASE_RE, _UPPER_CASE_RE):
            for match in pattern.findall(chunk.text):
                normalized = match.strip(".,:;()[]{} ")
                if normalized and normalized not in labels:
                    labels.append(normalized)

        if labels:
            return labels[:8]

        toc_tail = chunk.metadata.get("toc_path", "").split(" > ")[-1].strip()
        if toc_tail and len(toc_tail) > 3 and toc_tail.lower() not in _GENERIC_SECTION_TERMS:
            labels.append(toc_tail)
            return labels

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

    def _group_labels(self, labels: Sequence[str]) -> dict[str, list[str]]:
        long_labels = [label for label in labels if len(label.split()) >= 2 and not _is_acronym(label)]
        canonical_by_acronym = {
            acronym: label
            for label in sorted(long_labels, key=len, reverse=True)
            if (acronym := _label_acronym(label)) is not None
        }

        grouped: dict[str, list[str]] = {}
        for label in labels:
            normalized_label = label.strip(".,:;()[]{} ")
            if not normalized_label:
                continue
            canonical_label = canonical_by_acronym.get(normalized_label.upper(), normalized_label)
            key = normalize_entity_key(canonical_label)
            if not key:
                continue
            grouped.setdefault(key, []).extend([canonical_label, normalized_label])
        return {key: list(dict.fromkeys(values)) for key, values in grouped.items()}

    def _entity_mentions(
        self,
        sentence: str,
        entities: Sequence[ExtractedEntity],
    ) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        seen: set[tuple[str, int, int]] = set()
        for entity in entities:
            aliases = self._entity_aliases(entity)
            for alias in aliases:
                if len(alias) < 2:
                    continue
                pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
                for match in pattern.finditer(sentence):
                    key = (entity.key, match.start(), match.end())
                    if key in seen:
                        continue
                    seen.add(key)
                    mentions.append(EntityMention(match.start(), match.end(), entity))
        mentions.sort(key=lambda item: (item.start, -(item.end - item.start), item.entity.key))
        collapsed: list[EntityMention] = []
        for mention in mentions:
            if collapsed and mention.start < collapsed[-1].end and mention.entity.key == collapsed[-1].entity.key:
                if (mention.end - mention.start) > (collapsed[-1].end - collapsed[-1].start):
                    collapsed[-1] = mention
                continue
            collapsed.append(mention)
        return collapsed

    def _extract_sentence_relations(
        self,
        sentence: str,
        mentions: Sequence[EntityMention],
        chunk_id: str,
    ) -> list[ExtractedRelation]:
        relations: list[ExtractedRelation] = []
        for rule in _RELATION_RULES:
            for match in rule.pattern.finditer(sentence):
                left = self._nearest_left_mention(mentions, match.start())
                right = self._nearest_right_mention(mentions, match.end())
                if left is None or right is None or left.entity.key == right.entity.key:
                    continue
                source_key, target_key = self._apply_direction(
                    left.entity.key,
                    right.entity.key,
                    relation_type=rule.relation_type,
                    direction=rule.direction,
                )
                source_key, target_key = canonicalize_relation_pair(source_key, target_key, rule.relation_type)
                relations.append(
                    ExtractedRelation(
                        source_key=source_key,
                        target_key=target_key,
                        relation_type=canonicalize_relation_type(rule.relation_type),
                        description=self._description_for(sentence),
                        confidence=rule.confidence,
                        source_chunk_ids=[chunk_id],
                        metadata={"source_chunk_id": chunk_id},
                    )
                )
        return relations

    def _fallback_relations(
        self,
        sentence: str,
        mentions: Sequence[EntityMention],
        chunk_id: str,
    ) -> list[ExtractedRelation]:
        relations: list[ExtractedRelation] = []
        seen_pairs: set[tuple[str, str]] = set()
        for left, right in zip(mentions, mentions[1:], strict=False):
            source_key, target_key = canonicalize_relation_pair(
                left.entity.key,
                right.entity.key,
                "related_to",
            )
            if source_key == target_key or (source_key, target_key) in seen_pairs:
                continue
            seen_pairs.add((source_key, target_key))
            relations.append(
                ExtractedRelation(
                    source_key=source_key,
                    target_key=target_key,
                    relation_type="related_to",
                    description=self._description_for(sentence),
                    confidence=0.45,
                    source_chunk_ids=[chunk_id],
                    metadata={"source_chunk_id": chunk_id},
                )
            )
        return relations

    @staticmethod
    def _entity_aliases(entity: ExtractedEntity) -> list[str]:
        aliases = entity.metadata.get("aliases", "")
        values = [entity.label]
        if aliases:
            values.extend(alias for alias in aliases.split("||") if alias)
        return list(dict.fromkeys(values))

    @staticmethod
    def _nearest_left_mention(mentions: Sequence[EntityMention], offset: int) -> EntityMention | None:
        candidates = [mention for mention in mentions if mention.end <= offset]
        return None if not candidates else max(candidates, key=lambda mention: mention.end)

    @staticmethod
    def _nearest_right_mention(mentions: Sequence[EntityMention], offset: int) -> EntityMention | None:
        candidates = [mention for mention in mentions if mention.start >= offset]
        return None if not candidates else min(candidates, key=lambda mention: mention.start)

    @staticmethod
    def _apply_direction(
        left_key: str,
        right_key: str,
        *,
        relation_type: str,
        direction: str,
    ) -> tuple[str, str]:
        normalized_relation = canonicalize_relation_type(relation_type)
        if direction == "reverse":
            pair = (right_key, left_key)
        elif direction == "undirected":
            pair = canonicalize_relation_pair(left_key, right_key, normalized_relation)
            return pair
        else:
            pair = (left_key, right_key)
        return canonicalize_relation_pair(pair[0], pair[1], normalized_relation)

    @staticmethod
    def _description_for(text: str) -> str:
        normalized = " ".join(text.split())
        return normalized[:280]

    @staticmethod
    def _infer_entity_type(label: str) -> str:
        normalized_tokens = [token for token in re.split(r"[^A-Za-z0-9]+", label.lower()) if token]
        if not normalized_tokens:
            return "concept"
        last_token = _singularize_token(normalized_tokens[-1])
        return _ENTITY_TYPE_SUFFIXES.get(last_token, "concept")


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
        entity_by_key: dict[str, ExtractedEntity] = {}
        for entity in [*fallback_result.entities, *llm_result.entities]:
            canonical_key = normalize_entity_key(entity.key or entity.label)
            if not canonical_key:
                continue
            existing = entity_by_key.get(canonical_key)
            aliases = []
            if existing is not None:
                aliases.extend(existing.metadata.get("aliases", "").split("||"))
            aliases.extend(entity.metadata.get("aliases", "").split("||"))
            aliases.append(entity.label)
            preferred_label = choose_preferred_label([alias for alias in aliases if alias])
            entity_by_key[canonical_key] = ExtractedEntity(
                key=canonical_key,
                label=preferred_label or entity.label,
                entity_type=entity.entity_type if entity.entity_type != "concept" else (existing.entity_type if existing else "concept"),
                description=_choose_longer(existing.description if existing else "", entity.description),
                source_chunk_ids=list(dict.fromkeys([*(existing.source_chunk_ids if existing else []), *entity.source_chunk_ids])),
                metadata={
                    **({} if existing is None else existing.metadata),
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
            existing = relation_by_key.get(relation_key)
            relation_by_key[relation_key] = ExtractedRelation(
                source_key=source_key,
                target_key=target_key,
                relation_type=relation_type,
                description=_choose_longer(existing.description if existing else "", relation.description),
                confidence=max(relation.confidence, 0.0 if existing is None else existing.confidence),
                source_chunk_ids=list(dict.fromkeys([*(existing.source_chunk_ids if existing else []), *relation.source_chunk_ids])),
                metadata={**({} if existing is None else existing.metadata), **relation.metadata},
            )
        return EntityRelationExtractionResult(
            entities=list(entity_by_key.values()),
            relations=list(relation_by_key.values()),
        )


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
            existing = merged_entities.get(entity_key)
            evidence_chunk_ids = self._merge_chunk_ids(
                [] if existing is None else existing.evidence_chunk_ids,
                entity.source_chunk_ids,
            )
            description = entity.description if existing is None else self._choose_description(
                existing.description,
                entity.description,
            )
            aliases = []
            if existing is not None:
                aliases.extend(alias for alias in existing.metadata.get("aliases", "").split("||") if alias)
            aliases.extend(alias for alias in entity.metadata.get("aliases", "").split("||") if alias)
            aliases.append(entity.label)
            preferred_label = choose_preferred_label(aliases) or entity.label
            merged_entities[entity_key] = MergedEntity(
                node_id=node_id,
                key=entity_key,
                label=preferred_label,
                entity_type=entity.entity_type if existing is None or existing.entity_type == "concept" else existing.entity_type,
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
            existing = merged_relations.get(relation_key)
            evidence_chunk_ids = self._merge_chunk_ids(
                [] if existing is None else existing.evidence_chunk_ids,
                relation.source_chunk_ids,
            )
            description = relation.description if existing is None else self._choose_description(
                existing.description,
                relation.description,
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
                confidence=max(relation.confidence, 0.0 if existing is None else existing.confidence),
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
