from __future__ import annotations

from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field

from pkp.algorithms.extract.entity_relation_extractor import (
    EntityRelationExtractionResult,
    canonicalize_relation_pair,
    canonicalize_relation_type,
    choose_preferred_label,
    normalize_entity_key,
)
from pkp.types.content import Document


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
