from __future__ import annotations

from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field

from pkp.algorithms.extract.entity_relation_extractor import EntityRelationExtractionResult
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
            node_id = self._deterministic_id(entity.key, "entity")
            existing = merged_entities.get(entity.key)
            evidence_chunk_ids = self._merge_chunk_ids(
                [] if existing is None else existing.evidence_chunk_ids,
                entity.source_chunk_ids,
            )
            description = entity.description if existing is None else self._choose_description(
                existing.description,
                entity.description,
            )
            merged_entities[entity.key] = MergedEntity(
                node_id=node_id,
                key=entity.key,
                label=entity.label if existing is None else existing.label,
                entity_type=entity.entity_type if existing is None else existing.entity_type,
                description=description,
                evidence_chunk_ids=evidence_chunk_ids,
                metadata={
                    "doc_id": document.doc_id,
                    "doc_ids": document.doc_id,
                    "entity_key": entity.key,
                    "evidence_count": str(len(evidence_chunk_ids)),
                },
            )

        merged_relations: dict[tuple[str, str, str], MergedRelation] = {}
        for relation in extraction.relations:
            source = merged_entities.get(relation.source_key)
            target = merged_entities.get(relation.target_key)
            if source is None or target is None:
                continue
            relation_key = (source.node_id, target.node_id, relation.relation_type)
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
                    relation.relation_type,
                    "edge",
                ),
                from_node_id=source.node_id,
                to_node_id=target.node_id,
                relation_type=relation.relation_type,
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
