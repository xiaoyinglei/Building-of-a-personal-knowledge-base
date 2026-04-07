from __future__ import annotations

import json
from datetime import UTC, datetime

from rag.ingest.extract import PromptedEntityRelationExtractor
from rag.schema._types import AccessPolicy, Chunk, Document, DocumentType


class FakeEntityRelationBackend:
    def chat(self, prompt: str) -> str:
        del prompt
        return json.dumps(
            {
                "entities": [
                    {
                        "key": "alpha_engine",
                        "label": "Alpha Engine",
                        "entity_type": "system",
                        "description": "Alpha Engine supports Beta Service and depends on Gamma Index.",
                        "source_chunk_ids": ["chunk-1"],
                    },
                    {
                        "key": "beta_service",
                        "label": "Beta Service",
                        "entity_type": "service",
                        "description": "Beta Service is supported by Alpha Engine.",
                        "source_chunk_ids": ["chunk-1"],
                    },
                    {
                        "key": "gamma_index",
                        "label": "Gamma Index",
                        "entity_type": "index",
                        "description": "Gamma Index is a dependency of Alpha Engine.",
                        "source_chunk_ids": ["chunk-1"],
                    },
                ],
                "relations": [
                    {
                        "source_key": "alpha_engine",
                        "target_key": "beta_service",
                        "relation_type": "supports",
                        "description": "Alpha Engine supports Beta Service.",
                        "confidence": 1.0,
                        "source_chunk_ids": ["chunk-1"],
                    },
                    {
                        "source_key": "alpha_engine",
                        "target_key": "gamma_index",
                        "relation_type": "depends_on",
                        "description": "Alpha Engine depends on Gamma Index.",
                        "confidence": 1.0,
                        "source_chunk_ids": ["chunk-1"],
                    },
                ],
            },
            ensure_ascii=False,
        )


def test_prompted_extractor_returns_grounded_entities_and_relations() -> None:
    extractor = PromptedEntityRelationExtractor(model_provider=FakeEntityRelationBackend())
    document = Document(
        doc_id="doc-1",
        source_id="src-1",
        doc_type=DocumentType.ARTICLE,
        title="Alias Graph",
        authors=["tester"],
        created_at=datetime.now(UTC),
        language="en",
        effective_access_policy=AccessPolicy.default(),
    )
    chunk = Chunk(
        chunk_id="chunk-1",
        segment_id="seg-1",
        doc_id=document.doc_id,
        text=(
            "Alpha Engine (AE) supports Beta Service. "
            "Beta Service is supported by Alpha Engine. "
            "AE depends on Gamma Index."
        ),
        token_count=18,
        citation_anchor="#chunk-1",
        citation_span=(0, 120),
        effective_access_policy=AccessPolicy.default(),
        extraction_quality=0.95,
        embedding_ref=None,
        order_index=0,
        metadata={"toc_path": "Alias Graph"},
    )

    result = extractor.extract(document=document, chunks=[chunk])

    assert {entity.key for entity in result.entities} == {
        "alpha_engine",
        "beta_service",
        "gamma_index",
    }
    supports = [relation for relation in result.relations if relation.relation_type == "supports"]
    depends = [relation for relation in result.relations if relation.relation_type == "depends_on"]
    assert len(supports) == 1
    assert supports[0].source_key == "alpha_engine"
    assert supports[0].target_key == "beta_service"
    assert len(depends) == 1
    assert depends[0].source_key == "alpha_engine"
    assert depends[0].target_key == "gamma_index"
