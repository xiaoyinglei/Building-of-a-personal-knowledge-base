from __future__ import annotations

from datetime import UTC, datetime

from pkp.ingest.extract import HeuristicEntityRelationExtractor
from pkp.schema._types import AccessPolicy, Chunk, Document, DocumentType


def test_heuristic_extractor_canonicalizes_aliases_and_relation_direction() -> None:
    extractor = HeuristicEntityRelationExtractor()
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
