from __future__ import annotations

from datetime import UTC, datetime

from rag.ingest.chunkers.multimodal_chunk_router import special_type_for_element
from rag.ingest.chunkers.structured_chunker import ChunkSeed, merge_adjacent_seeds
from rag.ingest.chunkers.token_chunker import chunk_by_tokens
from rag.ingest.extract import EntityRelationExtractionResult, ExtractedEntity
from rag.ingest.pipeline import IngestRequest
from rag.schema import AccessPolicy, Chunk, DocumentType, SourceType
from rag.schema.core import Document, ParsedDocument, ParsedElement, ParsedSection
from rag.schema.runtime import CacheEntry, DocumentPipelineStage, DocumentProcessingStatus, DocumentStatusRecord
from rag.storage import StorageConfig
from tests.support import make_runtime


def test_token_chunker_produces_stable_child_chunks() -> None:
    chunks = chunk_by_tokens(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        chunk_token_size=4,
        chunk_overlap_token_size=1,
    )

    assert len(chunks) >= 3
    assert chunks[0].order_index == 0
    assert all(chunk.text for chunk in chunks)


def test_structured_chunker_merges_adjacent_markdown_seeds_with_same_path() -> None:
    seeds = [
        ChunkSeed(text="Overview", toc_path=["Quarterly Review", "Revenue"], page_numbers=[1]),
        ChunkSeed(text="expanded", toc_path=["Quarterly Review", "Revenue"], page_numbers=[1]),
    ]

    merged = merge_adjacent_seeds(seeds, source_type="markdown")

    assert len(merged) == 1
    assert merged[0].text == "Overview expanded"


def test_multimodal_router_classifies_special_elements() -> None:
    table = ParsedElement(element_id="table-1", kind="table", text="q1,q2")
    formula = ParsedElement(element_id="eq-1", kind="equation", text="E = mc^2")
    paragraph = ParsedElement(element_id="para-1", kind="paragraph", text="plain text")

    assert special_type_for_element(table) == "table"
    assert special_type_for_element(formula) == "formula"
    assert special_type_for_element(paragraph) is None


def test_storage_groups_include_document_chunk_status_graph_cache() -> None:
    storage = StorageConfig.in_memory().build()
    try:
        assert storage.documents is not None
        assert storage.chunks is not None
        assert storage.vectors is not None
        assert storage.graph is not None
        assert storage.status is not None
        assert storage.cache is not None

        saved_status = storage.status.save(
            DocumentStatusRecord(
                doc_id="doc-1",
                source_id="src-1",
                location="memory://alpha",
                content_hash="hash-1",
                status=DocumentProcessingStatus.PROCESSING,
                stage=DocumentPipelineStage.CHUNK,
                attempts=1,
            )
        )
        saved_cache = storage.cache.save(
            CacheEntry(
                namespace="extract",
                cache_key="doc-1::entities",
                payload={"entity_count": 3},
            )
        )

        assert storage.status.get("doc-1") == saved_status
        assert storage.cache.get("doc-1::entities", namespace="extract") == saved_cache
    finally:
        storage.close()


def test_graph_extraction_aggregates_document_chunks_into_single_llm_call() -> None:
    class _RecordingExtractor:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def extract(self, *, document: Document, chunks: list[Chunk]) -> EntityRelationExtractionResult:
            del document
            self.calls.append([chunk.chunk_id for chunk in chunks])
            return EntityRelationExtractionResult(
                entities=[
                    ExtractedEntity(
                        key="alpha_entity",
                        label="Alpha Entity",
                        entity_type="concept",
                        description="Aggregated entity",
                        source_chunk_ids=[chunk.chunk_id for chunk in chunks],
                    )
                ],
                relations=[],
            )

    core = make_runtime()
    try:
        extractor = _RecordingExtractor()
        core.ingest_pipeline.extractor = extractor
        document = Document(
            doc_id=1,
            source_id=1,
            doc_type=DocumentType.ARTICLE,
            title="Aggregated Graph",
            authors=[],
            file_hash="graph-doc-hash",
            version_group_id=1,
            created_at=datetime.now(UTC),
            language="zh",
            effective_access_policy=AccessPolicy.default(),
        )
        chunks = [
            Chunk(
                chunk_id="chunk-1",
                segment_id="seg-1",
                doc_id=str(document.doc_id),
                text="Alpha Entity appears in the first chunk.",
                token_count=8,
                citation_anchor="#chunk-1",
                citation_span=(0, 40),
                effective_access_policy=AccessPolicy.default(),
                extraction_quality=0.9,
                embedding_ref=None,
                order_index=0,
                content_hash="hash-1",
            ),
            Chunk(
                chunk_id="chunk-2",
                segment_id="seg-2",
                doc_id=str(document.doc_id),
                text="Alpha Entity is referenced again in the second chunk.",
                token_count=10,
                citation_anchor="#chunk-2",
                citation_span=(41, 95),
                effective_access_policy=AccessPolicy.default(),
                extraction_quality=0.9,
                embedding_ref=None,
                order_index=1,
                content_hash="hash-2",
            ),
        ]

        first = core.ingest_pipeline._extract_entities_and_relations(document=document, chunks=chunks)
        second = core.ingest_pipeline._extract_entities_and_relations(document=document, chunks=chunks)

        assert len(extractor.calls) == 1
        assert extractor.calls[0] == ["chunk-1", "chunk-2"]
        assert first.entities[0].source_chunk_ids == ["chunk-1", "chunk-2"]
        assert second.entities[0].source_chunk_ids == ["chunk-1", "chunk-2"]
    finally:
        core.stores.close()


def test_ragcore_insert_persists_document_chunks_entities_relations_and_status() -> None:
    core = make_runtime()
    try:
        result = core.insert(
            location="memory://reliability.md",
            source_type="markdown",
            owner="user",
            title="Reliability Graph",
            content_text=(
                "# Reliability Graph\n\n"
                "Evidence Quality supports Reliable Retrieval.\n\n"
                "Context Fusion uses Evidence Quality."
            ),
        )

        status = core.stores.status.get(result.document_id)
        entity_nodes = [
            node
            for node in core.stores.graph.list_nodes(node_type="entity")
            if node.metadata.get("doc_id") == result.document_id
        ]
        relation_edges = [
            edge for edge in core.stores.graph.list_edges() if edge.metadata.get("doc_id") == result.document_id
        ]

        assert result.document_id
        assert result.chunk_count > 0
        assert result.entity_count >= 2
        assert result.relation_count >= 1
        assert result.status == "ready"
        assert status is not None
        assert status.status is DocumentProcessingStatus.READY
        assert entity_nodes
        assert relation_edges
        assert core.stores.cache.list(namespace="extract")
    finally:
        core.stores.close()


def test_ragcore_merges_graph_and_entity_indexes_across_documents() -> None:
    core = make_runtime()
    try:
        first = core.insert(
            location="memory://alpha.txt",
            source_type="plain_text",
            owner="user",
            content_text="Alpha Engine supports Beta Service.",
        )
        second = core.insert(
            location="memory://beta.txt",
            source_type="plain_text",
            owner="user",
            content_text="Alpha Engine supports Beta Service.",
        )

        entity_nodes = sorted(core.stores.graph.list_nodes(node_type="entity"), key=lambda node: node.label)
        alpha_node = next(node for node in entity_nodes if node.label == "Alpha Engine")
        relation_edges = [edge for edge in core.stores.graph.list_edges() if edge.relation_type == "supports"]
        alpha_vector = core.stores.vector_repo.get_entry(alpha_node.node_id, item_kind="entity")

        assert [node.label for node in entity_nodes] == ["Alpha Engine", "Beta Service"]
        assert set(core.stores.graph.list_node_evidence_chunk_ids(alpha_node.node_id)) == {
            first.chunks[0].chunk_id,
            second.chunks[0].chunk_id,
        }
        assert len(relation_edges) == 1
        assert set(relation_edges[0].evidence_chunk_ids) == {
            first.chunks[0].chunk_id,
            second.chunks[0].chunk_id,
        }
        assert alpha_vector is not None
        assert set(alpha_vector.metadata.get("doc_ids", "").split(",")) == {
            first.document_id,
            second.document_id,
        }

        core.delete(doc_id=first.document_id)

        assert core.stores.graph.get_node(alpha_node.node_id) is not None
        assert set(core.stores.graph.list_node_evidence_chunk_ids(alpha_node.node_id)) == {second.chunks[0].chunk_id}
        remaining_edge = core.stores.graph.get_edge(relation_edges[0].edge_id)
        assert remaining_edge is not None
        assert remaining_edge.evidence_chunk_ids == [second.chunks[0].chunk_id]
    finally:
        core.stores.close()


def test_ragcore_insert_canonicalizes_alias_entities_and_relation_direction() -> None:
    core = make_runtime()
    try:
        core.insert(
            location="memory://alias-graph.txt",
            source_type="plain_text",
            owner="user",
            content_text=(
                "Alpha Engine (AE) supports Beta Service. "
                "Beta Service is supported by Alpha Engine. "
                "AE depends on Gamma Index."
            ),
        )

        entity_nodes = sorted(core.stores.graph.list_nodes(node_type="entity"), key=lambda node: node.label)
        edges = sorted(core.stores.graph.list_edges(), key=lambda edge: (edge.relation_type, edge.edge_id))

        alpha_node = next(node for node in entity_nodes if node.label == "Alpha Engine")
        beta_node = next(node for node in entity_nodes if node.label == "Beta Service")
        gamma_node = next(node for node in entity_nodes if node.label == "Gamma Index")
        supports_edges = [edge for edge in edges if edge.relation_type == "supports"]
        depends_edges = [edge for edge in edges if edge.relation_type == "depends_on"]

        assert [node.label for node in entity_nodes] == ["Alpha Engine", "Beta Service", "Gamma Index"]
        assert len(supports_edges) == 1
        assert supports_edges[0].from_node_id == alpha_node.node_id
        assert supports_edges[0].to_node_id == beta_node.node_id
        assert len(depends_edges) == 1
        assert depends_edges[0].from_node_id == alpha_node.node_id
        assert depends_edges[0].to_node_id == gamma_node.node_id
    finally:
        core.stores.close()


def test_ragcore_insert_many_processes_multiple_requests() -> None:
    core = make_runtime()
    try:
        result = core.insert_many(
            [
                IngestRequest(
                    location="memory://alpha.txt",
                    source_type="plain_text",
                    owner="user",
                    content_text="Alpha Engine supports Beta Service.",
                ),
                IngestRequest(
                    location="memory://beta.txt",
                    source_type="plain_text",
                    owner="user",
                    content_text="Gamma Index depends on Alpha Engine.",
                ),
            ]
        )

        assert result.success_count == 2
        assert result.failure_count == 0
        assert len(result.results) == 2
        assert result.errors == []
        assert {item.document_id for item in result.results}
    finally:
        core.stores.close()


def test_ragcore_merges_alias_entities_across_documents_when_alias_is_known() -> None:
    core = make_runtime()
    try:
        first = core.insert(
            location="memory://alpha-alias.txt",
            source_type="plain_text",
            owner="user",
            content_text="Alpha Engine (AE) supports Beta Service.",
        )
        second = core.insert(
            location="memory://alpha-followup.txt",
            source_type="plain_text",
            owner="user",
            content_text="AE depends on Gamma Index.",
        )

        entity_nodes = sorted(core.stores.graph.list_nodes(node_type="entity"), key=lambda node: node.label)
        alpha_node = next(node for node in entity_nodes if node.label == "Alpha Engine")
        gamma_node = next(node for node in entity_nodes if node.label == "Gamma Index")
        depends_edge = next(edge for edge in core.stores.graph.list_edges() if edge.relation_type == "depends_on")
        alpha_vector = core.stores.vector_repo.get_entry(alpha_node.node_id, item_kind="entity")

        assert [node.label for node in entity_nodes] == ["Alpha Engine", "Beta Service", "Gamma Index"]
        assert set(core.stores.graph.list_node_evidence_chunk_ids(alpha_node.node_id)) == {
            first.chunks[0].chunk_id,
            second.chunks[0].chunk_id,
        }
        assert depends_edge.from_node_id == alpha_node.node_id
        assert depends_edge.to_node_id == gamma_node.node_id
        assert alpha_vector is not None
        assert set(alpha_vector.metadata.get("doc_ids", "").split(",")) == {
            first.document_id,
            second.document_id,
        }
        assert "AE" in alpha_node.metadata.get("aliases", "").split("||")
    finally:
        core.stores.close()


def test_ragcore_insert_links_multimodal_nodes_into_graph_across_sections() -> None:
    core = make_runtime()
    try:
        parsed = ParsedDocument(
            title="Alpha Metrics",
            source_type=SourceType.MARKDOWN,
            doc_type=DocumentType.ARTICLE,
            authors=["tester"],
            language="en",
            visible_text=(
                "Alpha Engine coordinates ingestion. Metrics section contains a table about Alpha Engine throughput."
            ),
            sections=[
                ParsedSection(
                    toc_path=("Alpha Metrics", "Overview"),
                    heading_level=1,
                    page_range=(1, 1),
                    order_index=0,
                    text="Alpha Engine coordinates ingestion.",
                ),
                ParsedSection(
                    toc_path=("Alpha Metrics", "Metrics"),
                    heading_level=1,
                    page_range=(2, 2),
                    order_index=1,
                    text="Metrics section contains a table about Alpha Engine throughput.",
                ),
            ],
            elements=[
                ParsedElement(
                    element_id="table-1",
                    kind="table",
                    text="Alpha Engine throughput is 99 and latency is 12ms.",
                    toc_path=("Alpha Metrics", "Metrics"),
                    page_no=2,
                )
            ],
        )

        result = core.insert(
            location="memory://alpha-metrics.md",
            source_type="markdown",
            owner="user",
            content_text=parsed.visible_text,
            parsed_document=parsed,
        )

        table_node = next(node for node in core.stores.graph.list_nodes(node_type="table"))
        alpha_node = next(
            node for node in core.stores.graph.list_nodes(node_type="entity") if node.label == "Alpha Engine"
        )
        table_vector = core.stores.vector_repo.get_entry(table_node.node_id, item_kind="multimodal")
        table_chunk = next(chunk for chunk in result.chunks if chunk.special_chunk_type == "table")
        linking_edges = [
            edge
            for edge in core.stores.graph.list_edges_for_node(alpha_node.node_id)
            if edge.to_node_id == table_node.node_id or edge.from_node_id == table_node.node_id
        ]

        assert table_node.metadata.get("special_chunk_type") == "table"
        assert core.stores.graph.list_node_evidence_chunk_ids(table_node.node_id) == [table_chunk.chunk_id]
        assert any(edge.relation_type == "tabulated_in" for edge in linking_edges)
        assert table_vector is not None
        assert table_vector.metadata.get("chunk_id") == table_chunk.chunk_id
    finally:
        core.stores.close()
