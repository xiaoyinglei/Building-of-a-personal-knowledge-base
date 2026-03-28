from __future__ import annotations

from pkp.core.rag_core import RAGCore
from pkp.core.storage_config import StorageConfig
from pkp.algorithms.chunking.multimodal_chunk_router import special_type_for_element
from pkp.algorithms.chunking.structured_chunker import ChunkSeed, merge_adjacent_seeds
from pkp.algorithms.chunking.token_chunker import chunk_by_tokens
from pkp.repo.interfaces import ParsedElement
from pkp.types.storage import CacheEntry, DocumentPipelineStage, DocumentProcessingStatus, DocumentStatusRecord


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
    paragraph = ParsedElement(element_id="para-1", kind="paragraph", text="plain text")

    assert special_type_for_element(table) == "table"
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


def test_ragcore_insert_persists_document_chunks_entities_relations_and_status() -> None:
    core = RAGCore(storage=StorageConfig.in_memory())
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
            edge
            for edge in core.stores.graph.list_edges()
            if edge.metadata.get("doc_id") == result.document_id
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
