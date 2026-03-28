from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import EmbeddingProviderBinding
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.search.sqlite_vector_repo import SQLiteVectorRepo
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.runtime.adapters import SearchBackedRetrievalFactory
from pkp.types import (
    AccessPolicy,
    Chunk,
    Document,
    DocumentType,
    ExecutionLocationPreference,
    GraphEdge,
    GraphNode,
    Segment,
    Source,
    SourceType,
)


@dataclass
class StubEmbeddingProvider:
    vectors: dict[str, list[float]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vectors[text]) for text in texts]


def test_search_backed_retrieval_factory_local_retriever_returns_seed_and_neighbor_evidence(
    tmp_path: Path,
) -> None:
    metadata_repo, vector_repo, graph_repo = _seed_graph_backed_retrieval_state(tmp_path)
    factory = SearchBackedRetrievalFactory(
        metadata_repo=metadata_repo,
        fts_repo=SQLiteFTSRepo(tmp_path / "fts.sqlite3"),
        graph_repo=graph_repo,
    )
    binding = EmbeddingProviderBinding(
        provider=StubEmbeddingProvider({"Alpha Engine": [1.0, 0.0]}),
        space="default",
        location="local",
    )
    retriever = factory.local_retriever_from_repo(
        vector_repo,
        [binding],
        default_preference=ExecutionLocationPreference.LOCAL_ONLY,
    )
    retriever.prepare_for_policy(
        access_policy=AccessPolicy.default(),
        execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
    )

    results = retriever("Alpha Engine", ["doc-1"])

    assert [candidate.chunk_id for candidate in results[:2]] == ["chunk-1", "chunk-2"]
    assert results[0].score > results[1].score


def test_search_backed_retrieval_factory_global_retriever_returns_relation_supported_evidence(
    tmp_path: Path,
) -> None:
    metadata_repo, vector_repo, graph_repo = _seed_graph_backed_retrieval_state(tmp_path)
    factory = SearchBackedRetrievalFactory(
        metadata_repo=metadata_repo,
        fts_repo=SQLiteFTSRepo(tmp_path / "fts.sqlite3"),
        graph_repo=graph_repo,
    )
    binding = EmbeddingProviderBinding(
        provider=StubEmbeddingProvider({"depends on": [0.0, 1.0]}),
        space="default",
        location="local",
    )
    retriever = factory.global_retriever_from_repo(
        vector_repo,
        [binding],
        default_preference=ExecutionLocationPreference.LOCAL_ONLY,
    )
    retriever.prepare_for_policy(
        access_policy=AccessPolicy.default(),
        execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
    )

    results = retriever("depends on", ["doc-1"])

    assert [candidate.chunk_id for candidate in results[:2]] == ["chunk-2", "chunk-1"]
    assert results[0].score > results[1].score


def _seed_graph_backed_retrieval_state(
    tmp_path: Path,
) -> tuple[SQLiteMetadataRepo, SQLiteVectorRepo, SQLiteGraphRepo]:
    metadata_repo = SQLiteMetadataRepo(tmp_path / "metadata.sqlite3")
    vector_repo = SQLiteVectorRepo(tmp_path / "vectors.sqlite3")
    graph_repo = SQLiteGraphRepo(tmp_path / "graph.sqlite3")
    policy = AccessPolicy.default()

    source = Source(
        source_id="src-1",
        source_type=SourceType.MARKDOWN,
        location="memory://alpha-beta",
        owner="tester",
        content_hash="hash-1",
        effective_access_policy=policy,
        ingest_version=1,
    )
    document = Document(
        doc_id="doc-1",
        source_id=source.source_id,
        doc_type=DocumentType.ARTICLE,
        title="Alpha Beta Notes",
        authors=["tester"],
        created_at=datetime.now(UTC),
        language="en",
        effective_access_policy=policy,
    )
    segment = Segment(
        segment_id="seg-1",
        doc_id=document.doc_id,
        parent_segment_id=None,
        toc_path=["Alpha Beta Notes"],
        heading_level=1,
        page_range=(1, 1),
        order_index=0,
        anchor="alpha-beta",
    )
    chunk_one = Chunk(
        chunk_id="chunk-1",
        segment_id=segment.segment_id,
        doc_id=document.doc_id,
        text="Alpha Engine processes ingestion requests.",
        token_count=6,
        citation_anchor="#chunk-1",
        citation_span=(0, 40),
        effective_access_policy=policy,
        extraction_quality=0.95,
        embedding_ref=None,
        order_index=0,
    )
    chunk_two = Chunk(
        chunk_id="chunk-2",
        segment_id=segment.segment_id,
        doc_id=document.doc_id,
        text="Beta Service depends on Alpha Engine for upstream context.",
        token_count=9,
        citation_anchor="#chunk-2",
        citation_span=(41, 97),
        effective_access_policy=policy,
        extraction_quality=0.95,
        embedding_ref=None,
        order_index=1,
    )

    metadata_repo.save_source(source)
    metadata_repo.save_document(document, location=source.location, content_hash=source.content_hash)
    metadata_repo.save_segment(segment)
    metadata_repo.save_chunk(chunk_one)
    metadata_repo.save_chunk(chunk_two)

    alpha = GraphNode(node_id="entity-alpha", node_type="entity", label="Alpha Engine")
    beta = GraphNode(node_id="entity-beta", node_type="entity", label="Beta Service")
    relation = GraphEdge(
        edge_id="rel-alpha-beta",
        from_node_id=alpha.node_id,
        to_node_id=beta.node_id,
        relation_type="depends_on",
        confidence=0.94,
        evidence_chunk_ids=["chunk-2"],
    )
    graph_repo.save_node(alpha)
    graph_repo.bind_node_evidence(alpha.node_id, ["chunk-1"])
    graph_repo.save_node(beta)
    graph_repo.bind_node_evidence(beta.node_id, ["chunk-2"])
    graph_repo.save_edge(relation)

    vector_repo.upsert(
        alpha.node_id,
        [1.0, 0.0],
        metadata={"doc_id": document.doc_id, "segment_id": "", "text": "Alpha Engine"},
        item_kind="entity",
    )
    vector_repo.upsert(
        relation.edge_id,
        [0.0, 1.0],
        metadata={
            "doc_id": document.doc_id,
            "segment_id": "",
            "text": "Alpha Engine depends_on Beta Service",
        },
        item_kind="relation",
    )
    return metadata_repo, vector_repo, graph_repo
