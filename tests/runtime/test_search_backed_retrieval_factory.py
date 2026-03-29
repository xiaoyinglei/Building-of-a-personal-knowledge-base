from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pkp.algorithms.retrieval.search_backed_factory import RetrievedCandidate, SearchBackedRetrievalFactory
from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import EmbeddingProviderBinding
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.search.sqlite_vector_repo import SQLiteVectorRepo
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
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

    assert [candidate.chunk_id for candidate in results[:3]] == ["chunk-1", "chunk-2", "chunk-table"]
    assert results[0].score > results[1].score
    assert results[2].special_chunk_type == "table"


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

    assert [candidate.chunk_id for candidate in results[:3]] == ["chunk-2", "chunk-1", "chunk-table"]
    assert results[0].score > results[1].score
    assert results[2].special_chunk_type == "table"


def test_search_backed_retrieval_factory_local_retriever_reaches_two_hop_neighbors(
    tmp_path: Path,
) -> None:
    metadata_repo, vector_repo, graph_repo = _seed_multi_hop_graph_state(tmp_path)
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
    chunk_ids = [candidate.chunk_id for candidate in results]

    assert chunk_ids[:3] == ["chunk-1", "chunk-2", "chunk-3"]
    assert results[1].score > results[2].score


def test_search_backed_retrieval_factory_graph_expander_follows_seed_paths(
    tmp_path: Path,
) -> None:
    metadata_repo, vector_repo, graph_repo = _seed_multi_hop_graph_state(tmp_path)
    del vector_repo
    factory = SearchBackedRetrievalFactory(
        metadata_repo=metadata_repo,
        fts_repo=SQLiteFTSRepo(tmp_path / "fts.sqlite3"),
        graph_repo=graph_repo,
    )

    seed_candidates = [
        RetrievedCandidate(
            chunk_id="chunk-1",
            doc_id="doc-1",
            source_id="src-1",
            text="Alpha Engine powers ingestion and coordinates upstream context.",
            citation_anchor="#chunk-1",
            score=1.0,
            rank=1,
            source_kind="internal",
        )
    ]
    graph_candidates = factory.graph_expander(
        "How does Alpha Engine connect to Gamma Index?",
        ["doc-1"],
        seed_candidates[:1],
    )

    assert [candidate.chunk_id for candidate in graph_candidates[:2]] == ["chunk-2", "chunk-3"]
    assert all(candidate.source_kind == "graph" for candidate in graph_candidates[:2])


def test_search_backed_retrieval_factory_local_retriever_reaches_multimodal_nodes_across_sections(
    tmp_path: Path,
) -> None:
    metadata_repo, vector_repo, graph_repo = _seed_multimodal_graph_state(tmp_path)
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
    chunk_ids = [candidate.chunk_id for candidate in results[:2]]

    assert chunk_ids == ["chunk-1", "chunk-table"]
    assert results[1].special_chunk_type == "table"


def test_search_backed_retrieval_factory_graph_expander_reaches_multimodal_nodes_across_sections(
    tmp_path: Path,
) -> None:
    metadata_repo, vector_repo, graph_repo = _seed_multimodal_graph_state(tmp_path)
    del vector_repo
    factory = SearchBackedRetrievalFactory(
        metadata_repo=metadata_repo,
        fts_repo=SQLiteFTSRepo(tmp_path / "fts.sqlite3"),
        graph_repo=graph_repo,
    )
    seed_candidates = [
        RetrievedCandidate(
            chunk_id="chunk-1",
            doc_id="doc-1",
            source_id="src-1",
            text="Alpha Engine coordinates ingestion.",
            citation_anchor="#chunk-1",
            score=1.0,
            rank=1,
            source_kind="internal",
        )
    ]

    results = factory.graph_expander("Show the Alpha Engine table", ["doc-1"], seed_candidates)

    assert [candidate.chunk_id for candidate in results[:1]] == ["chunk-table"]
    assert results[0].source_kind == "graph"


def test_search_backed_retrieval_factory_special_retriever_uses_multimodal_vectors(
    tmp_path: Path,
) -> None:
    metadata_repo, vector_repo, graph_repo = _seed_multimodal_graph_state(tmp_path)
    factory = SearchBackedRetrievalFactory(
        metadata_repo=metadata_repo,
        fts_repo=SQLiteFTSRepo(tmp_path / "fts.sqlite3"),
        graph_repo=graph_repo,
    )
    binding = EmbeddingProviderBinding(
        provider=StubEmbeddingProvider({"throughput table": [0.0, 1.0]}),
        space="default",
        location="local",
    )
    retriever = factory.special_retriever_from_repo(
        vector_repo,
        [binding],
        default_preference=ExecutionLocationPreference.LOCAL_ONLY,
    )
    retriever.prepare_for_policy(
        access_policy=AccessPolicy.default(),
        execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
    )

    results = retriever("throughput table", ["doc-1"])

    assert [candidate.chunk_id for candidate in results[:1]] == ["chunk-table"]
    assert results[0].special_chunk_type == "table"


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
    chunk_table = Chunk(
        chunk_id="chunk-table",
        segment_id=segment.segment_id,
        doc_id=document.doc_id,
        text="Table: Alpha Engine throughput and Beta Service dependency metrics.",
        token_count=10,
        citation_anchor="#chunk-table",
        citation_span=(98, 160),
        effective_access_policy=policy,
        extraction_quality=0.95,
        embedding_ref=None,
        order_index=2,
        chunk_role="special",
        special_chunk_type="table",
    )

    metadata_repo.save_source(source)
    metadata_repo.save_document(document, location=source.location, content_hash=source.content_hash)
    metadata_repo.save_segment(segment)
    metadata_repo.save_chunk(chunk_one)
    metadata_repo.save_chunk(chunk_two)
    metadata_repo.save_chunk(chunk_table)

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


def _seed_multi_hop_graph_state(
    tmp_path: Path,
) -> tuple[SQLiteMetadataRepo, SQLiteVectorRepo, SQLiteGraphRepo]:
    metadata_repo = SQLiteMetadataRepo(tmp_path / "metadata-two-hop.sqlite3")
    vector_repo = SQLiteVectorRepo(tmp_path / "vectors-two-hop.sqlite3")
    graph_repo = SQLiteGraphRepo(tmp_path / "graph-two-hop.sqlite3")
    policy = AccessPolicy.default()

    source = Source(
        source_id="src-1",
        source_type=SourceType.MARKDOWN,
        location="memory://alpha-beta-gamma",
        owner="tester",
        content_hash="hash-2",
        effective_access_policy=policy,
        ingest_version=1,
    )
    document = Document(
        doc_id="doc-1",
        source_id=source.source_id,
        doc_type=DocumentType.ARTICLE,
        title="Alpha Beta Gamma Notes",
        authors=["tester"],
        created_at=datetime.now(UTC),
        language="en",
        effective_access_policy=policy,
    )
    segment = Segment(
        segment_id="seg-1",
        doc_id=document.doc_id,
        parent_segment_id=None,
        toc_path=["Alpha Beta Gamma Notes"],
        heading_level=1,
        page_range=(1, 1),
        order_index=0,
        anchor="alpha-beta-gamma",
    )
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            segment_id=segment.segment_id,
            doc_id=document.doc_id,
            text="Alpha Engine powers ingestion and coordinates upstream context.",
            token_count=9,
            citation_anchor="#chunk-1",
            citation_span=(0, 58),
            effective_access_policy=policy,
            extraction_quality=0.95,
            embedding_ref=None,
            order_index=0,
        ),
        Chunk(
            chunk_id="chunk-2",
            segment_id=segment.segment_id,
            doc_id=document.doc_id,
            text="Beta Service depends on Alpha Engine for upstream context.",
            token_count=9,
            citation_anchor="#chunk-2",
            citation_span=(59, 115),
            effective_access_policy=policy,
            extraction_quality=0.95,
            embedding_ref=None,
            order_index=1,
        ),
        Chunk(
            chunk_id="chunk-3",
            segment_id=segment.segment_id,
            doc_id=document.doc_id,
            text="Beta Service uses Gamma Index to store retrieval vectors.",
            token_count=9,
            citation_anchor="#chunk-3",
            citation_span=(116, 171),
            effective_access_policy=policy,
            extraction_quality=0.95,
            embedding_ref=None,
            order_index=2,
        ),
    ]

    metadata_repo.save_source(source)
    metadata_repo.save_document(document, location=source.location, content_hash=source.content_hash)
    metadata_repo.save_segment(segment)
    for chunk in chunks:
        metadata_repo.save_chunk(chunk)

    alpha = GraphNode(node_id="entity-alpha", node_type="entity", label="Alpha Engine")
    beta = GraphNode(node_id="entity-beta", node_type="entity", label="Beta Service")
    gamma = GraphNode(node_id="entity-gamma", node_type="entity", label="Gamma Index")
    edge_alpha_beta = GraphEdge(
        edge_id="rel-beta-alpha",
        from_node_id=beta.node_id,
        to_node_id=alpha.node_id,
        relation_type="depends_on",
        confidence=0.94,
        evidence_chunk_ids=["chunk-2"],
    )
    edge_beta_gamma = GraphEdge(
        edge_id="rel-beta-gamma",
        from_node_id=beta.node_id,
        to_node_id=gamma.node_id,
        relation_type="uses",
        confidence=0.92,
        evidence_chunk_ids=["chunk-3"],
    )
    graph_repo.save_node(alpha)
    graph_repo.bind_node_evidence(alpha.node_id, ["chunk-1"])
    graph_repo.save_node(beta)
    graph_repo.bind_node_evidence(beta.node_id, ["chunk-2"])
    graph_repo.save_node(gamma)
    graph_repo.bind_node_evidence(gamma.node_id, ["chunk-3"])
    graph_repo.save_edge(edge_alpha_beta)
    graph_repo.save_edge(edge_beta_gamma)

    vector_repo.upsert(
        alpha.node_id,
        [1.0, 0.0],
        metadata={"doc_id": document.doc_id, "segment_id": "", "text": "Alpha Engine"},
        item_kind="entity",
    )
    vector_repo.upsert(
        edge_alpha_beta.edge_id,
        [0.0, 1.0],
        metadata={
            "doc_id": document.doc_id,
            "segment_id": "",
            "text": "Beta Service depends_on Alpha Engine",
        },
        item_kind="relation",
    )
    return metadata_repo, vector_repo, graph_repo


def _seed_multimodal_graph_state(
    tmp_path: Path,
) -> tuple[SQLiteMetadataRepo, SQLiteVectorRepo, SQLiteGraphRepo]:
    metadata_repo = SQLiteMetadataRepo(tmp_path / "metadata-multimodal.sqlite3")
    vector_repo = SQLiteVectorRepo(tmp_path / "vectors-multimodal.sqlite3")
    graph_repo = SQLiteGraphRepo(tmp_path / "graph-multimodal.sqlite3")
    policy = AccessPolicy.default()

    source = Source(
        source_id="src-1",
        source_type=SourceType.MARKDOWN,
        location="memory://alpha-multimodal",
        owner="tester",
        content_hash="hash-3",
        effective_access_policy=policy,
        ingest_version=1,
    )
    document = Document(
        doc_id="doc-1",
        source_id=source.source_id,
        doc_type=DocumentType.ARTICLE,
        title="Alpha Multimodal Notes",
        authors=["tester"],
        created_at=datetime.now(UTC),
        language="en",
        effective_access_policy=policy,
    )
    overview = Segment(
        segment_id="seg-overview",
        doc_id=document.doc_id,
        parent_segment_id=None,
        toc_path=["Alpha Multimodal Notes", "Overview"],
        heading_level=1,
        page_range=(1, 1),
        order_index=0,
        anchor="overview",
    )
    metrics = Segment(
        segment_id="seg-metrics",
        doc_id=document.doc_id,
        parent_segment_id=None,
        toc_path=["Alpha Multimodal Notes", "Metrics"],
        heading_level=1,
        page_range=(2, 2),
        order_index=1,
        anchor="metrics",
    )
    chunk_one = Chunk(
        chunk_id="chunk-1",
        segment_id=overview.segment_id,
        doc_id=document.doc_id,
        text="Alpha Engine coordinates ingestion.",
        token_count=4,
        citation_anchor="#chunk-1",
        citation_span=(0, 35),
        effective_access_policy=policy,
        extraction_quality=0.95,
        embedding_ref=None,
        order_index=0,
    )
    chunk_table = Chunk(
        chunk_id="chunk-table",
        segment_id=metrics.segment_id,
        doc_id=document.doc_id,
        text="Table: Alpha Engine throughput is 99 and latency is 12ms.",
        token_count=10,
        citation_anchor="#chunk-table",
        citation_span=(36, 92),
        effective_access_policy=policy,
        extraction_quality=0.95,
        embedding_ref=None,
        order_index=1,
        chunk_role="special",
        special_chunk_type="table",
    )

    metadata_repo.save_source(source)
    metadata_repo.save_document(document, location=source.location, content_hash=source.content_hash)
    metadata_repo.save_segment(overview)
    metadata_repo.save_segment(metrics)
    metadata_repo.save_chunk(chunk_one)
    metadata_repo.save_chunk(chunk_table)

    alpha = GraphNode(node_id="entity-alpha", node_type="entity", label="Alpha Engine")
    section_overview = GraphNode(node_id=overview.segment_id, node_type="section", label="Overview")
    section_metrics = GraphNode(node_id=metrics.segment_id, node_type="section", label="Metrics")
    table = GraphNode(
        node_id="node-table-1",
        node_type="table",
        label="Table: Alpha Engine throughput is 99 and latency is 12ms.",
        metadata={"special_chunk_type": "table", "chunk_id": chunk_table.chunk_id},
    )
    graph_repo.save_node(alpha)
    graph_repo.bind_node_evidence(alpha.node_id, ["chunk-1"])
    graph_repo.save_node(section_overview)
    graph_repo.bind_node_evidence(section_overview.node_id, ["chunk-1"])
    graph_repo.save_node(section_metrics)
    graph_repo.bind_node_evidence(section_metrics.node_id, ["chunk-table"])
    graph_repo.save_node(table)
    graph_repo.bind_node_evidence(table.node_id, ["chunk-table"])
    graph_repo.save_edge(
        GraphEdge(
            edge_id="edge-doc-table",
            from_node_id=section_metrics.node_id,
            to_node_id=table.node_id,
            relation_type="contains_special",
            confidence=1.0,
            evidence_chunk_ids=["chunk-table"],
        )
    )
    graph_repo.save_edge(
        GraphEdge(
            edge_id="edge-alpha-table",
            from_node_id=alpha.node_id,
            to_node_id=table.node_id,
            relation_type="tabulated_in",
            confidence=0.93,
            evidence_chunk_ids=["chunk-table"],
        )
    )

    vector_repo.upsert(
        alpha.node_id,
        [1.0, 0.0],
        metadata={"doc_id": document.doc_id, "segment_id": "", "text": "Alpha Engine"},
        item_kind="entity",
    )
    vector_repo.upsert(
        table.node_id,
        [0.0, 1.0],
        metadata={
            "doc_id": document.doc_id,
            "segment_id": metrics.segment_id,
            "chunk_id": chunk_table.chunk_id,
            "text": "Table: Alpha Engine throughput is 99 and latency is 12ms.",
            "special_chunk_type": "table",
        },
        item_kind="multimodal",
    )
    return metadata_repo, vector_repo, graph_repo
