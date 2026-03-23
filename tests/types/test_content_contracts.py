from datetime import UTC, datetime

from pkp.types.access import AccessPolicy
from pkp.types.artifact import ArtifactStatus, ArtifactType, KnowledgeArtifact
from pkp.types.content import (
    Chunk,
    Document,
    DocumentType,
    GraphEdge,
    GraphNode,
    Segment,
    Source,
    SourceType,
)


def test_content_contracts_expose_required_fields() -> None:
    policy = AccessPolicy.default()
    source = Source(
        source_id="src-1",
        source_type=SourceType.MARKDOWN,
        location="notes/topic.md",
        owner="user",
        content_hash="hash",
        effective_access_policy=policy,
        ingest_version=1,
    )
    document = Document(
        doc_id="doc-1",
        source_id=source.source_id,
        doc_type=DocumentType.ARTICLE,
        title="Topic",
        authors=["user"],
        created_at=datetime.now(UTC),
        language="en",
        effective_access_policy=policy,
    )
    segment = Segment(
        segment_id="seg-1",
        doc_id=document.doc_id,
        parent_segment_id=None,
        toc_path=["Topic", "Section"],
        heading_level=2,
        page_range=(1, 1),
        order_index=0,
    )
    chunk = Chunk(
        chunk_id="chunk-1",
        segment_id=segment.segment_id,
        doc_id=document.doc_id,
        text="evidence",
        token_count=1,
        citation_anchor="Topic > Section",
        citation_span=(0, 8),
        effective_access_policy=policy,
        extraction_quality=0.9,
        embedding_ref="emb-1",
    )
    node = GraphNode(node_id="node-1", node_type="topic", label="Topic")
    edge = GraphEdge(
        edge_id="edge-1",
        from_node_id=node.node_id,
        to_node_id="node-2",
        relation_type="supports",
        confidence=0.8,
        evidence_chunk_ids=[chunk.chunk_id],
    )
    artifact = KnowledgeArtifact(
        artifact_id="artifact-1",
        artifact_type=ArtifactType.TOPIC_PAGE,
        title="Topic page",
        supported_chunk_ids=[chunk.chunk_id],
        confidence=0.91,
        status=ArtifactStatus.SUGGESTED,
        last_reviewed_at=datetime.now(UTC),
        body_markdown="# Topic",
    )

    assert source.source_id == "src-1"
    assert document.doc_id == "doc-1"
    assert segment.toc_path == ["Topic", "Section"]
    assert chunk.citation_anchor == "Topic > Section"
    assert edge.evidence_chunk_ids == ["chunk-1"]
    assert artifact.status is ArtifactStatus.SUGGESTED
