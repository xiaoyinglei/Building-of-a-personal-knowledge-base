from datetime import UTC, datetime, timedelta
from pathlib import Path

from pkp.storage._repo.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.schema._types import (
    AccessPolicy,
    ArtifactStatus,
    ArtifactType,
    Chunk,
    Document,
    DocumentType,
    KnowledgeArtifact,
    Segment,
    Source,
    SourceType,
)
from pkp.schema._types.storage import CacheEntry, DocumentPipelineStage, DocumentProcessingStatus, DocumentStatusRecord


def test_sqlite_metadata_repo_persists_source_document_chunks_and_artifacts(tmp_path: Path) -> None:
    repo = SQLiteMetadataRepo(tmp_path / "metadata.sqlite3")
    policy = AccessPolicy.default()
    source = Source(
        source_id="src-1",
        source_type=SourceType.MARKDOWN,
        location="data/samples/agent-rag-overview.md",
        owner="user",
        content_hash="hash-1",
        effective_access_policy=policy,
        ingest_version=1,
    )
    document = Document(
        doc_id="doc-1",
        source_id=source.source_id,
        doc_type=DocumentType.ARTICLE,
        title="Agentic RAG Overview",
        authors=["user"],
        created_at=datetime.now(UTC),
        language="en",
        effective_access_policy=policy,
    )
    segment = Segment(
        segment_id="seg-1",
        doc_id=document.doc_id,
        parent_segment_id=None,
        toc_path=["Agentic RAG Overview", "Reliability First"],
        heading_level=2,
        page_range=(1, 1),
        order_index=0,
        anchor="agentic-rag-overview/reliability-first",
    )
    chunk = Chunk(
        chunk_id="chunk-1",
        segment_id=segment.segment_id,
        doc_id=document.doc_id,
        text="Reliable retrieval is more important than fluent synthesis.",
        token_count=8,
        citation_anchor="Reliability First",
        citation_span=(0, 55),
        effective_access_policy=policy,
        extraction_quality=0.95,
        embedding_ref="emb-1",
    )
    artifact = KnowledgeArtifact(
        artifact_id="artifact-1",
        artifact_type=ArtifactType.TOPIC_PAGE,
        title="Agentic RAG",
        supported_chunk_ids=[chunk.chunk_id],
        confidence=0.87,
        status=ArtifactStatus.SUGGESTED,
        last_reviewed_at=datetime.now(UTC),
        body_markdown="# Agentic RAG",
        source_scope=[document.doc_id],
    )

    repo.save_source(source)
    repo.save_document_bundle(document, [segment], [chunk])
    repo.save_artifact(artifact)

    loaded_source = repo.get_source(source.source_id)
    loaded_document = repo.get_document(document.doc_id)
    loaded_segments = repo.list_segments(document.doc_id)
    loaded_chunks = repo.list_chunks(document.doc_id)
    loaded_artifacts = repo.list_artifacts()

    assert loaded_source is not None
    assert loaded_document is not None
    assert [item.segment_id for item in loaded_segments] == ["seg-1"]
    assert [item.chunk_id for item in loaded_chunks] == ["chunk-1"]
    assert [item.artifact_id for item in loaded_artifacts] == ["artifact-1"]


def test_sqlite_metadata_repo_supports_content_hash_lookup_for_dedup(tmp_path: Path) -> None:
    repo = SQLiteMetadataRepo(tmp_path / "metadata.sqlite3")
    source = Source(
        source_id="src-1",
        source_type=SourceType.PLAIN_TEXT,
        location="data/samples/conflict-a.txt",
        owner="user",
        content_hash="hash-dedup",
        effective_access_policy=AccessPolicy.default(),
        ingest_version=1,
    )

    repo.save_source(source)

    loaded = repo.find_source_by_content_hash("hash-dedup")
    assert loaded is not None
    assert loaded.source_id == "src-1"


def test_sqlite_metadata_repo_persists_document_status_and_cache_entries(tmp_path: Path) -> None:
    repo = SQLiteMetadataRepo(tmp_path / "metadata.sqlite3")
    status = DocumentStatusRecord(
        doc_id="doc-1",
        source_id="src-1",
        location="data/samples/agent-rag-overview.md",
        content_hash="hash-1",
        status=DocumentProcessingStatus.READY,
        stage=DocumentPipelineStage.PERSIST,
        attempts=1,
    )
    cache_entry = CacheEntry(
        namespace="query",
        cache_key="doc-1::summary",
        payload={"answer": "grounded"},
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
    )

    saved_status = repo.save_document_status(status)
    saved_cache = repo.save_cache_entry(cache_entry)

    assert repo.get_document_status(status.doc_id) == saved_status
    assert repo.list_document_statuses(status=DocumentProcessingStatus.READY.value) == [saved_status]
    assert repo.get_cache_entry(cache_entry.cache_key, namespace=cache_entry.namespace) == saved_cache
    assert repo.list_cache_entries(namespace="query") == [saved_cache]
