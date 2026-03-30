from __future__ import annotations

from rag.engine import RAG
from rag.schema._types.storage import DocumentProcessingStatus
from rag.storage import StorageConfig


def test_ragcore_delete_removes_retrieval_indexes_and_marks_document_deleted() -> None:
    core = RAG(storage=StorageConfig.in_memory())
    try:
        inserted = core.insert(
            source_type="plain_text",
            location="memory://alpha-engine",
            owner="user",
            content_text=(
                "Alpha Engine processes ingestion requests. Beta Service depends on Alpha Engine for upstream context."
            ),
        )

        before = core.query("What does Alpha Engine do?")
        deleted = core.delete(location="memory://alpha-engine")
        after = core.query("What does Alpha Engine do?")
        status = core.stores.status.get(inserted.document_id)

        assert before.retrieval.evidence.internal
        assert deleted.deleted_doc_ids == [inserted.document_id]
        assert deleted.deleted_source_ids == [inserted.source.source_id]
        assert deleted.deleted_chunk_ids
        assert status is not None
        assert status.status is DocumentProcessingStatus.DELETED
        assert core.stores.documents.list_documents(active_only=True) == []
        assert not after.retrieval.evidence.internal
    finally:
        core.stores.close()


def test_ragcore_rebuild_restores_deleted_document_from_stored_source() -> None:
    core = RAG(storage=StorageConfig.in_memory())
    try:
        inserted = core.insert(
            source_type="plain_text",
            location="memory://rebuildable-note",
            owner="user",
            content_text=("Alpha Engine processes ingestion requests. Gamma Index stores chunk vectors for retrieval."),
        )
        core.delete(location="memory://rebuildable-note")

        rebuilt = core.rebuild(location="memory://rebuildable-note")
        after = core.query("What does Gamma Index store?")
        status = core.stores.status.get(inserted.document_id)

        assert rebuilt.rebuilt_doc_ids == [inserted.document_id]
        assert rebuilt.results[0].document_id == inserted.document_id
        assert rebuilt.results[0].chunk_count > 0
        assert status is not None
        assert status.status is DocumentProcessingStatus.READY
        assert after.retrieval.evidence.internal
    finally:
        core.stores.close()


def test_ragcore_rebuild_marks_status_failed_when_source_payload_is_missing() -> None:
    core = RAG(storage=StorageConfig.in_memory())
    try:
        inserted = core.insert(
            source_type="plain_text",
            location="memory://broken-rebuild",
            owner="user",
            content_text="Alpha Engine processes ingestion requests.",
        )
        source = inserted.source
        object_key = source.metadata.get("object_key")
        assert object_key is not None
        core.delete(location="memory://broken-rebuild")
        core._object_store.path_for_key(object_key).unlink()

        try:
            core.rebuild(location="memory://broken-rebuild")
        except ValueError as exc:
            assert "No rebuildable source payload available" in str(exc)
        else:
            raise AssertionError("rebuild should fail when the stored source payload is missing")

        status = core.stores.status.get(inserted.document_id)
        assert status is not None
        assert status.status is DocumentProcessingStatus.FAILED
    finally:
        core.stores.close()
