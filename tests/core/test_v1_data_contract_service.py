from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from rag.schema.core import Document, DocumentType, SectionRecord, Source, SourceType
from rag.storage.v1_data_contract_service import V1DataContractService


class _FakeMetadataRepo:
    def __init__(self) -> None:
        self.existing_document: Document | None = None
        self.saved_sources: list[Source] = []
        self.saved_documents: list[Document] = []
        self.saved_sections: list[SectionRecord] = []
        self.incremented_doc_ids: list[int] = []
        self.index_state_calls: list[tuple[int, bool | None, bool | None, str | None]] = []
        self.deactivated_doc_ids: list[int] = []

    def find_document_by_hash(self, file_hash: str):
        return self.existing_document

    def increment_document_reference_count(self, doc_id: int, *, amount: int = 1) -> Document:
        self.incremented_doc_ids.append(doc_id)
        assert self.existing_document is not None
        self.existing_document = self.existing_document.model_copy(
            update={"reference_count": self.existing_document.reference_count + amount}
        )
        return self.existing_document

    def save_source(self, source: Source) -> Source:
        saved = source.model_copy(update={"source_id": 10})
        self.saved_sources.append(saved)
        return saved

    def save_document(self, document: Document) -> Document:
        saved = document.model_copy(update={"doc_id": 20, "version_group_id": 20})
        self.saved_documents.append(saved)
        return saved

    def save_section(self, section: SectionRecord) -> SectionRecord:
        saved = section.model_copy(update={"section_id": 30})
        self.saved_sections.append(saved)
        return saved

    def set_document_index_state(
        self,
        doc_id: int,
        *,
        is_indexed: bool | None = None,
        index_ready: bool | None = None,
        embedding_model_id: str | None = None,
        indexed_at=None,
        last_index_error: str | None = None,
    ) -> Document:
        self.index_state_calls.append((doc_id, is_indexed, index_ready, embedding_model_id))
        return self.saved_documents[-1] if self.saved_documents else self.existing_document  # type: ignore[return-value]

    def deactivate_document(self, doc_id: int) -> Document:
        self.deactivated_doc_ids.append(doc_id)
        assert self.existing_document is not None
        self.existing_document = self.existing_document.model_copy(update={"is_active": False})
        return self.existing_document


class _FakeMilvusRepo:
    def __init__(self) -> None:
        self.upsert_calls: list[tuple[object, list[float], str]] = []
        self.delete_calls: list[str] = []
        self.fail_delete = False
        self.search_calls: list[tuple[list[float], dict[str, object]]] = []

    def upsert_record(self, record, vector, *, embedding_space: str = "default") -> None:
        self.upsert_calls.append((record, list(vector), embedding_space))

    def delete(self, *, expr: str, item_kind=None, embedding_space=None) -> int:
        self.delete_calls.append(expr)
        if self.fail_delete:
            raise RuntimeError("delete failed")
        return 1

    def search(self, query_vector, **kwargs):
        self.search_calls.append((list(query_vector), dict(kwargs)))
        return []


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def _build_document(*, doc_id: int = 99, file_hash: str = "hash-1") -> Document:
    now = datetime(2026, 4, 19, tzinfo=UTC)
    return Document(
        doc_id=doc_id,
        source_id=10,
        title="Call 13812345678",
        doc_type=DocumentType.REPORT,
        language="zh",
        authors=["alice"],
        file_hash=file_hash,
        version_group_id=doc_id,
        doc_status="published",
        is_active=True,
        embedding_model_id="bge-m3",
        created_at=now,
        updated_at=now,
    )


def test_v1_data_contract_service_deduplicates_file_hash_and_returns_early() -> None:
    metadata_repo = _FakeMetadataRepo()
    metadata_repo.existing_document = _build_document()
    service = V1DataContractService(metadata_repo, _FakeMilvusRepo(), embedder=_FakeEmbedder())

    result = service.register_document(
        source=Source(source_type=SourceType.MARKDOWN, location="docs/a.md", content_hash=""),
        document=_build_document(doc_id=0, file_hash=""),
        file_bytes=b"same-file",
    )

    assert result.is_duplicate is True
    assert result.source is None
    assert result.document.doc_id == 99
    assert metadata_repo.incremented_doc_ids == [99]
    assert metadata_repo.saved_sources == []
    assert metadata_repo.saved_documents == []


def test_v1_data_contract_service_skips_embedding_when_not_urgent() -> None:
    metadata_repo = _FakeMetadataRepo()
    milvus_repo = _FakeMilvusRepo()
    document = _build_document(doc_id=20, file_hash="hash-2")
    metadata_repo.saved_documents.append(document)
    service = V1DataContractService(metadata_repo, milvus_repo, embedder=_FakeEmbedder())

    saved_section = service.save_section(
        document,
        SectionRecord(
            doc_id=document.doc_id,
            source_id=document.source_id,
            order_index=0,
            section_kind="body",
            content_hash="section-hash",
        ),
        source_type=SourceType.MARKDOWN,
        summary_text="联系我 13812345678",
        is_urgent=False,
    )

    assert saved_section.section_id == 30
    assert metadata_repo.saved_sections[0].metadata_json["summary_text"] == "联系我 [手机号脱敏]"
    assert milvus_repo.upsert_calls == []
    assert metadata_repo.index_state_calls[-1] == (20, False, False, "bge-m3")


def test_v1_data_contract_service_deactivate_document_logs_milvus_failures(caplog: pytest.LogCaptureFixture) -> None:
    metadata_repo = _FakeMetadataRepo()
    metadata_repo.existing_document = _build_document()
    milvus_repo = _FakeMilvusRepo()
    milvus_repo.fail_delete = True
    service = V1DataContractService(metadata_repo, milvus_repo, logger=logging.getLogger("v1-test"))

    with caplog.at_level(logging.ERROR):
        document = service.deactivate_document(99)

    assert document.is_active is False
    assert metadata_repo.deactivated_doc_ids == [99]
    assert milvus_repo.delete_calls == ["doc_id in [99]"]
    assert "failed to delete milvus vectors" in caplog.text
