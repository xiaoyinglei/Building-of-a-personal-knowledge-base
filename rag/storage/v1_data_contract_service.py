from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from rag.schema.core import (
    AssetRecord,
    AssetSummaryRecord,
    DocSummaryRecord,
    Document,
    SectionRecord,
    SectionSummaryRecord,
    Source,
    SourceType,
)
from rag.storage.repositories.postgres_metadata_repo import PostgresMetadataRepo
from rag.storage.search_backends.milvus_vector_repo import MilvusVectorRepo


_PHONE_PATTERN = re.compile(r"\b1[3-9]\d{9}\b")


@dataclass(frozen=True, slots=True)
class DocumentRegistrationResult:
    source: Source | None
    document: Document
    is_duplicate: bool


class V1DataContractService:
    def __init__(
        self,
        metadata_repo: PostgresMetadataRepo,
        milvus_repo: MilvusVectorRepo,
        *,
        embedder: object | None = None,
        embedding_space: str = "default",
        logger: logging.Logger | None = None,
    ) -> None:
        self.metadata_repo = metadata_repo
        self.milvus_repo = milvus_repo
        self.embedder = embedder
        self.embedding_space = embedding_space
        self.logger = logger or logging.getLogger(__name__)

    def compute_file_hash(self, file_bytes: bytes, *, algorithm: str = "sha256") -> str:
        hasher = hashlib.new(algorithm)
        hasher.update(file_bytes)
        return hasher.hexdigest()

    def mask_pii(self, text: str | None) -> str:
        if not text:
            return ""
        return _PHONE_PATTERN.sub("[手机号脱敏]", text)

    def register_document(
        self,
        *,
        source: Source,
        document: Document,
        file_bytes: bytes,
        hash_algorithm: str = "sha256",
        increment_reference: bool = True,
    ) -> DocumentRegistrationResult:
        file_hash = self.compute_file_hash(file_bytes, algorithm=hash_algorithm)
        existing = self.metadata_repo.find_document_by_hash(file_hash)
        if existing is not None:
            if increment_reference:
                existing = self.metadata_repo.increment_document_reference_count(existing.doc_id)
            return DocumentRegistrationResult(source=None, document=existing, is_duplicate=True)

        saved_source = self.metadata_repo.save_source(
            source.model_copy(
                update={
                    "content_hash": file_hash,
                    "original_file_name": self.mask_pii(source.original_file_name),
                }
            )
        )
        saved_document = self.metadata_repo.save_document(
            document.model_copy(
                update={
                    "source_id": saved_source.source_id,
                    "file_hash": file_hash,
                    "title": self._masked_optional_text(document.title),
                }
            )
        )
        return DocumentRegistrationResult(source=saved_source, document=saved_document, is_duplicate=False)

    def save_doc_summary(
        self,
        document: Document,
        *,
        source_type: SourceType | None,
        summary_text: str,
        is_urgent: bool = True,
    ) -> DocSummaryRecord:
        masked_summary = self.mask_pii(summary_text)
        updated_document = self.metadata_repo.save_document(
            document.model_copy(
                update={
                    "title": self._masked_optional_text(document.title),
                    "metadata_json": {
                        **document.metadata_json,
                        "summary_text": masked_summary,
                    },
                }
            )
        )
        record = DocSummaryRecord(
            doc_id=updated_document.doc_id,
            source_id=updated_document.source_id,
            version_group_id=updated_document.version_group_id,
            version_no=updated_document.version_no,
            doc_status=updated_document.doc_status,
            effective_date=updated_document.effective_date,
            updated_at=updated_document.updated_at,
            is_active=updated_document.is_active,
            index_ready=is_urgent,
            tenant_id=updated_document.tenant_id,
            department_id=updated_document.department_id,
            auth_tag=updated_document.auth_tag,
            source_type=source_type,
            embedding_model_id=updated_document.embedding_model_id,
            title=self._masked_optional_text(updated_document.title),
            summary_text=masked_summary,
            metadata_json=updated_document.metadata_json,
        )
        self._maybe_index_record(updated_document, record, text=masked_summary, is_urgent=is_urgent)
        return record

    def save_section(
        self,
        document: Document,
        section: SectionRecord,
        *,
        source_type: SourceType | None,
        summary_text: str,
        is_urgent: bool = True,
    ) -> SectionRecord:
        masked_summary = self.mask_pii(summary_text)
        saved_section = self.metadata_repo.save_section(
            section.model_copy(
                update={
                    "metadata_json": {
                        **section.metadata_json,
                        "summary_text": masked_summary,
                    }
                }
            )
        )
        record = SectionSummaryRecord(
            section_id=saved_section.section_id,
            doc_id=saved_section.doc_id,
            source_id=saved_section.source_id,
            version_group_id=document.version_group_id,
            version_no=document.version_no,
            doc_status=document.doc_status,
            effective_date=document.effective_date,
            updated_at=saved_section.updated_at,
            is_active=document.is_active,
            index_ready=is_urgent,
            tenant_id=document.tenant_id,
            department_id=document.department_id,
            auth_tag=document.auth_tag,
            source_type=source_type,
            embedding_model_id=document.embedding_model_id,
            page_start=saved_section.page_start,
            page_end=saved_section.page_end,
            section_kind=saved_section.section_kind,
            toc_path=saved_section.toc_path,
            summary_text=masked_summary,
            metadata_json=saved_section.metadata_json,
        )
        self._maybe_index_record(document, record, text=masked_summary, is_urgent=is_urgent)
        return saved_section

    def save_asset(
        self,
        document: Document,
        asset: AssetRecord,
        *,
        summary_text: str,
        is_urgent: bool = True,
    ) -> AssetRecord:
        masked_summary = self.mask_pii(summary_text)
        saved_asset = self.metadata_repo.save_asset(
            asset.model_copy(
                update={
                    "caption": self._masked_optional_text(asset.caption),
                    "metadata_json": {
                        **asset.metadata_json,
                        "summary_text": masked_summary,
                    },
                }
            )
        )
        record = AssetSummaryRecord(
            asset_id=saved_asset.asset_id,
            doc_id=saved_asset.doc_id,
            source_id=saved_asset.source_id,
            section_id=saved_asset.section_id,
            version_group_id=document.version_group_id,
            version_no=document.version_no,
            doc_status=document.doc_status,
            effective_date=document.effective_date,
            updated_at=saved_asset.updated_at,
            is_active=document.is_active,
            index_ready=is_urgent,
            tenant_id=document.tenant_id,
            department_id=document.department_id,
            auth_tag=document.auth_tag,
            embedding_model_id=document.embedding_model_id,
            asset_type=saved_asset.asset_type,
            page_no=saved_asset.page_no,
            caption=self._masked_optional_text(saved_asset.caption),
            summary_text=masked_summary,
            metadata_json=saved_asset.metadata_json,
        )
        self._maybe_index_record(document, record, text=masked_summary, is_urgent=is_urgent)
        return saved_asset

    def search(
        self,
        query_vector: list[float],
        *,
        item_kind: str = "section_summary",
        limit: int = 10,
        doc_ids: list[str] | None = None,
        expr: str | None = None,
    ):
        return self.milvus_repo.search(
            query_vector,
            limit=limit,
            doc_ids=doc_ids,
            expr=expr,
            embedding_space=self.embedding_space,
            item_kind=item_kind,
        )

    def deactivate_document(self, doc_id: int) -> Document:
        document = self.metadata_repo.deactivate_document(doc_id)
        try:
            self.milvus_repo.delete(expr=f"doc_id in [{doc_id}]")
        except Exception as exc:  # pragma: no cover - defensive logging path
            self.logger.error("failed to delete milvus vectors for doc_id=%s: %s", doc_id, exc)
        return document

    def _maybe_index_record(
        self,
        document: Document,
        record: DocSummaryRecord | SectionSummaryRecord | AssetSummaryRecord,
        *,
        text: str,
        is_urgent: bool,
    ) -> None:
        if not is_urgent:
            if not document.is_indexed or not document.index_ready:
                self.metadata_repo.set_document_index_state(
                    document.doc_id,
                    is_indexed=False,
                    index_ready=False,
                    embedding_model_id=document.embedding_model_id,
                    last_index_error=None,
                )
            return
        vector = self._embed_text(text)
        self.milvus_repo.upsert_record(record, vector, embedding_space=self.embedding_space)
        self.metadata_repo.set_document_index_state(
            document.doc_id,
            is_indexed=True,
            index_ready=True,
            embedding_model_id=record.embedding_model_id,
            last_index_error=None,
        )

    def _embed_text(self, text: str) -> list[float]:
        embed = getattr(self.embedder, "embed", None)
        if not callable(embed):
            raise RuntimeError("embedding capability is required when is_urgent=True")
        vectors = embed([text])
        if not vectors:
            raise RuntimeError("embedding provider returned no vectors")
        return [float(value) for value in vectors[0]]

    def _masked_optional_text(self, text: str | None) -> str | None:
        masked = self.mask_pii(text)
        return masked or None


__all__ = ["DocumentRegistrationResult", "V1DataContractService"]
