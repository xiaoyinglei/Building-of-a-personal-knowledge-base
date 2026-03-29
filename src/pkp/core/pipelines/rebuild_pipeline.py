from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pkp.core.pipelines.delete_pipeline import DeletePipeline, DeleteRequest, ResolvedLifecycleTarget
from pkp.core.pipelines.ingest_pipeline import IngestPipeline, IngestPipelineResult, IngestRequest
from pkp.repo.storage.file_object_store import FileObjectStore
from pkp.schema.document import Document, DocumentPipelineStage, DocumentProcessingStatus, SourceType


@dataclass(frozen=True, slots=True)
class RebuildRequest:
    doc_id: str | None = None
    source_id: str | None = None
    location: str | None = None


@dataclass(frozen=True, slots=True)
class RebuildPipelineResult:
    results: list[IngestPipelineResult] = field(default_factory=list)

    @property
    def rebuilt_doc_ids(self) -> list[str]:
        return [result.document_id for result in self.results]


@dataclass(slots=True)
class RebuildPipeline:
    ingest_pipeline: IngestPipeline
    delete_pipeline: DeletePipeline
    object_store: FileObjectStore | None

    def run(self, request: RebuildRequest) -> RebuildPipelineResult:
        delete_request = DeleteRequest(
            doc_id=request.doc_id,
            source_id=request.source_id,
            location=request.location,
        )
        targets = self.delete_pipeline.resolve_targets(delete_request, include_inactive=True)
        if not targets:
            raise ValueError("No document matched the rebuild request")

        results: list[IngestPipelineResult] = []
        for target in targets:
            results.append(self._rebuild_target(target))
        return RebuildPipelineResult(results=results)

    def _rebuild_target(self, target: ResolvedLifecycleTarget) -> IngestPipelineResult:
        self.ingest_pipeline._save_status(
            doc_id=target.document.doc_id,
            source_id=target.source.source_id,
            location=target.source.location,
            content_hash=target.source.content_hash,
            status=DocumentProcessingStatus.REBUILDING,
            stage=DocumentPipelineStage.REBUILD,
        )
        self.documents.deactivate_documents_for_location(target.source.location)
        self.documents.set_active(target.document.doc_id, active=False)
        stage = DocumentPipelineStage.REBUILD
        try:
            request, raw_bytes = self._build_ingest_request(target)
            parsed = self.ingest_pipeline._resolve_parsed_document(
                request=request,
                source_type=SourceType(target.source.source_type),
                raw_bytes=raw_bytes,
            )
            rebuilt_document = self._rebuilt_document(target.document, parsed)
            self.documents.save_document(
                rebuilt_document,
                location=target.source.location,
                content_hash=target.source.content_hash,
                active=False,
            )
            self.delete_pipeline.purge_document_artifacts(target, delete_chunk_records=True)

            stage = DocumentPipelineStage.CHUNK
            self.ingest_pipeline._save_status(
                doc_id=rebuilt_document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.REBUILDING,
                stage=stage,
            )
            segments, stored_chunks, indexed_chunks, processing = self.ingest_pipeline._prepare_chunks(
                location=target.source.location,
                source=target.source,
                document=rebuilt_document,
                parsed=parsed,
                access_policy=target.source.effective_access_policy,
            )

            stage = DocumentPipelineStage.PERSIST
            self.ingest_pipeline._save_status(
                doc_id=rebuilt_document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.REBUILDING,
                stage=stage,
            )
            for segment in segments:
                self.documents.save_segment(segment)
            self.chunks.save_many(stored_chunks)
            self.ingest_pipeline._index_chunks(
                source=target.source,
                document=rebuilt_document,
                segments=segments,
                chunks=indexed_chunks,
            )

            stage = DocumentPipelineStage.EXTRACT
            self.ingest_pipeline._save_status(
                doc_id=rebuilt_document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.REBUILDING,
                stage=stage,
            )
            entity_count, relation_count = self.ingest_pipeline._extract_and_persist_graph(
                source=target.source,
                document=rebuilt_document,
                chunks=indexed_chunks,
            )
            self.documents.save_document(
                rebuilt_document,
                location=target.source.location,
                content_hash=target.source.content_hash,
                active=True,
            )
            final_status = self.ingest_pipeline._save_status(
                doc_id=rebuilt_document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.READY,
                stage=DocumentPipelineStage.INDEX,
            )
        except Exception as exc:
            self.ingest_pipeline._save_status(
                doc_id=target.document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.FAILED,
                stage=stage,
                error_message=str(exc),
            )
            raise

        return IngestPipelineResult(
            source=target.source,
            document=rebuilt_document,
            segments=segments,
            chunks=indexed_chunks,
            is_duplicate=False,
            content_hash=target.source.content_hash,
            visible_text=parsed.visible_text,
            visual_semantics=parsed.visual_semantics,
            processing=processing,
            entity_count=entity_count,
            relation_count=relation_count,
            status=final_status.status.value,
        )

    @property
    def documents(self):
        return self.ingest_pipeline.documents

    @property
    def chunks(self):
        return self.ingest_pipeline.chunks

    def _build_ingest_request(self, target: ResolvedLifecycleTarget) -> tuple[IngestRequest, bytes]:
        source = target.source
        source_type = SourceType(source.source_type)
        object_key = source.metadata.get("object_key")
        file_path: Path | None = None
        raw_bytes: bytes | None = None
        if object_key and self.object_store is not None and self.object_store.exists(object_key):
            raw_bytes = self.object_store.read_bytes(object_key)
            if source_type in {SourceType.PDF, SourceType.DOCX, SourceType.IMAGE}:
                file_path = self.object_store.path_for_key(object_key)
        elif Path(source.location).exists():
            file_path = Path(source.location)
            raw_bytes = file_path.read_bytes()
        if raw_bytes is None:
            raise ValueError(f"No rebuildable source payload available for {source.location}")

        content_text = None
        if source_type in {
            SourceType.MARKDOWN,
            SourceType.PLAIN_TEXT,
            SourceType.PASTED_TEXT,
            SourceType.WEB,
            SourceType.BROWSER_CLIP,
        }:
            content_text = raw_bytes.decode("utf-8")

        return (
            IngestRequest(
                location=source.location,
                source_type=source_type,
                owner=source.owner,
                title=target.document.title,
                content_text=content_text,
                raw_bytes=raw_bytes,
                file_path=file_path,
            ),
            raw_bytes,
        )

    @staticmethod
    def _rebuilt_document(document: Document, parsed) -> Document:
        return document.model_copy(
            update={
                "doc_type": parsed.doc_type,
                "title": parsed.title or document.title,
                "authors": list(parsed.authors or document.authors),
                "language": parsed.language or document.language,
                "metadata": {
                    **parsed.metadata,
                    "location": document.metadata.get("location", ""),
                    "content_hash": document.metadata.get("content_hash", ""),
                },
            }
        )
