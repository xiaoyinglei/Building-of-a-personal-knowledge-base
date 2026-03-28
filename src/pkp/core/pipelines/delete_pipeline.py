from __future__ import annotations

from dataclasses import dataclass, field

from pkp.core.pipelines.ingest_pipeline import IngestPipeline
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.stores.chunk_store import ChunkStore
from pkp.stores.document_store import DocumentStore
from pkp.stores.graph_store import GraphStore
from pkp.stores.status_store import StatusStore
from pkp.stores.vector_store import VectorStore
from pkp.types.content import Chunk, Document, Source
from pkp.types.storage import DocumentPipelineStage, DocumentProcessingStatus


@dataclass(frozen=True, slots=True)
class DeleteRequest:
    doc_id: str | None = None
    source_id: str | None = None
    location: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedLifecycleTarget:
    source: Source
    document: Document
    chunks: list[Chunk]


@dataclass(frozen=True, slots=True)
class PurgeSummary:
    deleted_chunk_ids: list[str] = field(default_factory=list)
    deleted_node_ids: list[str] = field(default_factory=list)
    deleted_edge_ids: list[str] = field(default_factory=list)
    deleted_fts_count: int = 0
    deleted_vector_count: int = 0
    deleted_chunk_record_count: int = 0
    deleted_segment_record_count: int = 0


@dataclass(frozen=True, slots=True)
class DeletePipelineResult:
    deleted_doc_ids: list[str] = field(default_factory=list)
    deleted_source_ids: list[str] = field(default_factory=list)
    deleted_chunk_ids: list[str] = field(default_factory=list)
    deleted_node_ids: list[str] = field(default_factory=list)
    deleted_edge_ids: list[str] = field(default_factory=list)
    deleted_fts_count: int = 0
    deleted_vector_count: int = 0


@dataclass(slots=True)
class DeletePipeline:
    documents: DocumentStore
    chunks: ChunkStore
    vectors: VectorStore
    graph: GraphStore
    status: StatusStore
    fts_repo: SQLiteFTSRepo
    ingest_pipeline: IngestPipeline

    def run(self, request: DeleteRequest) -> DeletePipelineResult:
        targets = self.resolve_targets(request, include_inactive=False)
        if not targets:
            raise ValueError("No active document matched the delete request")

        deleted_doc_ids: list[str] = []
        deleted_source_ids: list[str] = []
        deleted_chunk_ids: list[str] = []
        deleted_node_ids: list[str] = []
        deleted_edge_ids: list[str] = []
        deleted_fts_count = 0
        deleted_vector_count = 0

        for target in targets:
            self.ingest_pipeline._save_status(
                doc_id=target.document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.DELETING,
                stage=DocumentPipelineStage.DELETE,
            )
            self.documents.set_active(target.document.doc_id, active=False)
            purge = self.purge_document_artifacts(target, delete_chunk_records=False)
            self.ingest_pipeline._save_status(
                doc_id=target.document.doc_id,
                source_id=target.source.source_id,
                location=target.source.location,
                content_hash=target.source.content_hash,
                status=DocumentProcessingStatus.DELETED,
                stage=DocumentPipelineStage.DELETE,
            )
            deleted_doc_ids.append(target.document.doc_id)
            deleted_source_ids.append(target.source.source_id)
            deleted_chunk_ids.extend(purge.deleted_chunk_ids)
            deleted_node_ids.extend(purge.deleted_node_ids)
            deleted_edge_ids.extend(purge.deleted_edge_ids)
            deleted_fts_count += purge.deleted_fts_count
            deleted_vector_count += purge.deleted_vector_count

        return DeletePipelineResult(
            deleted_doc_ids=deleted_doc_ids,
            deleted_source_ids=list(dict.fromkeys(deleted_source_ids)),
            deleted_chunk_ids=list(dict.fromkeys(deleted_chunk_ids)),
            deleted_node_ids=list(dict.fromkeys(deleted_node_ids)),
            deleted_edge_ids=list(dict.fromkeys(deleted_edge_ids)),
            deleted_fts_count=deleted_fts_count,
            deleted_vector_count=deleted_vector_count,
        )

    def resolve_targets(
        self,
        request: DeleteRequest,
        *,
        include_inactive: bool,
    ) -> list[ResolvedLifecycleTarget]:
        self._validate_request(request)
        if request.doc_id is not None:
            document = self.documents.get_document(request.doc_id)
            if document is None:
                return []
            if not include_inactive and not self.documents.is_active(document.doc_id):
                return []
            source = self.documents.get_source(document.source_id)
            if source is None:
                return []
            return [self._target(source, document)]

        if request.source_id is not None:
            source = self.documents.get_source(request.source_id)
            if source is None:
                return []
            documents = self.documents.list_documents(
                source.source_id,
                active_only=not include_inactive,
            )
            if not documents and include_inactive:
                documents = self.documents.list_documents(source.source_id, active_only=False)
            return [self._target(source, document) for document in documents]

        assert request.location is not None
        source = self.documents.get_latest_source_for_location(request.location)
        if source is None:
            return []
        documents = self.documents.list_documents(source.source_id, active_only=True)
        if not documents and include_inactive:
            documents = self.documents.list_documents(source.source_id, active_only=False)
        return [self._target(source, document) for document in documents]

    def purge_document_artifacts(
        self,
        target: ResolvedLifecycleTarget,
        *,
        delete_chunk_records: bool,
    ) -> PurgeSummary:
        chunk_ids = [chunk.chunk_id for chunk in target.chunks]
        deleted_fts_count = self.fts_repo.delete_by_chunk_ids(chunk_ids)
        deleted_vector_count = self.vectors.delete_for_documents([target.document.doc_id])
        deleted_node_ids, deleted_edge_ids = self.graph.delete_by_chunk_ids(chunk_ids)
        deleted_chunk_record_count = 0
        deleted_segment_record_count = 0
        if delete_chunk_records:
            deleted_chunk_record_count = self.chunks.delete_for_document(target.document.doc_id)
            deleted_segment_record_count = self.documents.delete_segments_for_document(target.document.doc_id)
        return PurgeSummary(
            deleted_chunk_ids=chunk_ids,
            deleted_node_ids=deleted_node_ids,
            deleted_edge_ids=deleted_edge_ids,
            deleted_fts_count=deleted_fts_count,
            deleted_vector_count=deleted_vector_count,
            deleted_chunk_record_count=deleted_chunk_record_count,
            deleted_segment_record_count=deleted_segment_record_count,
        )

    def _target(self, source: Source, document: Document) -> ResolvedLifecycleTarget:
        return ResolvedLifecycleTarget(
            source=source,
            document=document,
            chunks=self.chunks.list_by_document(document.doc_id),
        )

    @staticmethod
    def _validate_request(request: DeleteRequest) -> None:
        selectors = [request.doc_id, request.source_id, request.location]
        if sum(1 for value in selectors if value is not None) != 1:
            raise ValueError("DeleteRequest requires exactly one of doc_id, source_id, or location")
