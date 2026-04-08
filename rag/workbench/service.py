from __future__ import annotations

import base64
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from rag import RAGRuntime, StorageConfig
from rag.assembly import AssemblyRequest, CapabilityAssemblyService, CapabilityRequirements
from rag.ingest.pipeline import DeletePipelineResult, IngestPipelineResult, RebuildPipelineResult
from rag.retrieval import QueryOptions
from rag.runtime import _RUNTIME_CONTRACT_KEY, _RUNTIME_CONTRACT_NAMESPACE
from rag.schema.core import SourceType
from rag.schema.runtime import DocumentProcessingStatus, DocumentStatusRecord
from rag.storage import StorageBundle
from rag.workbench.models import (
    WorkbenchEvidenceItem,
    WorkbenchFileEntry,
    WorkbenchIndexSummary,
    WorkbenchModelProfile,
    WorkbenchOperationResult,
    WorkbenchQueryResult,
    WorkbenchState,
)

_SUPPORTED_SUFFIXES: dict[str, SourceType] = {
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".txt": SourceType.PLAIN_TEXT,
    ".text": SourceType.PLAIN_TEXT,
    ".pdf": SourceType.PDF,
    ".docx": SourceType.DOCX,
    ".pptx": SourceType.PPTX,
    ".xlsx": SourceType.XLSX,
    ".png": SourceType.IMAGE,
    ".jpg": SourceType.IMAGE,
    ".jpeg": SourceType.IMAGE,
    ".webp": SourceType.IMAGE,
    ".bmp": SourceType.IMAGE,
    ".gif": SourceType.IMAGE,
}


class WorkbenchService:
    def __init__(
        self,
        *,
        storage_root: Path,
        workspace_root: Path,
        storage_config: StorageConfig | None = None,
    ) -> None:
        self.storage_root = storage_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.storage_config = storage_config or StorageConfig(root=self.storage_root)
        self._digest_cache: dict[str, tuple[int, int, str]] = {}
        self._assembly_service = CapabilityAssemblyService()

    def get_state(self, *, active_profile_id: str | None = None, sync: bool = True) -> WorkbenchState:
        sync_messages: list[str] = []
        if sync:
            sync_messages = self.sync_workspace(active_profile_id=active_profile_id)
        with self._stores() as stores:
            runtime_contract = self._runtime_contract(stores)
            profiles = self._model_profiles(runtime_contract=runtime_contract)
            selected_profile_id = self._resolve_active_profile_id(
                profiles=profiles,
                requested=active_profile_id,
                runtime_contract=runtime_contract,
            )
            files = self._build_workspace_tree(stores)
            return WorkbenchState(
                storage_root=str(self.storage_root),
                workspace_root=str(self.workspace_root),
                backend_summary=self._backend_summary(),
                active_profile_id=selected_profile_id,
                model_profiles=profiles,
                index_summary=self._index_summary(stores, runtime_contract=runtime_contract),
                files_version=self._files_version(files),
                files=files,
                sync_messages=sync_messages,
            )

    def query(
        self,
        *,
        query_text: str,
        mode: str,
        profile_id: str | None = None,
        source_scope: list[str] | None = None,
    ) -> WorkbenchQueryResult:
        runtime = self._build_runtime(profile_id=profile_id, require_chat=False)
        try:
            normalized_mode = cast(
                Literal["bypass", "naive", "local", "global", "hybrid", "mix"],
                mode if mode in {"bypass", "naive", "local", "global", "hybrid", "mix"} else "mix",
            )
            result = runtime.query(
                query_text,
                options=QueryOptions(
                    mode=normalized_mode,
                    source_scope=tuple(source_scope or ()),
                ),
            )
        finally:
            runtime.close()
        diagnostics = result.retrieval.diagnostics.model_dump(mode="json")
        understanding = diagnostics.get("query_understanding")
        return WorkbenchQueryResult(
            query=result.query,
            mode=result.mode,
            profile_id=profile_id,
            answer_text=result.answer.answer_text,
            insufficient_evidence=result.answer.insufficient_evidence_flag,
            generation_provider=result.generation_provider,
            generation_model=result.generation_model,
            rerank_provider=result.retrieval.diagnostics.rerank_provider,
            mode_executor=result.retrieval.diagnostics.mode_executor,
            token_budget=result.context.token_budget,
            token_count=result.context.token_count,
            truncated_count=result.context.truncated_count,
            diagnostics=diagnostics,
            routing_decision=result.retrieval.decision.model_dump(mode="json"),
            query_understanding=cast("dict[str, object] | None", understanding),
            evidence=[
                WorkbenchEvidenceItem.model_validate(item.model_dump(mode="json"))
                for item in result.context.evidence
            ],
        )

    def save_file(
        self,
        *,
        relative_path: str,
        profile_id: str | None = None,
        content_text: str | None = None,
        content_base64: str | None = None,
        auto_ingest: bool = True,
    ) -> WorkbenchOperationResult:
        target = self._workspace_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._resolve_file_payload(content_text=content_text, content_base64=content_base64)
        target.write_bytes(payload)
        message = f"Saved {target.name}"
        if auto_ingest and self._source_type_for(target) is not None:
            result = self._ingest_path(target, profile_id=profile_id)
            message = self._format_ingest_message(target, result)
        return WorkbenchOperationResult(
            ok=True,
            message=message,
            state=self.get_state(active_profile_id=profile_id, sync=True),
        )

    def ingest_file(self, *, relative_path: str, profile_id: str | None = None) -> WorkbenchOperationResult:
        target = self._workspace_path(relative_path)
        result = self._ingest_path(target, profile_id=profile_id)
        return WorkbenchOperationResult(
            ok=True,
            message=self._format_ingest_message(target, result),
            state=self.get_state(active_profile_id=profile_id, sync=False),
        )

    def rebuild_file(self, *, relative_path: str, profile_id: str | None = None) -> WorkbenchOperationResult:
        target = self._workspace_path(relative_path)
        runtime = self._build_runtime(profile_id=profile_id, require_chat=False)
        try:
            result = runtime.rebuild(location=str(target))
        finally:
            runtime.close()
        return WorkbenchOperationResult(
            ok=True,
            message=self._format_rebuild_message(target, result),
            state=self.get_state(active_profile_id=profile_id, sync=False),
        )

    def delete_file(self, *, relative_path: str, profile_id: str | None = None) -> WorkbenchOperationResult:
        target = self._workspace_path(relative_path)
        delete_result = self._delete_index_entry(target, profile_id=profile_id)
        if target.exists():
            target.unlink()
        return WorkbenchOperationResult(
            ok=True,
            message=self._format_delete_message(target, delete_result),
            state=self.get_state(active_profile_id=profile_id, sync=False),
        )

    def sync_workspace(self, *, active_profile_id: str | None = None) -> list[str]:
        messages: list[str] = []
        with self._stores() as stores:
            latest_status = self._latest_status_by_location(stores)
            indexed_locations = self._indexed_workspace_locations(stores)
            filesystem_entries = list(self._iter_workspace_files())
            filesystem_paths = {str(path) for path in filesystem_entries}

        missing_on_disk = sorted(indexed_locations - filesystem_paths)
        if missing_on_disk:
            runtime = self._build_runtime(profile_id=active_profile_id, require_chat=False)
            try:
                for location in missing_on_disk:
                    deleted = runtime.delete(location=location)
                    if deleted.deleted_doc_ids:
                        messages.append(f"Removed missing file from index: {Path(location).name}")
            finally:
                runtime.close()

        reindex_runtime: RAGRuntime | None = None
        try:
            for path in filesystem_entries:
                source_type = self._source_type_for(path)
                if source_type is None:
                    continue
                digest = self._file_digest(path)
                status = latest_status.get(str(path))
                if (
                    status is not None
                    and status.status is DocumentProcessingStatus.FAILED
                    and status.content_hash == digest
                ):
                    continue
                with self._stores() as stores:
                    latest_source = stores.documents.get_latest_source_for_location(str(path))
                if latest_source is None:
                    reindex_runtime = reindex_runtime or self._build_runtime(
                        profile_id=active_profile_id,
                        require_chat=False,
                    )
                    result = reindex_runtime.insert(source_type=source_type.value, location=str(path), owner="user")
                    messages.append(self._format_ingest_message(path, result))
                    continue
                if latest_source.content_hash != digest:
                    reindex_runtime = reindex_runtime or self._build_runtime(
                        profile_id=active_profile_id,
                        require_chat=False,
                    )
                    rebuilt = reindex_runtime.rebuild(location=str(path))
                    messages.append(self._format_rebuild_message(path, rebuilt))
        finally:
            if reindex_runtime is not None:
                reindex_runtime.close()
        return messages

    def _build_runtime(self, *, profile_id: str | None, require_chat: bool) -> RAGRuntime:
        requirements = CapabilityRequirements(
            require_chat=require_chat,
            default_context_tokens=QueryOptions().max_context_tokens,
        )
        if profile_id:
            return RAGRuntime.from_profile(
                storage=self.storage_config,
                profile_id=profile_id,
                requirements=requirements,
                assembly_service=self._assembly_service,
            )
        return RAGRuntime.from_request(
            storage=self.storage_config,
            request=AssemblyRequest(requirements=requirements),
            assembly_service=self._assembly_service,
        )

    def _model_profiles(
        self,
        *,
        runtime_contract: dict[str, str | int | bool | None],
    ) -> list[WorkbenchModelProfile]:
        profiles: list[WorkbenchModelProfile] = []
        for profile in self._assembly_service.catalog_from_environment().assembly_profiles:
            preview_bundle = self._assembly_service.evaluate_request(
                self._assembly_service.request_for_profile(
                    profile.profile_id,
                    requirements=CapabilityRequirements(
                        require_chat=True,
                        default_context_tokens=QueryOptions().max_context_tokens,
                    ),
                )
            )
            governance = self._assembly_service.govern_runtime_contract(
                bundle=preview_bundle,
                stored_payload=runtime_contract,
            )
            compatible = preview_bundle.status != "invalid" and governance.status != "invalid"
            compatibility_error = None
            if governance.issues:
                compatibility_error = governance.issues[0].message
            elif preview_bundle.diagnostics.errors:
                compatibility_error = preview_bundle.diagnostics.errors[0].message
            elif preview_bundle.diagnostics.warnings:
                compatibility_error = preview_bundle.diagnostics.warnings[0].message
            profiles.append(
                WorkbenchModelProfile(
                    profile_id=profile.profile_id,
                    label=profile.label,
                    provider_kind="assembly-profile",
                    location=profile.location,
                    description=profile.description,
                    chat_model=preview_bundle.chat_bindings[0].model_name if preview_bundle.chat_bindings else None,
                    embedding_model=(
                        preview_bundle.embedding_bindings[0].model_name
                        if preview_bundle.embedding_bindings
                        else None
                    ),
                    rerank_model=(
                        preview_bundle.rerank_bindings[0].model_name
                        if preview_bundle.rerank_bindings
                        else None
                    ),
                    supports_chat=bool(preview_bundle.chat_bindings),
                    supports_embedding=bool(preview_bundle.embedding_bindings),
                    supports_rerank=bool(preview_bundle.rerank_bindings),
                    compatible_with_index=compatible,
                    compatibility_error=compatibility_error,
                )
            )
        return profiles

    def _build_workspace_tree(self, stores: StorageBundle) -> list[WorkbenchFileEntry]:
        latest_status = self._latest_status_by_location(stores)
        directories: set[Path] = {self.workspace_root}
        nodes: dict[str, WorkbenchFileEntry] = {}
        for path in self._iter_workspace_files(include_unsupported=True):
            for parent in path.parents:
                if parent == self.workspace_root.parent:
                    break
                if str(parent).startswith(str(self.workspace_root)):
                    directories.add(parent)
                if parent == self.workspace_root:
                    break
            nodes[str(path)] = self._file_entry(path, stores=stores, latest_status=latest_status)
        for directory in sorted(directories):
            if directory == self.workspace_root:
                continue
            rel_path = self._relative(directory)
            nodes[str(directory)] = WorkbenchFileEntry(
                name=directory.name,
                rel_path=rel_path,
                abs_path=str(directory),
                node_type="directory",
                sync_state="directory",
            )

        children_map: dict[str, list[WorkbenchFileEntry]] = {}
        for entry in nodes.values():
            parent_key = str(Path(entry.abs_path).parent)
            children_map.setdefault(parent_key, []).append(entry)

        def build_node(path: Path) -> WorkbenchFileEntry:
            entry = nodes[str(path)]
            if entry.node_type != "directory":
                return entry
            children = sorted(
                children_map.get(str(path), []),
                key=lambda item: (item.node_type != "directory", item.name),
            )
            return entry.model_copy(update={"children": [build_node(Path(child.abs_path)) for child in children]})

        roots = sorted(
            children_map.get(str(self.workspace_root), []),
            key=lambda item: (item.node_type != "directory", item.name),
        )
        return [build_node(Path(entry.abs_path)) for entry in roots]

    def _file_entry(
        self,
        path: Path,
        *,
        stores: StorageBundle,
        latest_status: dict[str, DocumentStatusRecord],
    ) -> WorkbenchFileEntry:
        stat = path.stat()
        source_type = self._source_type_for(path)
        status_record = latest_status.get(str(path))
        latest_source = stores.documents.get_latest_source_for_location(str(path))
        latest_document = stores.documents.get_latest_document_for_location(str(path))
        chunk_count = 0
        if latest_document is not None and stores.documents.is_active(latest_document.doc_id):
            chunk_count = len(stores.chunks.list_by_document(latest_document.doc_id))
        digest = self._file_digest(path) if source_type is not None else None
        sync_state = "unsupported"
        if source_type is not None:
            if latest_source is None:
                sync_state = (
                    "failed"
                    if status_record and status_record.status is DocumentProcessingStatus.FAILED
                    else "unindexed"
                )
            elif latest_source.content_hash != digest:
                sync_state = "out_of_sync"
            elif status_record and status_record.status is DocumentProcessingStatus.FAILED:
                sync_state = "failed"
            else:
                sync_state = "indexed"
        return WorkbenchFileEntry(
            name=path.name,
            rel_path=self._relative(path),
            abs_path=str(path),
            node_type="file",
            source_type=None if source_type is None else source_type.value,
            sync_state=sync_state,
            status=None if status_record is None else status_record.status.value,
            stage=None if status_record is None else str(status_record.stage),
            error_message=None if status_record is None else status_record.error_message,
            doc_id=None if latest_document is None else latest_document.doc_id,
            source_id=None if latest_source is None else latest_source.source_id,
            chunk_count=chunk_count,
            ingest_version=None if latest_source is None else latest_source.ingest_version,
            size_bytes=stat.st_size,
            modified_at=self._iso_mtime(stat.st_mtime),
        )

    def _index_summary(
        self,
        stores: StorageBundle,
        *,
        runtime_contract: dict[str, str | int | bool | None],
    ) -> WorkbenchIndexSummary:
        sources = stores.documents.list_sources()
        active_documents = stores.documents.list_documents(active_only=True)
        all_documents = stores.documents.list_documents(active_only=False)
        chunk_count = sum(len(stores.chunks.list_by_document(document.doc_id)) for document in active_documents)
        statuses: dict[str, int] = {}
        for record in stores.status.list():
            statuses[record.status.value] = statuses.get(record.status.value, 0) + 1
        return WorkbenchIndexSummary(
            sources=len(sources),
            documents=len(all_documents),
            active_documents=len(active_documents),
            chunks=chunk_count,
            vectors=stores.vector_repo.count_vectors(item_kind="chunk"),
            graph_nodes=len(stores.graph.list_nodes()),
            graph_edges=len(stores.graph.list_edges()),
            statuses=statuses,
            runtime_contract=runtime_contract,
        )

    def _runtime_contract(self, stores: StorageBundle) -> dict[str, str | int | bool | None]:
        entry = stores.cache.get(_RUNTIME_CONTRACT_KEY, namespace=_RUNTIME_CONTRACT_NAMESPACE)
        if entry is None or not isinstance(entry.payload, dict):
            return {}
        payload = cast("dict[str, str | int | bool | None]", entry.payload)
        return payload

    def _load_runtime_contract_payload(self) -> dict[str, str | int | bool | None]:
        with self._stores() as stores:
            return self._runtime_contract(stores)

    @staticmethod
    def _latest_status_by_location(stores: StorageBundle) -> dict[str, DocumentStatusRecord]:
        latest: dict[str, DocumentStatusRecord] = {}
        for record in stores.status.list():
            existing = latest.get(record.location)
            if existing is None or record.updated_at > existing.updated_at:
                latest[record.location] = record
        return latest

    def _indexed_workspace_locations(self, stores: StorageBundle) -> set[str]:
        indexed: set[str] = set()
        for source in stores.documents.list_sources():
            location = source.location
            if location.startswith(str(self.workspace_root)):
                indexed.add(location)
        return indexed

    @staticmethod
    def _resolve_active_profile_id(
        *,
        profiles: list[WorkbenchModelProfile],
        requested: str | None,
        runtime_contract: dict[str, str | int | bool | None],
    ) -> str | None:
        if requested is not None and any(profile.profile_id == requested for profile in profiles):
            return requested
        contract_embedding = runtime_contract.get("embedding_model_name")
        if isinstance(contract_embedding, str) and contract_embedding:
            for profile in profiles:
                if profile.embedding_model == contract_embedding:
                    return profile.profile_id
        return None if not profiles else profiles[0].profile_id

    @staticmethod
    def _profile_by_id[T](
        profiles: list[T],
        profile_id: str | None,
    ) -> T | None:
        if profile_id is None:
            return None
        for profile in profiles:
            candidate = getattr(profile, "profile_id", None)
            if candidate == profile_id:
                return profile
        return None

    def _ingest_path(self, path: Path, *, profile_id: str | None) -> IngestPipelineResult:
        source_type = self._source_type_for(path)
        if source_type is None:
            raise RuntimeError(f"Unsupported source type: {path.suffix}")
        runtime = self._build_runtime(profile_id=profile_id, require_chat=False)
        try:
            return runtime.insert(source_type=source_type.value, location=str(path), owner="user")
        finally:
            runtime.close()

    def _delete_index_entry(self, path: Path, *, profile_id: str | None) -> DeletePipelineResult | None:
        runtime = self._build_runtime(profile_id=profile_id, require_chat=False)
        try:
            return runtime.delete(location=str(path))
        except ValueError:
            return None
        finally:
            runtime.close()

    @staticmethod
    def _format_ingest_message(path: Path, result: IngestPipelineResult) -> str:
        duplicate = " (duplicate repair)" if result.is_duplicate else ""
        return f"Ingested {path.name}: {result.chunk_count} chunks{duplicate}"

    @staticmethod
    def _format_rebuild_message(path: Path, result: RebuildPipelineResult) -> str:
        return f"Rebuilt {path.name}: {len(result.results)} document(s)"

    @staticmethod
    def _format_delete_message(path: Path, result: DeletePipelineResult | None) -> str:
        if result is None:
            return f"Deleted file {path.name}"
        return f"Deleted {path.name}: {len(result.deleted_doc_ids)} document(s) removed from index"

    @staticmethod
    def _resolve_file_payload(*, content_text: str | None, content_base64: str | None) -> bytes:
        if content_text is not None:
            return content_text.encode("utf-8")
        if content_base64 is None:
            raise RuntimeError("Missing file payload")
        return base64.b64decode(content_base64.encode("ascii"))

    def _workspace_path(self, relative_path: str) -> Path:
        raw = (self.workspace_root / relative_path).resolve()
        if self.workspace_root not in raw.parents and raw != self.workspace_root:
            raise RuntimeError("Path escapes the workspace root")
        return raw

    def _file_digest(self, path: Path) -> str:
        stat = path.stat()
        cache_key = str(path)
        cached = self._digest_cache.get(cache_key)
        if cached is not None and cached[0] == int(stat.st_mtime_ns) and cached[1] == int(stat.st_size):
            return cached[2]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self._digest_cache[cache_key] = (int(stat.st_mtime_ns), int(stat.st_size), digest)
        return digest

    def _iter_workspace_files(self, *, include_unsupported: bool = False) -> list[Path]:
        paths = [
            path
            for path in self.workspace_root.rglob("*")
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(self.workspace_root).parts)
        ]
        if include_unsupported:
            return sorted(paths)
        return sorted(path for path in paths if self._source_type_for(path) is not None)

    @staticmethod
    def _source_type_for(path: Path) -> SourceType | None:
        return _SUPPORTED_SUFFIXES.get(path.suffix.lower())

    def _files_version(self, files: list[WorkbenchFileEntry]) -> str:
        payload = "".join(self._flatten_file_signature(entry) for entry in files)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _flatten_file_signature(self, entry: WorkbenchFileEntry) -> str:
        parts = [
            entry.rel_path,
            entry.sync_state,
            str(entry.status),
            str(entry.stage),
            str(entry.size_bytes),
            str(entry.modified_at),
            str(entry.chunk_count),
        ]
        return "|".join(parts) + "".join(self._flatten_file_signature(child) for child in entry.children)

    @staticmethod
    def _iso_mtime(timestamp: float) -> str:
        from datetime import UTC, datetime

        return datetime.fromtimestamp(timestamp, UTC).isoformat()

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.workspace_root))

    @staticmethod
    def _backend_summary() -> list[str]:
        return [
            "metadata: sqlite",
            "vectors: sqlite",
            "graph: sqlite",
            "fts: sqlite",
            "objects: local",
        ]

    @contextmanager
    def _stores(self) -> Any:
        stores = self.storage_config.build()
        try:
            yield stores
        finally:
            stores.close()


__all__ = ["WorkbenchService"]
