from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from rag import RAG, StorageConfig
from rag.engine import _RUNTIME_CONTRACT_KEY, _RUNTIME_CONTRACT_NAMESPACE
from rag.ingest.ingest import DeletePipelineResult, IngestPipelineResult, RebuildPipelineResult
from rag.llm._providers.fallback_embedding_repo import FallbackEmbeddingRepo
from rag.llm._providers.local_bge_provider_repo import LocalBgeProviderRepo
from rag.llm.embedding import EmbeddingProviderBinding, OllamaProviderRepo, OpenAIProviderRepo
from rag.query import QueryOptions
from rag.schema._types.storage import DocumentProcessingStatus, DocumentStatusRecord
from rag.schema._types.text import load_env_file
from rag.schema.document import SourceType
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


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_gemini_base_url(url: str) -> str:
    normalized = url.rstrip("/")
    if "generativelanguage.googleapis.com" in normalized and not normalized.endswith("/openai"):
        return f"{normalized}/openai"
    return normalized


class _ChatOnlyProviderAdapter:
    def __init__(self, provider: object) -> None:
        self._provider = provider

    @property
    def provider_name(self) -> str:
        value = getattr(self._provider, "provider_name", None)
        if isinstance(value, str) and value:
            return value
        return self._provider.__class__.__name__.lower()

    @property
    def chat_model_name(self) -> str | None:
        value = getattr(self._provider, "chat_model_name", None)
        return value if isinstance(value, str) and value else None

    def chat(self, prompt: str) -> str:
        chat = getattr(self._provider, "chat", None)
        if not callable(chat):
            raise RuntimeError("Selected model does not support chat")
        return str(chat(prompt))


@dataclass(frozen=True, slots=True)
class _ProviderProfile:
    profile_id: str
    label: str
    provider_kind: str
    location: str
    chat_model: str | None
    embedding_model: str | None
    rerank_model: str | None
    supports_chat: bool
    supports_embedding: bool
    supports_rerank: bool
    factory: Callable[[], object]

    def create_provider(self) -> object:
        return self.factory()


class WorkbenchService:
    def __init__(
        self,
        *,
        storage_root: Path,
        workspace_root: Path,
        storage_config: StorageConfig | None = None,
    ) -> None:
        load_env_file()
        self.storage_root = storage_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.storage_config = storage_config or StorageConfig(root=self.storage_root)
        self._digest_cache: dict[str, tuple[int, int, str]] = {}
        self._bridge_legacy_env()

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
        core = self._build_core(profile_id=profile_id, require_chat=False)
        try:
            normalized_mode = cast(
                Literal["bypass", "naive", "local", "global", "hybrid", "mix"],
                mode if mode in {"bypass", "naive", "local", "global", "hybrid", "mix"} else "mix",
            )
            result = core.query(
                query_text,
                options=QueryOptions(
                    mode=normalized_mode,
                    source_scope=tuple(source_scope or ()),
                ),
            )
        finally:
            core.stores.close()
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
        core = self._build_core(profile_id=profile_id, require_chat=False)
        try:
            result = core.rebuild(location=str(target))
        finally:
            core.stores.close()
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
            core = self._build_core(profile_id=active_profile_id, require_chat=False)
            try:
                for location in missing_on_disk:
                    deleted = core.delete(location=location)
                    if deleted.deleted_doc_ids:
                        messages.append(f"Removed missing file from index: {Path(location).name}")
            finally:
                core.stores.close()

        reindex_core: RAG | None = None
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
                    reindex_core = reindex_core or self._build_core(profile_id=active_profile_id, require_chat=False)
                    result = reindex_core.insert(source_type=source_type.value, location=str(path), owner="user")
                    messages.append(self._format_ingest_message(path, result))
                    continue
                if latest_source.content_hash != digest:
                    reindex_core = reindex_core or self._build_core(profile_id=active_profile_id, require_chat=False)
                    rebuilt = reindex_core.rebuild(location=str(path))
                    messages.append(self._format_rebuild_message(path, rebuilt))
        finally:
            if reindex_core is not None:
                reindex_core.stores.close()
        return messages

    def _build_core(self, *, profile_id: str | None, require_chat: bool) -> RAG:
        runtime_contract = self._load_runtime_contract_payload()
        profiles = self._discover_profiles()
        embedding_profile = self._select_embedding_profile(
            profiles=profiles,
            requested_profile_id=profile_id,
            runtime_contract=runtime_contract,
        )
        bindings: list[EmbeddingProviderBinding] = []
        if embedding_profile is not None:
            bindings.append(
                EmbeddingProviderBinding(
                    provider=embedding_profile.create_provider(),
                    space="default",
                    location=embedding_profile.location,
                )
            )
        selected_profile = self._profile_by_id(profiles, profile_id)
        if selected_profile is not None and selected_profile.supports_chat:
            chat_provider = selected_profile.create_provider()
            if embedding_profile is None or selected_profile.profile_id != embedding_profile.profile_id:
                chat_provider = _ChatOnlyProviderAdapter(chat_provider)
            bindings.insert(
                0,
                EmbeddingProviderBinding(
                    provider=chat_provider,
                    space="default",
                    location=selected_profile.location,
                ),
            )
        if not bindings:
            bindings = [EmbeddingProviderBinding(provider=FallbackEmbeddingRepo(), space="default", location="local")]
        core = RAG(storage=self.storage_config, embedding_bindings=tuple(bindings))
        if require_chat and not any(callable(getattr(binding.provider, "chat", None)) for binding in bindings):
            core.stores.close()
            raise RuntimeError("No chat-capable provider is configured for the selected profile.")
        return core

    def _discover_profiles(self) -> list[_ProviderProfile]:
        profiles: list[_ProviderProfile] = []

        api_key = _first_env("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "PKP_OPENAI__API_KEY")
        base_url = _first_env("OPENAI_BASE_URL", "GEMINI_BASE_URL", "PKP_OPENAI__BASE_URL")
        chat_model = _first_env("OPENAI_MODEL", "GEMINI_CHAT_MODEL", "PKP_OPENAI__MODEL")
        embedding_model = _first_env(
            "OPENAI_EMBEDDING_MODEL",
            "GEMINI_EMBEDDING_MODEL",
            "PKP_OPENAI__EMBEDDING_MODEL",
        )
        if api_key and chat_model and embedding_model:
            normalized_base = _normalize_gemini_base_url(base_url or "https://api.openai.com/v1")

            def create_openai_provider() -> object:
                return OpenAIProviderRepo(
                    api_key=api_key,
                    base_url=normalized_base,
                    model=chat_model,
                    embedding_model=embedding_model,
                )

            label = "Gemini (OpenAI compatible)" if "generativelanguage.googleapis.com" in normalized_base else "OpenAI"
            profiles.append(
                _ProviderProfile(
                    profile_id="openai-compatible",
                    label=f"{label} / {chat_model}",
                    provider_kind="openai-compatible",
                    location="cloud",
                    chat_model=chat_model,
                    embedding_model=embedding_model,
                    rerank_model=os.environ.get("RAG_RERANK_MODEL") or os.environ.get("PKP_LOCAL_BGE__RERANK_MODEL"),
                    supports_chat=True,
                    supports_embedding=True,
                    supports_rerank=bool(
                        os.environ.get("RAG_RERANK_MODEL")
                        or os.environ.get("RAG_RERANK_MODEL_PATH")
                        or os.environ.get("PKP_LOCAL_BGE__RERANK_MODEL")
                    ),
                    factory=create_openai_provider,
                )
            )

        ollama_base = _first_env("OLLAMA_BASE_URL", "PKP_OLLAMA__BASE_URL")
        ollama_chat = _first_env("OLLAMA_CHAT_MODEL", "PKP_OLLAMA__CHAT_MODEL")
        ollama_embedding = _first_env("OLLAMA_EMBEDDING_MODEL", "PKP_OLLAMA__EMBEDDING_MODEL")
        if ollama_base and ollama_chat:
            def create_ollama_provider() -> object:
                return OllamaProviderRepo(
                    base_url=ollama_base,
                    chat_model=ollama_chat,
                    embedding_model=ollama_embedding,
                )

            profiles.append(
                _ProviderProfile(
                    profile_id="ollama",
                    label=f"Ollama / {ollama_chat}",
                    provider_kind="ollama",
                    location="local",
                    chat_model=ollama_chat,
                    embedding_model=ollama_embedding,
                    rerank_model=os.environ.get("RAG_RERANK_MODEL") or os.environ.get("PKP_LOCAL_BGE__RERANK_MODEL"),
                    supports_chat=True,
                    supports_embedding=bool(ollama_embedding),
                    supports_rerank=bool(
                        os.environ.get("RAG_RERANK_MODEL")
                        or os.environ.get("RAG_RERANK_MODEL_PATH")
                        or os.environ.get("PKP_LOCAL_BGE__RERANK_MODEL")
                    ),
                    factory=create_ollama_provider,
                )
            )

        local_bge_enabled = (_first_env("PKP_LOCAL_BGE__ENABLED", "RAG_LOCAL_BGE_ENABLED") or "").lower()
        local_bge_model = _first_env("PKP_LOCAL_BGE__EMBEDDING_MODEL", "RAG_LOCAL_BGE_EMBEDDING_MODEL")
        local_bge_path = _first_env("PKP_LOCAL_BGE__EMBEDDING_MODEL_PATH", "RAG_LOCAL_BGE_EMBEDDING_MODEL_PATH")
        local_rerank = _first_env("RAG_RERANK_MODEL", "PKP_LOCAL_BGE__RERANK_MODEL")
        local_rerank_path = _first_env("RAG_RERANK_MODEL_PATH", "PKP_LOCAL_BGE__RERANK_MODEL_PATH")
        if local_bge_model and local_bge_enabled not in {"0", "false", "no", "off"}:
            def create_local_bge_provider() -> object:
                return LocalBgeProviderRepo(
                    embedding_model=local_bge_model,
                    embedding_model_path=local_bge_path,
                    rerank_model=local_rerank or "BAAI/bge-reranker-v2-m3",
                    rerank_model_path=local_rerank_path,
                )

            profiles.append(
                _ProviderProfile(
                    profile_id="local-bge",
                    label=f"Local BGE / {local_bge_model}",
                    provider_kind="local-bge",
                    location="local",
                    chat_model=None,
                    embedding_model=local_bge_model,
                    rerank_model=local_rerank,
                    supports_chat=False,
                    supports_embedding=True,
                    supports_rerank=bool(local_rerank or local_rerank_path),
                    factory=create_local_bge_provider,
                )
            )
        return profiles

    def _model_profiles(
        self,
        *,
        runtime_contract: dict[str, str | int | bool | None],
    ) -> list[WorkbenchModelProfile]:
        contract_embedding = runtime_contract.get("embedding_model_name")
        profiles: list[WorkbenchModelProfile] = []
        for profile in self._discover_profiles():
            compatibility_error: str | None = None
            compatible = True
            if isinstance(contract_embedding, str) and contract_embedding:
                if profile.embedding_model != contract_embedding and profile.supports_embedding:
                    compatibility_error = (
                        "index embedding model is "
                        f"{contract_embedding}, profile embedding model is {profile.embedding_model}"
                    )
                    compatible = False
                if not profile.supports_embedding and profile.supports_chat:
                    compatible = True
            profiles.append(
                WorkbenchModelProfile(
                    profile_id=profile.profile_id,
                    label=profile.label,
                    provider_kind=profile.provider_kind,
                    location=profile.location,
                    chat_model=profile.chat_model,
                    embedding_model=profile.embedding_model,
                    rerank_model=profile.rerank_model,
                    supports_chat=profile.supports_chat,
                    supports_embedding=profile.supports_embedding,
                    supports_rerank=profile.supports_rerank,
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

    def _select_embedding_profile(
        self,
        *,
        profiles: list[_ProviderProfile],
        requested_profile_id: str | None,
        runtime_contract: dict[str, str | int | bool | None],
    ) -> _ProviderProfile | None:
        contract_embedding = runtime_contract.get("embedding_model_name")
        if isinstance(contract_embedding, str) and contract_embedding:
            for profile in profiles:
                if profile.embedding_model == contract_embedding:
                    return profile
        selected = self._profile_by_id(profiles, requested_profile_id)
        if selected is not None and selected.supports_embedding:
            return selected
        for profile in profiles:
            if profile.supports_embedding:
                return profile
        return None

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
        core = self._build_core(profile_id=profile_id, require_chat=False)
        try:
            return core.insert(source_type=source_type.value, location=str(path), owner="user")
        finally:
            core.stores.close()

    def _delete_index_entry(self, path: Path, *, profile_id: str | None) -> DeletePipelineResult | None:
        core = self._build_core(profile_id=profile_id, require_chat=False)
        try:
            return core.delete(location=str(path))
        except ValueError:
            return None
        finally:
            core.stores.close()

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

    def _bridge_legacy_env(self) -> None:
        if "RAG_RERANK_MODEL" not in os.environ and "PKP_LOCAL_BGE__RERANK_MODEL" in os.environ:
            os.environ["RAG_RERANK_MODEL"] = os.environ["PKP_LOCAL_BGE__RERANK_MODEL"]
        if "RAG_RERANK_MODEL_PATH" not in os.environ and "PKP_LOCAL_BGE__RERANK_MODEL_PATH" in os.environ:
            os.environ["RAG_RERANK_MODEL_PATH"] = os.environ["PKP_LOCAL_BGE__RERANK_MODEL_PATH"]
        if "RAG_INDEX_EMBEDDING_MODEL" not in os.environ and "PKP_OPENAI__EMBEDDING_MODEL" in os.environ:
            os.environ["RAG_INDEX_EMBEDDING_MODEL"] = os.environ["PKP_OPENAI__EMBEDDING_MODEL"]

    @contextmanager
    def _stores(self) -> Any:
        stores = self.storage_config.build()
        try:
            yield stores
        finally:
            stores.close()


__all__ = ["WorkbenchService"]
