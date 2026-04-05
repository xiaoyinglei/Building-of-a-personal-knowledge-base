from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from rag.workbench.models import WorkbenchFileEntry
from rag.workbench.service import WorkbenchService


def _isolate_model_env(monkeypatch: MonkeyPatch) -> None:
    for key, value in {
        "OPENAI_API_KEY": "",
        "OPENAI_MODEL": "",
        "OPENAI_EMBEDDING_MODEL": "",
        "OPENAI_BASE_URL": "",
        "GEMINI_API_KEY": "",
        "GEMINI_CHAT_MODEL": "",
        "GEMINI_EMBEDDING_MODEL": "",
        "GEMINI_BASE_URL": "",
        "PKP_OPENAI__API_KEY": "",
        "PKP_OPENAI__MODEL": "",
        "PKP_OPENAI__EMBEDDING_MODEL": "",
        "PKP_OPENAI__BASE_URL": "",
        "PKP_OLLAMA__BASE_URL": "",
        "PKP_OLLAMA__CHAT_MODEL": "",
        "PKP_OLLAMA__EMBEDDING_MODEL": "",
        "PKP_LOCAL_BGE__ENABLED": "false",
        "PKP_LOCAL_BGE__EMBEDDING_MODEL": "",
        "PKP_LOCAL_BGE__EMBEDDING_MODEL_PATH": "",
        "PKP_LOCAL_BGE__RERANK_MODEL": "",
        "PKP_LOCAL_BGE__RERANK_MODEL_PATH": "",
        "RAG_INDEX_EMBEDDING_MODEL": "",
        "RAG_RERANK_MODEL": "",
        "RAG_RERANK_MODEL_PATH": "",
    }.items():
        monkeypatch.setenv(key, value)


def _find_file(entries: list[WorkbenchFileEntry], rel_path: str) -> WorkbenchFileEntry | None:
    for entry in entries:
        if entry.rel_path == rel_path:
            return entry
        match = _find_file(entry.children, rel_path)
        if match is not None:
            return match
    return None


def test_workbench_state_syncs_workspace_into_index(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _isolate_model_env(monkeypatch)
    workspace_root = tmp_path / "docs"
    workspace_root.mkdir()
    note = workspace_root / "alpha.md"
    note.write_text("# Alpha\n\nAlpha Engine handles ingestion.\n", encoding="utf-8")

    service = WorkbenchService(storage_root=tmp_path / ".rag", workspace_root=workspace_root)
    state = service.get_state(sync=True)

    file_entry = _find_file(state.files, "alpha.md")
    assert file_entry is not None
    assert file_entry.sync_state == "indexed"
    assert state.index_summary.active_documents == 1
    assert state.index_summary.chunks >= 1
    profile_ids = {profile.profile_id for profile in state.model_profiles}
    assert {"local_full", "local_retrieval_cloud_chat", "cloud_full", "test_minimal"} <= profile_ids


def test_workbench_query_returns_evidence_for_selected_document(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _isolate_model_env(monkeypatch)
    workspace_root = tmp_path / "docs"
    workspace_root.mkdir()
    note = workspace_root / "alpha.md"
    note.write_text(
        "# Alpha\n\nAlpha Engine handles ingestion.\nBeta Service depends on Alpha Engine.\n",
        encoding="utf-8",
    )

    service = WorkbenchService(storage_root=tmp_path / ".rag", workspace_root=workspace_root)
    state = service.get_state(sync=True)
    file_entry = _find_file(state.files, "alpha.md")
    assert file_entry is not None

    result = service.query(
        query_text="What does Alpha Engine handle?",
        mode="mix",
        source_scope=[] if file_entry.doc_id is None else [file_entry.doc_id],
    )

    assert result.answer_text
    assert result.evidence
    assert result.query_understanding is not None


def test_workbench_delete_removes_disk_file_and_index(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _isolate_model_env(monkeypatch)
    workspace_root = tmp_path / "docs"
    workspace_root.mkdir()

    service = WorkbenchService(storage_root=tmp_path / ".rag", workspace_root=workspace_root)
    save_result = service.save_file(
        relative_path="notes/delete-me.md",
        content_text="# Delete Me\n\nThis file should disappear.\n",
        auto_ingest=True,
    )

    created = workspace_root / "notes" / "delete-me.md"
    assert created.exists()
    assert save_result.state is not None

    delete_result = service.delete_file(relative_path="notes/delete-me.md")

    assert delete_result.ok
    assert not created.exists()
    assert delete_result.state is not None
    assert _find_file(delete_result.state.files, "notes/delete-me.md") is None
