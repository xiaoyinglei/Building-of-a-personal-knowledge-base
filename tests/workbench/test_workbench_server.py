from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import urlopen

from pytest import MonkeyPatch

from rag.workbench.server import create_workbench_server


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


def test_workbench_server_exposes_state_endpoint(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _isolate_model_env(monkeypatch)
    workspace_root = tmp_path / "docs"
    workspace_root.mkdir()
    (workspace_root / "server-note.md").write_text("# Note\n\nServer smoke.\n", encoding="utf-8")

    server = create_workbench_server(
        storage_root=tmp_path / ".rag",
        workspace_root=workspace_root,
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_port)
        with urlopen(f"http://127.0.0.1:{port}/api/state?sync=1") as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["workspace_root"] == str(workspace_root.resolve())
    assert payload["index_summary"]["active_documents"] == 1
    assert payload["files"]
