import json
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

import rag.cli as cli
from rag.cli import app

runner = CliRunner()


def test_cli_ingest_query_delete_rebuild_round_trip(tmp_path: Path) -> None:
    storage_root = tmp_path / ".rag"

    ingest = runner.invoke(
        app,
        [
            "ingest",
            "--storage-root",
            str(storage_root),
            "--source-type",
            "plain_text",
            "--location",
            "memory://note-1",
            "--content",
            "Alpha Engine handles ingestion. Beta Service depends on Alpha Engine.",
        ],
    )

    query = runner.invoke(
        app,
        [
            "query",
            "--storage-root",
            str(storage_root),
            "--query",
            "What does Alpha Engine handle?",
            "--json",
        ],
    )
    payload = json.loads(query.stdout)

    delete = runner.invoke(
        app,
        [
            "delete",
            "--storage-root",
            str(storage_root),
            "--location",
            "memory://note-1",
        ],
    )

    rebuild = runner.invoke(
        app,
        [
            "rebuild",
            "--storage-root",
            str(storage_root),
            "--location",
            "memory://note-1",
        ],
    )

    assert ingest.exit_code == 0
    assert query.exit_code == 0
    assert delete.exit_code == 0
    assert rebuild.exit_code == 0
    assert payload["answer"]["answer_text"]
    assert payload["context"]["evidence"]


def test_cli_rejects_missing_source_payload_for_ingest(tmp_path: Path) -> None:
    storage_root = tmp_path / ".rag"

    result = runner.invoke(
        app,
        [
            "ingest",
            "--storage-root",
            str(storage_root),
            "--source-type",
            "plain_text",
            "--location",
            "memory://note-1",
        ],
    )

    assert result.exit_code != 0
    assert "content" in result.stdout.lower() or "content" in result.stderr.lower()


def test_cli_main_delegates_to_typer_app(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeApp:
        def __call__(self) -> None:
            calls.append("called")

    monkeypatch.setattr(cli, "app", FakeApp())

    cli.main()

    assert calls == ["called"]
