from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from rag import RAG, StorageConfig
from rag.query import QueryMode, QueryOptions
from rag.schema.document import SourceType

app = typer.Typer(add_completion=False, no_args_is_help=True)
DEFAULT_STORAGE_ROOT = Path(".rag")
STORAGE_ROOT_OPTION = typer.Option("--storage-root")
SOURCE_TYPE_OPTION = typer.Option("--source-type")
LOCATION_OPTION = typer.Option("--location")
CONTENT_OPTION = typer.Option("--content")
TITLE_OPTION = typer.Option("--title")
OWNER_OPTION = typer.Option("--owner")
QUERY_OPTION = typer.Option("--query")
MODE_OPTION = typer.Option("--mode")
JSON_OPTION = typer.Option("--json")
DOC_ID_OPTION = typer.Option("--doc-id")
SOURCE_ID_OPTION = typer.Option("--source-id")


def _core(storage_root: Path) -> RAG:
    return RAG(storage=StorageConfig(root=storage_root))


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, default=_json_default))


def _requires_content(source_type: SourceType) -> bool:
    return source_type in {SourceType.PLAIN_TEXT, SourceType.PASTED_TEXT, SourceType.BROWSER_CLIP}


@app.command()
def ingest(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    source_type: Annotated[SourceType | None, SOURCE_TYPE_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
    content: Annotated[str | None, CONTENT_OPTION] = None,
    title: Annotated[str | None, TITLE_OPTION] = None,
    owner: Annotated[str, OWNER_OPTION] = "user",
) -> None:
    if source_type is None:
        raise typer.BadParameter("--source-type is required")
    if location is None or not location.strip():
        raise typer.BadParameter("--location is required")
    if _requires_content(source_type) and content is None:
        raise typer.BadParameter("--content is required for text-based ingest")

    result = _core(storage_root).insert(
        source_type=source_type.value,
        location=location,
        owner=owner,
        title=title,
        content_text=content,
    )
    _echo_json(result)


@app.command()
def query(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    query: Annotated[str | None, QUERY_OPTION] = None,
    mode: Annotated[QueryMode, MODE_OPTION] = QueryMode.MIX,
    json_output: Annotated[bool, JSON_OPTION] = False,
) -> None:
    if query is None or not query.strip():
        raise typer.BadParameter("--query is required")
    result = _core(storage_root).query(query, options=QueryOptions(mode=mode.value))
    if json_output:
        _echo_json(result)
        return
    typer.echo(result.answer.answer_text)


@app.command()
def delete(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    doc_id: Annotated[str | None, DOC_ID_OPTION] = None,
    source_id: Annotated[str | None, SOURCE_ID_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
) -> None:
    result = _core(storage_root).delete(doc_id=doc_id, source_id=source_id, location=location)
    _echo_json(result)


@app.command()
def rebuild(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    doc_id: Annotated[str | None, DOC_ID_OPTION] = None,
    source_id: Annotated[str | None, SOURCE_ID_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
) -> None:
    result = _core(storage_root).rebuild(doc_id=doc_id, source_id=source_id, location=location)
    _echo_json(
        {
            "rebuilt_doc_ids": result.rebuilt_doc_ids,
            "results": result.results,
        }
    )


__all__ = ["app", "main"]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
