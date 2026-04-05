from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageConfig
from rag.query import QueryMode, QueryOptions
from rag.schema.document import SourceType
from rag.workbench import find_free_port, run_workbench_server

app = typer.Typer(add_completion=False, no_args_is_help=True)
DEFAULT_STORAGE_ROOT = Path(".rag")
DEFAULT_WORKSPACE_ROOT = Path("data/test_corpus/tech_docs")
STORAGE_ROOT_OPTION = typer.Option("--storage-root")
WORKSPACE_ROOT_OPTION = typer.Option("--workspace-root")
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
HOST_OPTION = typer.Option("--host")
PORT_OPTION = typer.Option("--port")
OPEN_BROWSER_OPTION = typer.Option("--open-browser/--no-open-browser")
PROFILE_OPTION = typer.Option("--profile", help="Recommended assembly profile to use.")


def _runtime(storage_root: Path, *, profile_id: str | None = None, require_chat: bool = False) -> RAGRuntime:
    request = (
        CapabilityRequirements(
            require_chat=require_chat,
            default_context_tokens=QueryOptions().max_context_tokens,
        )
    )
    if profile_id:
        return RAGRuntime.from_profile(
            storage=StorageConfig(root=storage_root),
            profile_id=profile_id,
            requirements=request,
        )
    return RAGRuntime.from_request(
        storage=StorageConfig(root=storage_root),
        request=AssemblyRequest(requirements=request),
    )


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
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
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

    with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
        result = runtime.insert(
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
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    query: Annotated[str | None, QUERY_OPTION] = None,
    mode: Annotated[QueryMode, MODE_OPTION] = QueryMode.MIX,
    json_output: Annotated[bool, JSON_OPTION] = False,
) -> None:
    if query is None or not query.strip():
        raise typer.BadParameter("--query is required")
    with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
        result = runtime.query(query, options=QueryOptions(mode=mode.value))
    if json_output:
        _echo_json(result)
        return
    typer.echo(result.answer.answer_text)


@app.command()
def delete(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    doc_id: Annotated[str | None, DOC_ID_OPTION] = None,
    source_id: Annotated[str | None, SOURCE_ID_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
) -> None:
    with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
        result = runtime.delete(doc_id=doc_id, source_id=source_id, location=location)
    _echo_json(result)


@app.command()
def rebuild(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    doc_id: Annotated[str | None, DOC_ID_OPTION] = None,
    source_id: Annotated[str | None, SOURCE_ID_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
) -> None:
    with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
        result = runtime.rebuild(doc_id=doc_id, source_id=source_id, location=location)
    _echo_json(
        {
            "rebuilt_doc_ids": result.rebuilt_doc_ids,
            "results": result.results,
        }
    )


@app.command("profiles")
def list_profiles(
    json_output: Annotated[bool, JSON_OPTION] = False,
) -> None:
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(
            requirements=CapabilityRequirements(
                require_chat=False,
                default_context_tokens=QueryOptions().max_context_tokens,
            )
        ),
    )
    try:
        catalog = runtime.catalog
        payload = [
            {
                "profile_id": profile.profile_id,
                "label": profile.label,
                "description": profile.description,
                "location": profile.location,
                "recommended_requirements": {
                    "require_embedding": profile.recommended_requirements.require_embedding,
                    "require_chat": profile.recommended_requirements.require_chat,
                    "require_rerank": profile.recommended_requirements.require_rerank,
                    "allow_degraded": profile.recommended_requirements.allow_degraded,
                },
            }
            for profile in catalog.assembly_profiles
        ]
        if json_output:
            _echo_json(payload)
            return
        for profile in payload:
            typer.echo(f"{profile['profile_id']}: {profile['label']} [{profile['location']}]")
            typer.echo(f"  {profile['description']}")
    finally:
        runtime.close()


@app.command()
def workbench(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    workspace_root: Annotated[Path, WORKSPACE_ROOT_OPTION] = DEFAULT_WORKSPACE_ROOT,
    host: Annotated[str, HOST_OPTION] = "127.0.0.1",
    port: Annotated[int, PORT_OPTION] = 0,
    open_browser: Annotated[bool, OPEN_BROWSER_OPTION] = True,
) -> None:
    resolved_port = port if port > 0 else find_free_port(host)
    run_workbench_server(
        storage_root=storage_root,
        workspace_root=workspace_root,
        host=host,
        port=resolved_port,
        open_browser=open_browser,
    )


__all__ = ["app", "main"]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
