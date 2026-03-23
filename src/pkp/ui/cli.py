from __future__ import annotations

import json
from typing import cast

import typer

from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.dependencies import build_container, set_container_factory

app = typer.Typer(add_completion=False)


@app.command()
def health() -> None:
    typer.echo("ok")


@app.command()
def ingest(
    source_type: str = typer.Option(..., "--source-type"),
    location: str = typer.Option(..., "--location"),
) -> None:
    container = build_container()
    result = container.ingest_runtime.ingest_source(source_type=source_type, location=location)
    typer.echo(json.dumps(result, ensure_ascii=True))


@app.command()
def query(
    query: str = typer.Option(..., "--query"),
    mode: str = typer.Option("fast", "--mode"),
) -> None:
    container = build_container()
    runtime = container.deep_research_runtime if mode == "deep" else container.fast_query_runtime
    response = runtime.run(
        query,
        build_execution_policy(
            task_type=TaskType.RESEARCH if mode == "deep" else TaskType.LOOKUP,
            complexity_level=(ComplexityLevel.L4_RESEARCH if mode == "deep" else ComplexityLevel.L1_DIRECT),
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.CLOUD_FIRST,
        ),
    )
    typer.echo(response.conclusion)


@app.command("approve-artifact")
def approve_artifact(artifact_id: str = typer.Option(..., "--artifact-id")) -> None:
    container = build_container()
    result = container.artifact_promotion_runtime.approve(artifact_id)
    payload = result if isinstance(result, dict) else result.model_dump(mode="json")
    typer.echo(json.dumps(cast(dict[str, object], payload), ensure_ascii=True))


__all__ = ["app", "set_container_factory"]
