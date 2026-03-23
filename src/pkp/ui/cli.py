from __future__ import annotations

import json
from typing import cast

import typer

from pkp.bootstrap import load_settings
from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.dependencies import build_container, set_container_factory

app = typer.Typer(add_completion=False)
SOURCE_SCOPE_OPTION = typer.Option(None, "--source-scope")
LATENCY_BUDGET_OPTION = typer.Option(None, "--latency-budget")
TOKEN_BUDGET_OPTION = typer.Option(None, "--token-budget")
EXECUTION_LOCATION_OPTION = typer.Option(None, "--execution-location-preference")
FALLBACK_ALLOWED_OPTION = typer.Option(None, "--fallback-allowed/--no-fallback-allowed")
COST_BUDGET_OPTION = typer.Option(None, "--cost-budget")
INGEST_LOCATION_OPTION = typer.Option(None, "--location")
INGEST_CONTENT_OPTION = typer.Option(None, "--content")
INGEST_TITLE_OPTION = typer.Option(None, "--title")


@app.command()
def health() -> None:
    typer.echo("ok")


@app.command()
def ingest(
    source_type: str = typer.Option(..., "--source-type"),
    location: str | None = INGEST_LOCATION_OPTION,
    content: str | None = INGEST_CONTENT_OPTION,
    title: str | None = INGEST_TITLE_OPTION,
) -> None:
    if location is None and content is None:
        raise typer.BadParameter("either --location or --content is required")

    container = build_container()
    result = container.ingest_runtime.ingest_source(
        source_type=source_type,
        location=location,
        content=content,
        title=title,
    )
    typer.echo(json.dumps(result, ensure_ascii=True))


@app.command()
def query(
    query: str = typer.Option(..., "--query"),
    mode: str = typer.Option("fast", "--mode"),
    source_scope: list[str] | None = SOURCE_SCOPE_OPTION,
    latency_budget: int | None = LATENCY_BUDGET_OPTION,
    token_budget: int | None = TOKEN_BUDGET_OPTION,
    execution_location_preference: ExecutionLocationPreference | None = EXECUTION_LOCATION_OPTION,
    fallback_allowed: bool | None = FALLBACK_ALLOWED_OPTION,
    cost_budget: float | None = COST_BUDGET_OPTION,
) -> None:
    container = build_container()
    settings = load_settings()
    is_deep = mode == "deep"
    runtime = container.deep_research_runtime if mode == "deep" else container.fast_query_runtime
    effective_latency_budget = (
        latency_budget if latency_budget is not None else settings.runtime.default_wall_clock_budget_seconds
    )
    effective_cost_budget = cost_budget if cost_budget is not None else 1.0
    effective_token_budget = token_budget if token_budget is not None else settings.runtime.max_token_budget
    effective_execution_location_preference = (
        execution_location_preference or settings.runtime.execution_location_preference
    )
    effective_fallback_allowed = fallback_allowed if fallback_allowed is not None else settings.runtime.fallback_allowed
    response = runtime.run(
        query,
        build_execution_policy(
            task_type=TaskType.RESEARCH if is_deep else TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L4_RESEARCH if is_deep else ComplexityLevel.L1_DIRECT,
            access_policy=default_access_policy(),
            source_scope=source_scope or [],
            latency_budget=effective_latency_budget,
            cost_budget=effective_cost_budget,
            token_budget=effective_token_budget,
            execution_location_preference=effective_execution_location_preference,
            fallback_allowed=effective_fallback_allowed,
        ),
    )
    typer.echo(response.conclusion)


@app.command("approve-artifact")
def approve_artifact(artifact_id: str = typer.Option(..., "--artifact-id")) -> None:
    container = build_container()
    result = container.artifact_promotion_runtime.approve(artifact_id)
    payload = result if isinstance(result, dict) else result.model_dump(mode="json")
    typer.echo(json.dumps(cast(dict[str, object], payload), ensure_ascii=True))


@app.command("list-artifacts")
def list_artifacts() -> None:
    container = build_container()
    artifacts = container.artifact_promotion_runtime.list_artifacts()
    payload = [artifact if isinstance(artifact, dict) else artifact.model_dump(mode="json") for artifact in artifacts]
    typer.echo(json.dumps(cast(list[dict[str, object]], payload), ensure_ascii=True))


@app.command("show-artifact")
def show_artifact(artifact_id: str = typer.Option(..., "--artifact-id")) -> None:
    container = build_container()
    artifact = container.artifact_promotion_runtime.get_artifact(artifact_id)
    payload = artifact if isinstance(artifact, dict) else artifact.model_dump(mode="json")
    typer.echo(json.dumps(cast(dict[str, object], payload), ensure_ascii=True))


__all__ = ["app", "set_container_factory"]
