from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import typer

from pkp.bootstrap import load_settings
from pkp.config import build_execution_policy, default_access_policy
from pkp.eval.offline_eval_service import (
    run_builtin_offline_eval,
    run_file_offline_eval,
)
from pkp.types import (
    AccessPolicy,
    ComplexityLevel,
    ExecutionLocation,
    ExecutionLocationPreference,
    ExternalRetrievalPolicy,
    Residency,
    RuntimeMode,
    TaskType,
)
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
RESIDENCY_OPTION = typer.Option(None, "--residency")
EXTERNAL_RETRIEVAL_OPTION = typer.Option(None, "--external-retrieval")
ALLOWED_RUNTIME_OPTION = typer.Option(None, "--allowed-runtime")
ALLOWED_LOCATION_OPTION = typer.Option(None, "--allowed-location")
SENSITIVITY_TAG_OPTION = typer.Option(None, "--sensitivity-tag")
SESSION_ID_OPTION = typer.Option(None, "--session-id")


@app.command()
def health(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    container = build_container()
    diagnostics_runtime = getattr(container, "diagnostics_runtime", None)
    if diagnostics_runtime is None:
        payload: dict[str, object] = {
            "status": "ok",
            "providers": [],
            "indices": {
                "documents": 0,
                "chunks": 0,
                "vectors": 0,
                "missing_vectors": 0,
            },
        }
    else:
        report = diagnostics_runtime.report()
        payload = report if isinstance(report, dict) else report.model_dump(mode="json")
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=True))
        return
    typer.echo(str(payload.get("status", "ok")))


@app.command("repair-indexes")
def repair_indexes() -> None:
    container = build_container()
    result = container.ingest_runtime.repair_indexes()
    typer.echo(json.dumps(result, ensure_ascii=True))


@app.command()
def ingest(
    source_type: str = typer.Option(..., "--source-type"),
    location: str | None = INGEST_LOCATION_OPTION,
    content: str | None = INGEST_CONTENT_OPTION,
    title: str | None = INGEST_TITLE_OPTION,
    residency: Residency | None = RESIDENCY_OPTION,
    external_retrieval: ExternalRetrievalPolicy | None = EXTERNAL_RETRIEVAL_OPTION,
    allowed_runtime: list[RuntimeMode] | None = ALLOWED_RUNTIME_OPTION,
    allowed_location: list[ExecutionLocation] | None = ALLOWED_LOCATION_OPTION,
    sensitivity_tag: list[str] | None = SENSITIVITY_TAG_OPTION,
) -> None:
    if location is None and content is None:
        raise typer.BadParameter("either --location or --content is required")

    container = build_container()
    access_policy = _build_access_policy(
        residency=residency,
        external_retrieval=external_retrieval,
        allowed_runtime=allowed_runtime,
        allowed_location=allowed_location,
        sensitivity_tag=sensitivity_tag,
    )
    result = container.ingest_runtime.ingest_source(
        source_type=source_type,
        location=location,
        content=content,
        title=title,
        access_policy=access_policy,
    )
    typer.echo(json.dumps(result, ensure_ascii=True))


@app.command("process-file")
def process_file(
    location: str = typer.Option(..., "--location"),
    title: str | None = INGEST_TITLE_OPTION,
    residency: Residency | None = RESIDENCY_OPTION,
    external_retrieval: ExternalRetrievalPolicy | None = EXTERNAL_RETRIEVAL_OPTION,
    allowed_runtime: list[RuntimeMode] | None = ALLOWED_RUNTIME_OPTION,
    allowed_location: list[ExecutionLocation] | None = ALLOWED_LOCATION_OPTION,
    sensitivity_tag: list[str] | None = SENSITIVITY_TAG_OPTION,
) -> None:
    container = build_container()
    access_policy = _build_access_policy(
        residency=residency,
        external_retrieval=external_retrieval,
        allowed_runtime=allowed_runtime,
        allowed_location=allowed_location,
        sensitivity_tag=sensitivity_tag,
    )
    result = container.ingest_runtime.process_file(
        location=location,
        title=title,
        access_policy=access_policy,
    )
    typer.echo(json.dumps(result, ensure_ascii=True))


@app.command("evaluate-retrieval")
def evaluate_retrieval(
    output_dir: str = typer.Option("data/eval/generated", "--output-dir"),
    top_k: int = typer.Option(5, "--top-k", min=1),
) -> None:
    result = run_builtin_offline_eval(Path(output_dir), top_k=top_k)
    payload = result if isinstance(result, dict) else result.model_dump(mode="json")
    typer.echo(json.dumps(cast(dict[str, object], payload), ensure_ascii=True))


@app.command("evaluate-file")
def evaluate_file(
    location: str = typer.Option(..., "--location"),
    questions_file: str = typer.Option(..., "--questions-file"),
    output_dir: str = typer.Option("data/eval/generated-file", "--output-dir"),
    top_k: int = typer.Option(5, "--top-k", min=1),
) -> None:
    result = run_file_offline_eval(
        file_path=Path(location),
        questions_path=Path(questions_file),
        output_dir=Path(output_dir),
        top_k=top_k,
    )
    payload = result if isinstance(result, dict) else result.model_dump(mode="json")
    typer.echo(json.dumps(cast(dict[str, object], payload), ensure_ascii=True))


@app.command()
def query(
    query: str = typer.Option(..., "--query"),
    mode: str = typer.Option("fast", "--mode"),
    json_output: bool = typer.Option(False, "--json"),
    session_id: str | None = SESSION_ID_OPTION,
    source_scope: list[str] | None = SOURCE_SCOPE_OPTION,
    latency_budget: int | None = LATENCY_BUDGET_OPTION,
    token_budget: int | None = TOKEN_BUDGET_OPTION,
    execution_location_preference: ExecutionLocationPreference | None = EXECUTION_LOCATION_OPTION,
    fallback_allowed: bool | None = FALLBACK_ALLOWED_OPTION,
    cost_budget: float | None = COST_BUDGET_OPTION,
    residency: Residency | None = RESIDENCY_OPTION,
    external_retrieval: ExternalRetrievalPolicy | None = EXTERNAL_RETRIEVAL_OPTION,
    allowed_runtime: list[RuntimeMode] | None = ALLOWED_RUNTIME_OPTION,
    allowed_location: list[ExecutionLocation] | None = ALLOWED_LOCATION_OPTION,
    sensitivity_tag: list[str] | None = SENSITIVITY_TAG_OPTION,
) -> None:
    container = build_container()
    settings = load_settings()
    is_deep = mode == "deep"
    effective_latency_budget = (
        latency_budget if latency_budget is not None else settings.runtime.default_wall_clock_budget_seconds
    )
    effective_cost_budget = cost_budget if cost_budget is not None else 1.0
    effective_token_budget = token_budget if token_budget is not None else settings.runtime.max_token_budget
    effective_execution_location_preference = (
        execution_location_preference or settings.runtime.execution_location_preference
    )
    effective_fallback_allowed = fallback_allowed if fallback_allowed is not None else settings.runtime.fallback_allowed
    access_policy = _build_access_policy(
        residency=residency,
        external_retrieval=external_retrieval,
        allowed_runtime=allowed_runtime,
        allowed_location=allowed_location,
        sensitivity_tag=sensitivity_tag,
    )
    policy = build_execution_policy(
        task_type=TaskType.RESEARCH if is_deep else TaskType.LOOKUP,
        complexity_level=ComplexityLevel.L4_RESEARCH if is_deep else ComplexityLevel.L1_DIRECT,
        access_policy=access_policy or default_access_policy(),
        source_scope=source_scope or [],
        latency_budget=effective_latency_budget,
        cost_budget=effective_cost_budget,
        token_budget=effective_token_budget,
        execution_location_preference=effective_execution_location_preference,
        fallback_allowed=effective_fallback_allowed,
    )
    if is_deep:
        response = container.deep_research_runtime.run(query, policy, session_id=session_id or "default")
    else:
        response = container.fast_query_runtime.run(query, policy)
    if json_output:
        typer.echo(json.dumps(cast(dict[str, object], response.model_dump(mode="json")), ensure_ascii=True))
        return
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


@app.command("show-session")
def show_session(session_id: str = typer.Option(..., "--session-id")) -> None:
    container = build_container()
    snapshot = container.session_runtime.get(session_id)
    typer.echo(json.dumps(cast(dict[str, object], asdict(snapshot)), ensure_ascii=True))


__all__ = ["app", "set_container_factory"]


def main() -> None:
    app()


def _build_access_policy(
    *,
    residency: Residency | None,
    external_retrieval: ExternalRetrievalPolicy | None,
    allowed_runtime: list[RuntimeMode] | None,
    allowed_location: list[ExecutionLocation] | None,
    sensitivity_tag: list[str] | None,
) -> AccessPolicy | None:
    if (
        residency is None
        and external_retrieval is None
        and not allowed_runtime
        and not allowed_location
        and not sensitivity_tag
    ):
        return None

    defaults = default_access_policy()
    return AccessPolicy(
        residency=residency or defaults.residency,
        external_retrieval=external_retrieval or defaults.external_retrieval,
        allowed_runtimes=frozenset(allowed_runtime or defaults.allowed_runtimes),
        allowed_locations=frozenset(allowed_location or defaults.allowed_locations),
        sensitivity_tags=frozenset(sensitivity_tag or defaults.sensitivity_tags),
    )


if __name__ == "__main__":
    main()
