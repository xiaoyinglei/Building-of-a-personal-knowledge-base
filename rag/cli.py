from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageConfig
from rag.agent import AgentTaskRequest
from rag.answer_benchmarks import AnswerBenchmarkEvaluator, build_chat_judge
from rag.benchmark_diagnostics import (
    BenchmarkDiagnosticsPostProcessor,
    build_diagnostic_context,
    build_runtime_for_diagnostics,
    load_run_summary,
)
from rag.benchmarks import (
    FIQA_DATASET,
    MEDICAL_RETRIEVAL_DATASET,
    RetrievalBenchmarkEvaluator,
    benchmark_access_policy,
    benchmark_dataset_spec,
    build_runtime_for_benchmark,
    default_benchmark_paths,
    download_public_benchmark,
    ensure_benchmark_layout,
    ingest_prepared_documents,
    prepare_public_benchmark,
)
from rag.retrieval import QueryMode, QueryOptions
from rag.schema.core import SourceType
from rag.schema.runtime import ExecutionLocationPreference
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
DATASET_OPTION = typer.Option("--dataset", help="Public benchmark dataset.")


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


@app.command("analyze-task")
def analyze_task(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    query: Annotated[str | None, QUERY_OPTION] = None,
    json_output: Annotated[bool, JSON_OPTION] = False,
    allow_web: Annotated[bool, typer.Option("--allow-web/--no-allow-web")] = False,
    expected_output: Annotated[str, typer.Option("--expected-output")] = "structured_analysis_report",
    response_style: Annotated[str, typer.Option("--response-style")] = "formal",
    max_subtasks: Annotated[int, typer.Option("--max-subtasks")] = 5,
    retry_budget: Annotated[int, typer.Option("--retry-budget")] = 2,
) -> None:
    if query is None or not query.strip():
        raise typer.BadParameter("--query is required")
    with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
        result = runtime.analyze_task(
            AgentTaskRequest(
                user_query=query,
                allow_web=allow_web,
                expected_output=expected_output,
                response_style=response_style,
                max_subtasks=max_subtasks,
                retry_budget=retry_budget,
            )
        )
    if json_output:
        _echo_json(result)
        return
    typer.echo(result.final_report.executive_summary if result.final_report is not None else "")


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


@app.command("benchmark-download")
def benchmark_download(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    result = download_public_benchmark(dataset, paths.raw_dir if raw_dir is None else raw_dir, force=force)
    _echo_json(result)


@app.command("benchmark-prepare")
def benchmark_prepare(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
    prepared_dir: Annotated[Path | None, typer.Option("--prepared-dir")] = None,
    split: Annotated[str | None, typer.Option("--split")] = None,
    no_mini: Annotated[bool, typer.Option("--no-mini")] = False,
    mini_query_count: Annotated[int | None, typer.Option("--mini-query-count")] = None,
    mini_doc_count: Annotated[int | None, typer.Option("--mini-doc-count")] = None,
) -> None:
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    spec = benchmark_dataset_spec(dataset)
    result = prepare_public_benchmark(
        dataset,
        paths.raw_dir if raw_dir is None else raw_dir,
        paths.prepared_root if prepared_dir is None else prepared_dir,
        split=split or spec.default_split,
        build_mini=not no_mini,
        mini_query_count=mini_query_count,
        mini_target_doc_count=mini_doc_count,
    )
    _echo_json(result)


@app.command("benchmark-ingest")
def benchmark_ingest(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    variant: Annotated[str, typer.Option("--variant")] = "full",
    profile_id: Annotated[str, PROFILE_OPTION] = "local_full",
    storage_root: Annotated[Path | None, STORAGE_ROOT_OPTION] = None,
    documents_path: Annotated[Path | None, typer.Option("--documents-path")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 64,
    chat_provider: Annotated[str | None, typer.Option("--chat-provider")] = None,
    chat_model: Annotated[str | None, typer.Option("--chat-model")] = None,
    chat_model_path: Annotated[str | None, typer.Option("--chat-model-path")] = None,
    chat_backend: Annotated[str | None, typer.Option("--chat-backend")] = None,
    vector_backend: Annotated[str, typer.Option("--vector-backend")] = "sqlite",
    vector_dsn: Annotated[str | None, typer.Option("--vector-dsn")] = None,
    vector_namespace: Annotated[str | None, typer.Option("--vector-namespace")] = None,
    vector_collection_prefix: Annotated[str | None, typer.Option("--vector-collection-prefix")] = None,
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error")] = False,
    skip_graph_extraction: Annotated[bool, typer.Option("--skip-graph-extraction/--with-graph-extraction")] = True,
) -> None:
    if variant not in {"full", "mini"}:
        raise typer.BadParameter("variant must be one of: full, mini")
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    runtime = build_runtime_for_benchmark(
        storage_root=storage_root or paths.index_variant_dir(variant),
        profile_id=profile_id,
        require_chat=not skip_graph_extraction,
        require_rerank=False,
        skip_graph_extraction=skip_graph_extraction,
        chat_provider_kind=chat_provider,
        chat_model=chat_model,
        chat_model_path=chat_model_path,
        chat_backend=chat_backend,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    try:
        result = ingest_prepared_documents(
            runtime,
            dataset=dataset,
            documents_path=documents_path or (paths.prepared_variant_dir(variant) / "documents.jsonl"),
            batch_size=max(batch_size, 1),
            continue_on_error=continue_on_error,
        )
        _echo_json(result)
    finally:
        runtime.close()


@app.command("benchmark-evaluate")
def benchmark_evaluate(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    variant: Annotated[str, typer.Option("--variant")] = "full",
    profile_id: Annotated[str, PROFILE_OPTION] = "local_full",
    storage_root: Annotated[Path | None, STORAGE_ROOT_OPTION] = None,
    queries_path: Annotated[Path | None, typer.Option("--queries-path")] = None,
    qrels_path: Annotated[Path | None, typer.Option("--qrels-path")] = None,
    eval_dir: Annotated[Path | None, typer.Option("--eval-dir")] = None,
    mode: Annotated[QueryMode, MODE_OPTION] = QueryMode.MIX,
    top_k: Annotated[int, typer.Option("--top-k")] = 10,
    chunk_top_k: Annotated[int | None, typer.Option("--chunk-top-k")] = None,
    vector_backend: Annotated[str, typer.Option("--vector-backend")] = "sqlite",
    vector_dsn: Annotated[str | None, typer.Option("--vector-dsn")] = None,
    vector_namespace: Annotated[str | None, typer.Option("--vector-namespace")] = None,
    vector_collection_prefix: Annotated[str | None, typer.Option("--vector-collection-prefix")] = None,
    rerank_enabled: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    enable_query_understanding_llm: Annotated[
        bool, typer.Option("--enable-query-understanding-llm/--disable-query-understanding-llm")
    ] = False,
    chat_provider: Annotated[str | None, typer.Option("--chat-provider")] = None,
    chat_model: Annotated[str | None, typer.Option("--chat-model")] = None,
    chat_model_path: Annotated[str | None, typer.Option("--chat-model-path")] = None,
    chat_backend: Annotated[str | None, typer.Option("--chat-backend")] = None,
    split: Annotated[str | None, typer.Option("--split")] = None,
) -> None:
    if variant not in {"full", "mini"}:
        raise typer.BadParameter("variant must be one of: full, mini")
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    spec = benchmark_dataset_spec(dataset)
    top_k = max(top_k, 1)
    chunk_top_k = max(chunk_top_k or max(top_k * 4, 40), top_k)
    runtime = build_runtime_for_benchmark(
        storage_root=storage_root or paths.index_variant_dir(variant),
        profile_id=profile_id,
        require_chat=enable_query_understanding_llm,
        require_rerank=rerank_enabled,
        chat_provider_kind=chat_provider,
        chat_model=chat_model,
        chat_model_path=chat_model_path,
        chat_backend=chat_backend,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    try:
        runtime.retrieval_service.query_understanding_service._enable_llm = enable_query_understanding_llm
        summary = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset=dataset,
            split=split or spec.default_split,
            retrieval_mode=mode.value,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            rerank_enabled=rerank_enabled,
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
            access_policy=benchmark_access_policy(),
        ).evaluate(
            queries_path=queries_path or (paths.prepared_variant_dir(variant) / "queries.jsonl"),
            qrels_path=qrels_path or (paths.prepared_variant_dir(variant) / "qrels.jsonl"),
            eval_dir=eval_dir or paths.eval_variant_dir("retrieval", variant),
        )
        payload = summary.as_json()
        payload["variant"] = variant
        _echo_json(payload)
    finally:
        runtime.close()


@app.command("benchmark-answer-evaluate")
def benchmark_answer_evaluate(
    dataset: Annotated[str, DATASET_OPTION] = MEDICAL_RETRIEVAL_DATASET,
    variant: Annotated[str, typer.Option("--variant")] = "mini",
    profile_id: Annotated[str, PROFILE_OPTION] = "local_full",
    storage_root: Annotated[Path | None, STORAGE_ROOT_OPTION] = None,
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    mode: Annotated[QueryMode, MODE_OPTION] = QueryMode.NAIVE,
    top_k: Annotated[int, typer.Option("--top-k")] = 10,
    chunk_top_k: Annotated[int, typer.Option("--chunk-top-k")] = 20,
    retrieval_pool_k: Annotated[int, typer.Option("--retrieval-pool-k")] = 20,
    rerank_enabled: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    rerank_pool_k: Annotated[int, typer.Option("--rerank-pool-k")] = 10,
    answer_context_top_k: Annotated[int | None, typer.Option("--answer-context-top-k")] = None,
    query_limit: Annotated[int | None, typer.Option("--query-limit")] = None,
    judge_subset_size: Annotated[int, typer.Option("--judge-subset-size")] = 250,
    judge_seed: Annotated[int, typer.Option("--judge-seed")] = 42,
    local_judge_profile: Annotated[str, typer.Option("--local-judge-profile")] = "local_full",
    review_judge_profile: Annotated[str, typer.Option("--review-judge-profile")] = "local_retrieval_cloud_chat",
    disable_review: Annotated[bool, typer.Option("--disable-review")] = False,
    review_confidence_threshold: Annotated[float, typer.Option("--review-confidence-threshold")] = 0.75,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    embedding_model_path: Annotated[str | None, typer.Option("--embedding-model-path")] = None,
    chat_provider: Annotated[str | None, typer.Option("--chat-provider")] = None,
    chat_model: Annotated[str | None, typer.Option("--chat-model")] = None,
    chat_model_path: Annotated[str | None, typer.Option("--chat-model-path")] = None,
    chat_backend: Annotated[str | None, typer.Option("--chat-backend")] = None,
) -> None:
    if variant not in {"full", "mini"}:
        raise typer.BadParameter("variant must be one of: full, mini")
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset), tasks=("retrieval", "ingest"))
    prepared_dir = paths.prepared_variant_dir(variant)
    resolved_storage_root = storage_root or paths.index_variant_dir(variant)
    resolved_output_root = output_root or (Path("data") / "eval" / "answers" / dataset / variant)
    runtime = build_runtime_for_benchmark(
        storage_root=resolved_storage_root,
        profile_id=profile_id,
        require_chat=True,
        require_rerank=rerank_enabled,
        embedding_provider_kind=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        chat_provider_kind=chat_provider,
        chat_model=chat_model,
        chat_model_path=chat_model_path,
        chat_backend=chat_backend,
    )
    try:
        local_judge = build_chat_judge(profile_id=local_judge_profile, allow_missing=False)
        review_judge = (
            None
            if disable_review
            else build_chat_judge(profile_id=review_judge_profile, require_cloud=True, allow_missing=True)
        )
        payload = AnswerBenchmarkEvaluator(
            runtime=runtime,
            dataset=dataset,
            variant=variant,
            retrieval_mode=mode.value,
            top_k=max(top_k, 1),
            chunk_top_k=max(chunk_top_k, max(top_k, 1)),
            retrieval_pool_k=retrieval_pool_k,
            rerank_enabled=rerank_enabled,
            rerank_pool_k=rerank_pool_k,
            answer_context_top_k=answer_context_top_k,
            judge_subset_size=max(judge_subset_size, 0),
            judge_seed=judge_seed,
            local_judge=local_judge,
            review_judge=review_judge,
            review_confidence_threshold=review_confidence_threshold,
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
        ).evaluate(
            queries_path=prepared_dir / "queries.jsonl",
            qrels_path=prepared_dir / "qrels.jsonl",
            documents_path=prepared_dir / "documents.jsonl",
            output_root=resolved_output_root,
            query_limit=query_limit,
        )
        _echo_json(payload)
    finally:
        runtime.close()


@app.command("benchmark-diagnose")
def benchmark_diagnose(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    run_id: Annotated[str, typer.Option("--run-id")] = ...,
    variant: Annotated[str, typer.Option("--variant")] = "full",
    profile_id: Annotated[str, PROFILE_OPTION] = "local_full",
    storage_root: Annotated[Path | None, STORAGE_ROOT_OPTION] = None,
    queries_path: Annotated[Path | None, typer.Option("--queries-path")] = None,
    qrels_path: Annotated[Path | None, typer.Option("--qrels-path")] = None,
    diagnostics_root: Annotated[Path, typer.Option("--diagnostics-root")] = Path("data/eval/diagnostics"),
    mode: Annotated[QueryMode | None, MODE_OPTION] = None,
    top_k: Annotated[int | None, typer.Option("--top-k")] = None,
    chunk_top_k: Annotated[int | None, typer.Option("--chunk-top-k")] = None,
    retrieval_pool_k: Annotated[int | None, typer.Option("--retrieval-pool-k")] = None,
    rerank_enabled: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    rerank_pool_k: Annotated[int | None, typer.Option("--rerank-pool-k")] = None,
    disable_parent_backfill: Annotated[bool, typer.Option("--disable-parent-backfill")] = False,
    enable_query_understanding_llm: Annotated[
        bool, typer.Option("--enable-query-understanding-llm/--disable-query-understanding-llm")
    ] = False,
    query_limit: Annotated[int | None, typer.Option("--query-limit")] = None,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    embedding_model_path: Annotated[str | None, typer.Option("--embedding-model-path")] = None,
) -> None:
    if variant not in {"full", "mini"}:
        raise typer.BadParameter("variant must be one of: full, mini")
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    run_dir = paths.eval_variant_dir("retrieval", variant) / "runs" / run_id
    summary = load_run_summary(run_dir)
    context = build_diagnostic_context(
        dataset=dataset,
        run_id=run_id,
        variant=variant,
        profile_id=profile_id,
        storage_root=storage_root or paths.index_variant_dir(variant),
        queries_path=queries_path or (paths.prepared_variant_dir(variant) / "queries.jsonl"),
        qrels_path=qrels_path or (paths.prepared_variant_dir(variant) / "qrels.jsonl"),
        retrieval_mode=(mode.value if mode is not None else str(summary.get("retrieval_mode") or "mix")),
        top_k=max(top_k or int(summary.get("top_k") or 10), 1),
        chunk_top_k=max(chunk_top_k or int(summary.get("chunk_top_k") or 10), top_k or int(summary.get("top_k") or 10)),
        retrieval_pool_k=retrieval_pool_k,
        rerank_enabled=bool(summary.get("rerank_enabled", True)) if rerank_enabled is None else rerank_enabled,
        rerank_pool_k=rerank_pool_k,
        enable_parent_backfill=not disable_parent_backfill,
        query_limit=query_limit,
        enable_query_understanding_llm=enable_query_understanding_llm,
        embedding_provider_kind=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
    )
    runtime = build_runtime_for_diagnostics(context)
    try:
        payload = BenchmarkDiagnosticsPostProcessor(runtime=runtime, context=context).diagnose(
            diagnostics_root=diagnostics_root
        )
        _echo_json({key: str(value) for key, value in payload.items()})
    finally:
        runtime.close()


__all__ = ["app", "main"]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
