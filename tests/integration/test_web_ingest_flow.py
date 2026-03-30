from pathlib import Path

from typer.testing import CliRunner

from pkp.interfaces._bootstrap import build_test_container
from pkp.interfaces._config import build_execution_policy, default_access_policy
from pkp.interfaces._runtime.adapters import RetrievedCandidate
from pkp.schema._types import (
    AccessPolicy,
    ComplexityLevel,
    ExecutionLocation,
    ExecutionLocationPreference,
    ExternalRetrievalPolicy,
    TaskType,
)
from pkp.interfaces._ui.cli import app as cli_app
from pkp.interfaces._ui.dependencies import clear_container_factory, set_container_factory

runner = CliRunner()


def test_web_ingest_flow_via_cli_location_compatibility(tmp_path: Path) -> None:
    container = build_test_container(tmp_path)
    clear_container_factory()
    set_container_factory(lambda: container)

    ingest = runner.invoke(
        cli_app,
        [
            "ingest",
            "--source-type",
            "web",
            "--location",
            "https://example.com/article",
        ],
    )
    response = container.fast_query_runtime.run(
        "What happens when cloud synthesis fails?",
        build_execution_policy(
            task_type=TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L1_DIRECT,
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
    )

    assert ingest.exit_code == 0
    assert response.evidence


def test_browser_clip_ingest_flow_via_inline_html(tmp_path: Path) -> None:
    container = build_test_container(tmp_path)
    clear_container_factory()
    set_container_factory(lambda: container)

    ingest = runner.invoke(
        cli_app,
        [
            "ingest",
            "--source-type",
            "browser_clip",
            "--title",
            "Clipped Article",
            "--content",
            (
                "<html><body><article><h1>Clipped Article</h1>"
                "<p>Cloud synthesis falls back to local generation.</p></article></body></html>"
            ),
        ],
    )
    response = container.fast_query_runtime.run(
        "What happens when cloud synthesis fails?",
        build_execution_policy(
            task_type=TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L1_DIRECT,
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
    )

    assert ingest.exit_code == 0
    assert any("cloud synthesis falls back to local generation" in item.text.lower() for item in response.evidence)


def test_runtime_query_policy_can_block_external_evidence(tmp_path: Path) -> None:
    container = build_test_container(tmp_path)
    clear_container_factory()
    set_container_factory(lambda: container)
    retrieval_adapter = container.deep_research_runtime._retrieval_service
    retrieval_service = retrieval_adapter._retrieval_service
    retrieval_service._web_retriever = lambda _query, _scope: [
        RetrievedCandidate(
            chunk_id="external-1",
            doc_id="external-doc-1",
            source_id="https://example.com/external",
            text="External evidence about retrieval reliability trends.",
            citation_anchor="External Note",
            score=0.9,
            rank=1,
            source_kind="external",
        )
    ]

    runner.invoke(
        cli_app,
        [
            "ingest",
            "--source-type",
            "markdown",
            "--title",
            "Internal Notes",
            "--content",
            "# Reliability\n\nInternal evidence about retrieval reliability.",
        ],
    )

    allow_response = container.deep_research_runtime.run(
        "Research retrieval reliability trends",
        build_execution_policy(
            task_type=TaskType.RESEARCH,
            complexity_level=ComplexityLevel.L4_RESEARCH,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
    )
    deny_response = container.deep_research_runtime.run(
        "Research retrieval reliability trends",
        build_execution_policy(
            task_type=TaskType.RESEARCH,
            complexity_level=ComplexityLevel.L4_RESEARCH,
            access_policy=AccessPolicy(
                external_retrieval=ExternalRetrievalPolicy.DENY,
                allowed_locations=frozenset({ExecutionLocation.LOCAL, ExecutionLocation.CLOUD}),
            ),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
    )

    assert any(item.evidence_kind == "external" for item in allow_response.evidence)
    assert all(item.evidence_kind != "external" for item in deny_response.evidence)
