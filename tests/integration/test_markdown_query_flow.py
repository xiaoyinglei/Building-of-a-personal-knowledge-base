from pathlib import Path

from typer.testing import CliRunner

from pkp.bootstrap import build_test_container
from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.cli import app as cli_app
from pkp.ui.dependencies import clear_container_factory, set_container_factory

runner = CliRunner()


def test_markdown_query_flow_with_inline_markdown_ingest(
    tmp_path: Path,
) -> None:
    container = build_test_container(tmp_path)
    clear_container_factory()
    set_container_factory(lambda: container)

    ingest = runner.invoke(
        cli_app,
        [
            "ingest",
            "--source-type",
            "markdown",
            "--title",
            "Agent RAG Overview",
            "--content",
            "# Reliability First\n\nEvidence quality is more important than fluent synthesis.",
        ],
    )
    response = container.fast_query_runtime.run(
        "What is more important than fluent synthesis?",
        build_execution_policy(
            task_type=TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L1_DIRECT,
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
    )

    assert ingest.exit_code == 0
    assert response.conclusion
    assert response.evidence
    assert any(
        "evidence quality is more important than fluent synthesis" in item.text.lower() for item in response.evidence
    )
    assert response.uncertainty
    assert response.preservation_suggestion is not None
