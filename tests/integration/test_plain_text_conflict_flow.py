from pathlib import Path

from typer.testing import CliRunner

from pkp.bootstrap import build_test_container
from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.cli import app as cli_app
from pkp.ui.dependencies import clear_container_factory, set_container_factory

runner = CliRunner()


def test_plain_text_conflict_flow_surfaces_conflicts_for_inline_pasted_text(
    tmp_path: Path,
) -> None:
    container = build_test_container(tmp_path)
    clear_container_factory()
    set_container_factory(lambda: container)

    first_ingest = runner.invoke(
        cli_app,
        [
            "ingest",
            "--source-type",
            "pasted_text",
            "--title",
            "Conflict A",
            "--content",
            "The default retrieval path is Fast Path.",
        ],
    )
    second_ingest = runner.invoke(
        cli_app,
        [
            "ingest",
            "--source-type",
            "pasted_text",
            "--title",
            "Conflict B",
            "--content",
            "The default retrieval path is Deep Path.",
        ],
    )
    response = container.deep_research_runtime.run(
        "Compare the default retrieval path in the two documents",
        build_execution_policy(
            task_type=TaskType.COMPARISON,
            complexity_level=ComplexityLevel.L3_COMPARATIVE,
            access_policy=default_access_policy(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
    )

    assert first_ingest.exit_code == 0
    assert second_ingest.exit_code == 0
    assert response.runtime_mode == "deep"
    assert response.differences_or_conflicts
