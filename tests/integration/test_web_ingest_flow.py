from pathlib import Path

from typer.testing import CliRunner

from pkp.bootstrap import build_test_container
from pkp.config import build_execution_policy, default_access_policy
from pkp.types import ComplexityLevel, ExecutionLocationPreference, TaskType
from pkp.ui.cli import app as cli_app
from pkp.ui.dependencies import clear_container_factory, set_container_factory

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
