from pathlib import Path

import fitz
import pytest
from docx import Document as WordDocument
from PIL import Image, ImageDraw

from pkp.repo.interfaces import OcrRegion, OcrResult
from pkp.service.ingest_service import IngestService
from pkp.types.content import SourceType
from pkp.types.processing import ChunkingStrategy, ChunkRole


class FakeOcrVisionRepo:
    def extract(self, image_path: Path) -> OcrResult:
        del image_path
        return OcrResult(
            visible_text="Quarterly revenue chart. Figure 1. Revenue grew 20 percent.",
            visual_semantics="bar chart with a caption below it",
            regions=[
                OcrRegion(text="Quarterly revenue chart", bbox=(0, 0, 240, 60)),
                OcrRegion(text="Figure 1", bbox=(0, 61, 120, 90)),
                OcrRegion(text="Revenue grew 20 percent", bbox=(0, 91, 260, 140)),
            ],
        )


def _write_markdown(path: Path) -> None:
    path.write_text(
        "# Quarterly Review\n\n"
        "Overview paragraph for the report.\n\n"
        "## Revenue\n\n"
        "Revenue expanded because enterprise demand improved.\n\n"
        "## Risks\n\n"
        "Supply chain risk remains manageable.\n",
        encoding="utf-8",
    )


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Quarterly Review\nRevenue expanded because enterprise demand improved.")
    page = document.new_page()
    page.insert_text((72, 72), "Risks\nSupply chain risk remains manageable.")
    document.save(path)
    document.close()


def _write_docx(path: Path, *, with_headings: bool) -> None:
    document = WordDocument()
    if with_headings:
        document.add_heading("Quarterly Review", level=1)
        document.add_paragraph("Overview paragraph for the report.")
        document.add_heading("Revenue", level=2)
        document.add_paragraph("Revenue expanded because enterprise demand improved.")
        document.add_heading("Risks", level=2)
        document.add_paragraph("Supply chain risk remains manageable.")
    else:
        document.add_paragraph("Quarterly Review")
        document.add_paragraph("Overview paragraph for the report.")
        document.add_paragraph("Revenue expanded because enterprise demand improved.")
        document.add_paragraph("Supply chain risk remains manageable.")
    document.save(path)


def _write_image(path: Path) -> None:
    image = Image.new("RGB", (640, 320), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 40), "Quarterly revenue chart", fill="black")
    draw.text((20, 120), "Figure 1", fill="black")
    draw.text((20, 200), "Revenue grew 20 percent", fill="black")
    image.save(path)


@pytest.mark.parametrize(
    ("filename", "writer", "expected_type", "expected_strategy"),
    [
        ("quarterly.md", _write_markdown, SourceType.MARKDOWN, ChunkingStrategy.HIERARCHICAL),
        ("quarterly.pdf", _write_pdf, SourceType.PDF, ChunkingStrategy.HYBRID),
        (
            "quarterly.docx",
            lambda path: _write_docx(path, with_headings=True),
            SourceType.DOCX,
            ChunkingStrategy.HIERARCHICAL,
        ),
        ("quarterly.png", _write_image, SourceType.IMAGE, ChunkingStrategy.IMAGE),
    ],
)
def test_ingest_file_builds_uniform_processing_package(
    tmp_path: Path,
    filename: str,
    writer: object,
    expected_type: SourceType,
    expected_strategy: ChunkingStrategy,
) -> None:
    path = tmp_path / filename
    writer(path)
    service = IngestService.create_in_memory(tmp_path / "runtime", ocr_repo=FakeOcrVisionRepo())

    result = service.ingest_file(location=str(path), file_path=path, owner="user")

    assert result.processing is not None
    assert result.source.source_type is expected_type
    assert result.processing.analysis.source_type is expected_type
    assert result.processing.routing.selected_strategy is expected_strategy
    assert result.processing.stats.total_chunks == (
        len(result.processing.parent_chunks)
        + len(result.processing.child_chunks)
        + len(result.processing.special_chunks)
    )
    assert result.processing.metadata_summary["schema_version"] == "chunk-pipeline/v1"
    assert result.processing.child_chunks or result.processing.special_chunks


def test_docx_secondary_route_uses_hybrid_when_heading_quality_is_low(tmp_path: Path) -> None:
    path = tmp_path / "low-quality.docx"
    _write_docx(path, with_headings=False)
    service = IngestService.create_in_memory(tmp_path / "runtime", ocr_repo=FakeOcrVisionRepo())

    result = service.ingest_file(location=str(path), file_path=path, owner="user")

    assert result.processing is not None
    assert result.processing.analysis.heading_quality_score < 0.5
    assert result.processing.routing.selected_strategy is ChunkingStrategy.HYBRID


def test_image_pipeline_produces_special_chunks_and_parent_child_links(tmp_path: Path) -> None:
    path = tmp_path / "chart.png"
    _write_image(path)
    service = IngestService.create_in_memory(tmp_path / "runtime", ocr_repo=FakeOcrVisionRepo())

    result = service.ingest_file(location=str(path), file_path=path, owner="user")

    assert result.processing is not None
    assert {chunk.chunk_role for chunk in result.processing.parent_chunks} == {ChunkRole.PARENT}
    assert all(chunk.parent_chunk_id for chunk in result.processing.child_chunks)
    assert {chunk.special_chunk_type for chunk in result.processing.special_chunks} >= {
        "image_summary",
        "ocr_region",
    }
    assert len({chunk.chunk_id for chunk in result.processing.special_chunks}) == len(
        result.processing.special_chunks
    )


def test_pipeline_reports_clear_error_for_unsupported_file_types(tmp_path: Path) -> None:
    path = tmp_path / "notes.xlsx"
    path.write_text("not supported", encoding="utf-8")
    service = IngestService.create_in_memory(tmp_path / "runtime", ocr_repo=FakeOcrVisionRepo())

    with pytest.raises(ValueError, match="Unsupported file type"):
        service.ingest_file(location=str(path), file_path=path, owner="user")
