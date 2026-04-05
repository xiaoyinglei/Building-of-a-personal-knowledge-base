from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageDraw

from rag.utils._contracts import OcrResult, ParsedDocument
from tests.support import make_ingest_service


class FakeOcrVisionRepo:
    def extract(self, image_path: Path) -> OcrResult:
        return OcrResult(
            visible_text="diagram label",
            visual_semantics="two blocks and a caption",
            regions=[],
        )


def _write_text_image(path: Path) -> None:
    image = Image.new("RGB", (520, 180), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Quarterly revenue chart", fill="black")
    draw.text((20, 90), "Revenue grew 20 percent", fill="black")
    image.save(path)


def test_image_ingest_stores_visible_text_and_visual_semantics(tmp_path: Path) -> None:
    image_path = tmp_path / "diagram.png"
    image = Image.new("RGB", (120, 80), color="white")
    image.save(image_path)

    service = make_ingest_service(tmp_path, ocr_repo=FakeOcrVisionRepo())
    result = service.ingest_image(location=str(image_path), image_path=image_path, owner="user")

    assert result.segments[0].visible_text == "diagram label"
    assert result.segments[0].visual_semantics is not None
    assert "two blocks" in result.segments[0].visual_semantics
    assert result.chunks[0].text == "diagram label"


@pytest.mark.skipif(sys.platform != "darwin", reason="default OCR repo uses macOS Vision OCR")
def test_default_image_ingest_uses_real_ocr_text(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    _write_text_image(image_path)

    service = make_ingest_service(tmp_path)
    result = service.ingest_image(location=str(image_path), image_path=image_path, owner="user")

    assert "Quarterly" in result.visible_text
    assert "Revenue" in result.visible_text
    assert result.visible_text != "520x180 RGB image"
    assert result.processing is not None
    assert any("Quarterly" in chunk.text for chunk in result.processing.child_chunks)


def test_image_ingest_falls_back_to_ocr_parser_when_docling_image_parse_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "fallback.png"
    image = Image.new("RGB", (120, 80), color="white")
    image.save(image_path)

    service = make_ingest_service(tmp_path, ocr_repo=FakeOcrVisionRepo())

    def fail_parse(*args: object, **kwargs: object) -> ParsedDocument:
        raise ValueError("Docling failed")

    service.docling_parser.parse = cast(object, fail_parse)  # type: ignore[assignment]

    result = service.ingest_image(location=str(image_path), image_path=image_path, owner="user")

    assert result.visible_text == "diagram label"
    assert result.segments[0].visible_text == "diagram label"
