from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from pkp.repo.vision.ocr_vision_repo import OCRMacVisionRepo


def _write_text_image(path: Path) -> None:
    image = Image.new("RGB", (520, 180), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Quarterly revenue chart", fill="black")
    draw.text((20, 90), "Revenue grew 20 percent", fill="black")
    image.save(path)


@pytest.mark.skipif(sys.platform != "darwin", reason="OCRMac relies on macOS Vision OCR")
def test_ocrmac_repo_extracts_visible_text_and_regions(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    _write_text_image(image_path)

    repo = OCRMacVisionRepo(language_preferences=("en-US",), min_confidence=0.0)
    result = repo.extract(image_path)

    assert "Quarterly revenue chart" in result.visible_text
    assert "Revenue grew 20 percent" in result.visible_text
    assert result.visual_semantics.startswith("Image containing text:")
    assert len(result.regions) >= 2
    assert all(region.bbox is not None for region in result.regions)
