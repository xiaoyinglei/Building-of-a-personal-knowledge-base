from pathlib import Path

from PIL import Image

from pkp.repo.interfaces import OcrResult
from pkp.service.ingest_service import IngestService


class FakeOcrVisionRepo:
    def extract(self, image_path: Path) -> OcrResult:
        return OcrResult(
            visible_text="diagram label",
            visual_semantics="two blocks and a caption",
            regions=[],
        )


def test_image_ingest_stores_visible_text_and_visual_semantics(tmp_path: Path) -> None:
    image_path = tmp_path / "diagram.png"
    image = Image.new("RGB", (120, 80), color="white")
    image.save(image_path)

    service = IngestService.create_in_memory(tmp_path, ocr_repo=FakeOcrVisionRepo())
    result = service.ingest_image(location=str(image_path), image_path=image_path, owner="user")

    assert result.segments[0].visible_text == "diagram label"
    assert "two blocks" in result.segments[0].visual_semantics
    assert result.chunks[0].text == "diagram label"
