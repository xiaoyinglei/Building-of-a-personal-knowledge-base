from __future__ import annotations

from pathlib import Path

from PIL import Image

from pkp.repo.interfaces import OcrVisionRepo, ParsedDocument, ParsedSection
from pkp.repo.parse._util import default_title_from_location, slugify
from pkp.types.content import DocumentType


class ImageParserRepo:
    def __init__(self, ocr_repo: OcrVisionRepo) -> None:
        self._ocr_repo = ocr_repo

    def parse(
        self,
        image_path: Path,
        *,
        location: str,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        document_title = title or default_title_from_location(location)
        ocr_result = self._ocr_repo.extract(image_path)
        with Image.open(image_path) as image:
            image_metadata = {
                "image_width": str(image.width),
                "image_height": str(image.height),
                "image_mode": image.mode,
                "source_type": "image",
                "location": location,
            }

        section = ParsedSection(
            toc_path=(document_title,),
            heading_level=1,
            page_range=None,
            order_index=0,
            text=ocr_result.visible_text.strip(),
            anchor_hint=slugify(document_title),
            metadata={
                "visual_semantics": ocr_result.visual_semantics,
                "visible_text": ocr_result.visible_text,
                **image_metadata,
            },
        )
        return ParsedDocument(
            title=document_title,
            doc_type=DocumentType.IMAGE,
            authors=[owner],
            language="en",
            sections=[section],
            visible_text=ocr_result.visible_text.strip(),
            visual_semantics=ocr_result.visual_semantics,
            metadata=image_metadata,
        )
