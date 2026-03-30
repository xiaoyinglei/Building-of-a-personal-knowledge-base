from __future__ import annotations

from pathlib import Path
from typing import cast

from PIL import Image

from rag.document._parse._util import default_title_from_location, normalize_whitespace, slugify
from rag.schema._types.content import DocumentType, SourceType
from rag.utils._contracts import OcrVisionRepo, ParsedDocument, ParsedElement, ParsedSection


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
        normalized_visible_text = normalize_whitespace(ocr_result.visible_text)
        elements = [
            ParsedElement(
                element_id=f"{slugify(document_title)}-ocr-{index}",
                kind="ocr_region",
                text=normalize_whitespace(region.text),
                toc_path=(document_title,),
                page_no=1,
                bbox=(
                    None
                    if region.bbox is None
                    else cast(
                        tuple[float, float, float, float],
                        tuple(float(value) for value in region.bbox),
                    )
                ),
                metadata={"source_type": "image", "region_index": str(index)},
            )
            for index, region in enumerate(ocr_result.regions)
            if normalize_whitespace(region.text)
        ]

        section = ParsedSection(
            toc_path=(document_title,),
            heading_level=1,
            page_range=None,
            order_index=0,
            text=normalized_visible_text,
            anchor_hint=slugify(document_title),
            metadata={
                "visual_semantics": ocr_result.visual_semantics,
                "visible_text": normalized_visible_text,
                **image_metadata,
            },
        )
        return ParsedDocument(
            title=document_title,
            source_type=SourceType.IMAGE,
            doc_type=DocumentType.IMAGE,
            authors=[owner],
            language="en",
            sections=[section],
            visible_text=normalized_visible_text,
            visual_semantics=ocr_result.visual_semantics,
            elements=elements,
            metadata=image_metadata,
        )
