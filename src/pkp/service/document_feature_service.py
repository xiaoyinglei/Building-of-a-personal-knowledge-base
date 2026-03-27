from __future__ import annotations

from pkp.repo.interfaces import ParsedDocument
from pkp.types.processing import DocumentFeatures


class DocumentFeatureService:
    def analyze(self, parsed: ParsedDocument) -> DocumentFeatures:
        word_count = len(parsed.visible_text.split())
        section_count = len([section for section in parsed.sections if section.text.strip()])
        heading_count = len([element for element in parsed.elements if element.kind == "section_header"])
        table_count = len([element for element in parsed.elements if element.kind == "table"])
        figure_count = len([element for element in parsed.elements if element.kind == "figure"])
        caption_count = len([element for element in parsed.elements if element.kind == "caption"])
        ocr_region_count = len([element for element in parsed.elements if element.kind == "ocr_region"])
        structure_depth = max((len(section.toc_path) - 1 for section in parsed.sections), default=1)
        avg_section_words = word_count / max(section_count, 1)
        heading_quality_score = self._heading_quality_score(
            section_count=section_count,
            heading_count=heading_count,
            structure_depth=structure_depth,
        )
        return DocumentFeatures(
            source_type=parsed.source_type,
            section_count=section_count,
            word_count=word_count,
            heading_count=heading_count,
            heading_quality_score=heading_quality_score,
            table_count=table_count,
            figure_count=figure_count,
            caption_count=caption_count,
            ocr_region_count=ocr_region_count,
            avg_section_words=avg_section_words,
            structure_depth=structure_depth,
            has_dense_structure=heading_count >= 2 and structure_depth >= 2,
            metadata={
                "page_count": str(parsed.page_count or 0),
                "has_visual_semantics": str(bool(parsed.visual_semantics)),
            },
        )

    @staticmethod
    def _heading_quality_score(
        *,
        section_count: int,
        heading_count: int,
        structure_depth: int,
    ) -> float:
        if heading_count <= 0 or section_count <= 0:
            return 0.0
        density = min(heading_count / section_count, 1.0)
        depth_score = min(structure_depth / 4.0, 1.0)
        return round((density * 0.7) + (depth_score * 0.3), 4)
