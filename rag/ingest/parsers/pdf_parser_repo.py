from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore[import-untyped]

from rag.ingest.parsers.util import default_title_from_location, normalize_whitespace
from rag.schema.core import DocumentType, ParsedDocument, ParsedSection, SourceType


class PDFParserRepo:
    def parse(
        self,
        pdf_path: Path,
        *,
        location: str,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        document_title = title or default_title_from_location(location)
        sections: list[ParsedSection] = []
        visible_parts: list[str] = []

        with fitz.open(pdf_path) as document:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = normalize_whitespace(page.get_text("text"))
                if text:
                    visible_parts.append(text)
                sections.append(
                    ParsedSection(
                        toc_path=(document_title, f"Page {page_index + 1}"),
                        heading_level=None,
                        page_range=(page_index + 1, page_index + 1),
                        order_index=page_index,
                        text=text,
                        anchor_hint=f"page-{page_index + 1}",
                    )
                )

        return ParsedDocument(
            title=document_title,
            source_type=SourceType.PDF,
            doc_type=DocumentType.REPORT,
            authors=[owner],
            language="en",
            sections=sections,
            visible_text=normalize_whitespace(" ".join(visible_parts)),
            metadata={"location": location, "source_type": "pdf"},
        )
