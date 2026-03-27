from __future__ import annotations

from pkp.repo.interfaces import ParsedDocument, ParsedSection
from pkp.repo.parse._util import (
    default_title_from_location,
    extract_heading_text,
    normalize_whitespace,
)
from pkp.types.content import DocumentType, SourceType


class MarkdownParserRepo:
    def parse(
        self,
        markdown: str,
        *,
        location: str,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        document_title = title or default_title_from_location(location)
        sections: list[ParsedSection] = []
        heading_stack: list[str] = [document_title]
        current_lines: list[str] = []
        current_order = 0
        current_heading_level: int | None = 1

        def flush_section() -> None:
            nonlocal current_lines, current_order
            if not current_lines and not sections:
                return
            text = normalize_whitespace("\n".join(current_lines))
            sections.append(
                ParsedSection(
                    toc_path=tuple(heading_stack),
                    heading_level=current_heading_level,
                    page_range=None,
                    order_index=current_order,
                    text=text,
                    anchor_hint="-".join(part.lower().replace(" ", "-") for part in heading_stack),
                )
            )
            current_lines = []
            current_order += 1

        for raw_line in markdown.splitlines():
            stripped = raw_line.strip()
            heading = extract_heading_text(stripped)
            if heading is None:
                if stripped:
                    current_lines.append(stripped)
                continue

            flush_section()
            level, heading_text = heading
            heading_stack = heading_stack[: max(level - 1, 0)] + [heading_text]
            current_heading_level = level

        flush_section()

        if not sections:
            sections = [
                ParsedSection(
                    toc_path=(document_title,),
                    heading_level=1,
                    page_range=None,
                    order_index=0,
                    text=normalize_whitespace(markdown),
                    anchor_hint=document_title.lower().replace(" ", "-"),
                )
            ]

        visible_text = normalize_whitespace(" ".join(section.text for section in sections))
        return ParsedDocument(
            title=document_title,
            source_type=SourceType.MARKDOWN,
            doc_type=DocumentType.ARTICLE,
            authors=[owner],
            language="en",
            sections=sections,
            visible_text=visible_text,
            metadata={"location": location, "source_type": "markdown"},
        )
