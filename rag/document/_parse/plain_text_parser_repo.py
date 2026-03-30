from __future__ import annotations

from rag.document._parse._util import default_title_from_location, normalize_whitespace, slugify
from rag.schema._types.content import DocumentType, SourceType
from rag.utils._contracts import ParsedDocument, ParsedSection


class PlainTextParserRepo:
    def parse(
        self,
        text: str,
        *,
        location: str,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        document_title = title or default_title_from_location(location)
        normalized = normalize_whitespace(text)
        section = ParsedSection(
            toc_path=(document_title,),
            heading_level=1,
            page_range=None,
            order_index=0,
            text=normalized,
            anchor_hint=slugify(document_title),
        )
        return ParsedDocument(
            title=document_title,
            source_type=SourceType.PLAIN_TEXT,
            doc_type=DocumentType.NOTE,
            authors=[owner],
            language="en",
            sections=[section],
            visible_text=normalized,
            metadata={"location": location, "source_type": "plain_text"},
        )
