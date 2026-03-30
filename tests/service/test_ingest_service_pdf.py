from pathlib import Path

import fitz  # type: ignore[import-untyped]

from rag.ingest.ingest import IngestService


def test_pdf_ingest_produces_page_anchored_segments_and_chunks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Page one content")
    page = document.new_page()
    page.insert_text((72, 72), "Page two content")
    document.save(pdf_path)
    document.close()

    service = IngestService.create_in_memory(tmp_path)
    result = service.ingest_pdf(location=str(pdf_path), pdf_path=pdf_path, owner="user")

    assert result.segments[0].page_range == (1, 1)
    assert result.segments[0].anchor is not None
    assert result.segments[0].anchor.startswith("sample.pdf#page-1")
    assert "Page one content" in result.chunks[0].text
