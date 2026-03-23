from pathlib import Path

from pkp.service.ingest_service import IngestService


def test_web_ingest_extracts_headings_and_article_text(tmp_path: Path) -> None:
    service = IngestService.create_in_memory(tmp_path)

    html = (
        "<html><head><title>Sample Article</title></head>"
        "<body><article>"
        "<h1>Sample Article</h1>"
        "<p>Lead paragraph.</p>"
        "<h2>Details</h2>"
        "<p>Body text.</p>"
        "</article></body></html>"
    )
    result = service.ingest_web(
        location="https://example.com/article",
        html=html,
        owner="user",
    )

    assert result.segments[0].toc_path == ["Sample Article"]
    assert result.segments[1].toc_path == ["Sample Article", "Details"]
    assert "Lead paragraph." in result.chunks[0].text
