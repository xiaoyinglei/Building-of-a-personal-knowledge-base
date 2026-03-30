from dataclasses import dataclass
from pathlib import Path

from pkp.ingest.ingest import IngestService
from pkp.types.content import SourceType


@dataclass
class FakeWebFetchRepo:
    calls: list[str]
    html: str

    def fetch(self, location: str) -> str:
        self.calls.append(location)
        return self.html


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


def test_web_ingest_url_fetches_html_before_parsing(tmp_path: Path) -> None:
    fetch_repo = FakeWebFetchRepo(
        calls=[],
        html=(
            "<html><head><title>Remote Article</title></head>"
            "<body><article><h1>Remote Article</h1><p>Remote body.</p></article></body></html>"
        ),
    )
    service = IngestService.create_in_memory(tmp_path, web_fetch_repo=fetch_repo)

    result = service.ingest_web_url(
        location="https://example.com/article",
        owner="user",
    )

    assert fetch_repo.calls == ["https://example.com/article"]
    assert result.source.location == "https://example.com/article"
    assert result.document.title == "Remote Article"


def test_browser_clip_ingest_preserves_source_type_and_title(tmp_path: Path) -> None:
    service = IngestService.create_in_memory(tmp_path)

    html = (
        "<html><head><title>Original Page</title></head>"
        "<body><article><h1>Clipped Heading</h1><p>Captured body.</p></article></body></html>"
    )
    result = service.ingest_web(
        location="https://example.com/clipped",
        html=html,
        owner="user",
        title="Saved Clip",
        source_type=SourceType.BROWSER_CLIP,
    )

    assert result.source.source_type is SourceType.BROWSER_CLIP
    assert result.document.title == "Saved Clip"
    assert result.segments[0].toc_path == ["Clipped Heading"]
