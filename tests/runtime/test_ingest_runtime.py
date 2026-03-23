from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pkp.runtime.ingest_runtime import IngestRuntime


@dataclass
class FakeIngestService:
    calls: list[tuple[str, str]]

    def ingest(self, source_type: str, location: str) -> dict[str, int | str]:
        self.calls.append((source_type, location))
        return {"source_id": "src-1", "chunk_count": 4}


def test_ingest_runtime_delegates_to_ingest_service() -> None:
    service = FakeIngestService(calls=[])
    runtime = IngestRuntime(ingest_service=service)

    result = runtime.ingest_source(source_type="markdown", location="data/example.md")

    assert service.calls == [("markdown", "data/example.md")]
    assert result["chunk_count"] == 4


@dataclass
class FakeWebIngestService:
    calls: list[str]

    def ingest_web_url(self, *, location: str, owner: str, title: str | None = None) -> object:
        del title
        self.calls.append(location)
        return type(
            "Result",
            (),
            {
                "source": type("Source", (), {"source_id": "src-web"})(),
                "document": type("Document", (), {"doc_id": "doc-web"})(),
                "chunks": [object()],
            },
        )()


def test_ingest_runtime_routes_web_urls_without_local_file_reads(tmp_path) -> None:
    service = FakeWebIngestService(calls=[])
    runtime = IngestRuntime(ingest_service=service, base_path=tmp_path)

    result = runtime.ingest_source(source_type="web", location="https://example.com/article")

    assert service.calls == ["https://example.com/article"]
    assert result["source_id"] == "src-web"


@dataclass
class FakeTypedIngestService:
    calls: list[dict[str, str | None]]

    def ingest_markdown(
        self,
        *,
        location: str,
        markdown: str,
        owner: str,
        title: str | None = None,
    ) -> object:
        self.calls.append(
            {
                "method": "markdown",
                "location": location,
                "content": markdown,
                "owner": owner,
                "title": title,
                "source_type": None,
            }
        )
        return _result("src-markdown", "doc-markdown")

    def ingest_plain_text(
        self,
        *,
        location: str,
        text: str,
        owner: str,
        title: str | None = None,
        source_type: str | None = None,
    ) -> object:
        self.calls.append(
            {
                "method": "plain_text",
                "location": location,
                "content": text,
                "owner": owner,
                "title": title,
                "source_type": source_type,
            }
        )
        return _result("src-plain", "doc-plain")

    def ingest_web(
        self,
        *,
        location: str,
        html: str,
        owner: str,
        title: str | None = None,
        source_type: str | None = None,
    ) -> object:
        self.calls.append(
            {
                "method": "web",
                "location": location,
                "content": html,
                "owner": owner,
                "title": title,
                "source_type": source_type,
            }
        )
        return _result("src-web-inline", "doc-web-inline")

    def ingest_web_url(self, *, location: str, owner: str, title: str | None = None) -> object:
        self.calls.append(
            {
                "method": "web_url",
                "location": location,
                "content": None,
                "owner": owner,
                "title": title,
                "source_type": None,
            }
        )
        return _result("src-web-url", "doc-web-url")


def _result(source_id: str, doc_id: str) -> object:
    return type(
        "Result",
        (),
        {
            "source": type("Source", (), {"source_id": source_id})(),
            "document": type("Document", (), {"doc_id": doc_id})(),
            "chunks": [object(), object()],
        },
    )()


def test_ingest_runtime_uses_inline_markdown_content_without_file_reads(tmp_path: Path) -> None:
    service = FakeTypedIngestService(calls=[])
    runtime = IngestRuntime(ingest_service=service, base_path=tmp_path)

    result = runtime.ingest_source(
        source_type="markdown",
        location="virtual://note.md",
        content="# Inline\n\nBody",
        title="Inline note",
    )

    assert service.calls == [
        {
            "method": "markdown",
            "location": "virtual://note.md",
            "content": "# Inline\n\nBody",
            "owner": "user",
            "title": "Inline note",
            "source_type": None,
        }
    ]
    assert result["source_id"] == "src-markdown"


def test_ingest_runtime_routes_pasted_text_into_plain_text_ingest(tmp_path: Path) -> None:
    service = FakeTypedIngestService(calls=[])
    runtime = IngestRuntime(ingest_service=service, base_path=tmp_path)
    content = "Captured text"

    runtime.ingest_source(
        source_type="pasted_text",
        content=content,
        title="Capture",
    )

    assert service.calls == [
        {
            "method": "plain_text",
            "location": f"inline://pasted_text/{sha256(content.encode('utf-8')).hexdigest()[:16]}",
            "content": content,
            "owner": "user",
            "title": "Capture",
            "source_type": "pasted_text",
        }
    ]


def test_ingest_runtime_routes_browser_clip_html_without_remote_fetch(tmp_path: Path) -> None:
    service = FakeTypedIngestService(calls=[])
    runtime = IngestRuntime(ingest_service=service, base_path=tmp_path)
    content = "<html><body><article><h1>Clip</h1></article></body></html>"

    result = runtime.ingest_source(
        source_type="browser_clip",
        content=content,
        title="Clip title",
    )

    assert service.calls == [
        {
            "method": "web",
            "location": f"inline://browser_clip/{sha256(content.encode('utf-8')).hexdigest()[:16]}",
            "content": content,
            "owner": "user",
            "title": "Clip title",
            "source_type": "browser_clip",
        }
    ]
    assert result["source_id"] == "src-web-inline"
