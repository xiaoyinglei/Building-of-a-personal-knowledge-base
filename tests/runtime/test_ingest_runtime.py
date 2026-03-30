from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pytest

from pkp.interfaces._runtime.ingest_runtime import IngestRuntime
from pkp.schema._types import (
    AccessPolicy,
    ExecutionLocation,
    ExternalRetrievalPolicy,
    Residency,
    RuntimeMode,
)


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

    def ingest_web_url(
        self,
        *,
        location: str,
        owner: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> object:
        del title, access_policy
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


@dataclass
class FakeFilePipelineService:
    calls: list[dict[str, object | None]]

    def ingest_file(
        self,
        *,
        location: str,
        file_path: Path,
        owner: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> object:
        self.calls.append(
            {
                "location": location,
                "file_path": file_path,
                "owner": owner,
                "title": title,
                "access_policy": access_policy,
            }
        )
        processing = type(
            "Processing",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "analysis": {"source_type": "docx"},
                    "routing": {"selected_strategy": "hierarchical"},
                    "stats": {"total_chunks": 6},
                },
            },
        )()
        return type(
            "Result",
            (),
            {
                "source": type("Source", (), {"source_id": "src-file", "source_type": "docx"})(),
                "document": type("Document", (), {"doc_id": "doc-file"})(),
                "processing": processing,
            },
        )()


def test_ingest_runtime_routes_web_urls_without_local_file_reads(tmp_path) -> None:
    service = FakeWebIngestService(calls=[])
    runtime = IngestRuntime(ingest_service=service, base_path=tmp_path)

    result = runtime.ingest_source(source_type="web", location="https://example.com/article")

    assert service.calls == ["https://example.com/article"]
    assert result["source_id"] == "src-web"


def test_ingest_runtime_process_file_uses_unified_pipeline_entry(tmp_path: Path) -> None:
    service = FakeFilePipelineService(calls=[])
    runtime = IngestRuntime(ingest_service=service, base_path=tmp_path)
    file_path = tmp_path / "report.docx"
    file_path.write_bytes(b"docx")

    result = runtime.process_file(location="report.docx")

    assert service.calls == [
        {
            "location": "report.docx",
            "file_path": tmp_path / "report.docx",
            "owner": "user",
            "title": None,
            "access_policy": None,
        }
    ]
    assert result["source_id"] == "src-file"
    assert result["processing"]["routing"]["selected_strategy"] == "hierarchical"


@dataclass
class FakeTypedIngestService:
    calls: list[dict[str, object | None]]

    def ingest_markdown(
        self,
        *,
        location: str,
        markdown: str,
        owner: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> object:
        self.calls.append(
            {
                "method": "markdown",
                "location": location,
                "content": markdown,
                "owner": owner,
                "title": title,
                "source_type": None,
                "access_policy": access_policy,
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
        access_policy: AccessPolicy | None = None,
    ) -> object:
        self.calls.append(
            {
                "method": "plain_text",
                "location": location,
                "content": text,
                "owner": owner,
                "title": title,
                "source_type": source_type,
                "access_policy": access_policy,
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
        access_policy: AccessPolicy | None = None,
    ) -> object:
        self.calls.append(
            {
                "method": "web",
                "location": location,
                "content": html,
                "owner": owner,
                "title": title,
                "source_type": source_type,
                "access_policy": access_policy,
            }
        )
        return _result("src-web-inline", "doc-web-inline")

    def ingest_web_url(
        self,
        *,
        location: str,
        owner: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> object:
        self.calls.append(
            {
                "method": "web_url",
                "location": location,
                "content": None,
                "owner": owner,
                "title": title,
                "source_type": None,
                "access_policy": access_policy,
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
            "access_policy": None,
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
            "access_policy": None,
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
            "access_policy": None,
        }
    ]
    assert result["source_id"] == "src-web-inline"


def test_ingest_runtime_passes_access_policy_into_typed_ingest_service(tmp_path: Path) -> None:
    service = FakeTypedIngestService(calls=[])
    runtime = IngestRuntime(ingest_service=service, base_path=tmp_path)
    policy = AccessPolicy(
        residency=Residency.LOCAL_REQUIRED,
        external_retrieval=ExternalRetrievalPolicy.DENY,
        allowed_runtimes=frozenset({RuntimeMode.FAST}),
        allowed_locations=frozenset({ExecutionLocation.LOCAL}),
        sensitivity_tags=frozenset({"private"}),
    )

    runtime.ingest_source(
        source_type="pasted_text",
        content="Sensitive note",
        title="Sensitive",
        access_policy=policy,
    )

    assert service.calls[0]["access_policy"] == policy


def test_ingest_runtime_rejects_non_utf8_plain_text_files(tmp_path: Path) -> None:
    service = FakeTypedIngestService(calls=[])
    runtime = IngestRuntime(ingest_service=service, base_path=tmp_path)
    binary_path = tmp_path / "notes.bin"
    binary_path.write_bytes(b"\xff\xfe\x00\x87")

    with pytest.raises(ValueError, match="not UTF-8 text"):
        runtime.ingest_source(source_type="plain_text", location="notes.bin")
