from dataclasses import dataclass

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

    def ingest_web_url(self, *, location: str, owner: str) -> object:
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
