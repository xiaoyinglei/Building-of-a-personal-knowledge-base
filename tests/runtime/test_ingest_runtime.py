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
