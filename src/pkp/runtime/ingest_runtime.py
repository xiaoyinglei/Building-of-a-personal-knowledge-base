from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from pkp.service.ingest_service import IngestResult


class GenericIngestProtocol(Protocol):
    def ingest(self, source_type: str, location: str) -> dict[str, int | str]: ...


class TypedIngestProtocol(Protocol):
    def ingest_markdown(self, *, location: str, markdown: str, owner: str) -> IngestResult: ...

    def ingest_plain_text(
        self,
        *,
        location: str,
        text: str,
        owner: str,
        title: str | None = None,
    ) -> IngestResult: ...

    def ingest_pdf(self, *, location: str, pdf_path: Path, owner: str) -> IngestResult: ...

    def ingest_image(self, *, location: str, image_path: Path, owner: str) -> IngestResult: ...

    def ingest_web(self, *, location: str, html: str, owner: str) -> IngestResult: ...


class IngestRuntime:
    def __init__(
        self,
        ingest_service: object,
        *,
        base_path: Path | None = None,
    ) -> None:
        self._ingest_service = ingest_service
        self._base_path = base_path or Path.cwd()

    def ingest_source(self, *, source_type: str, location: str) -> dict[str, int | str]:
        if hasattr(self._ingest_service, "ingest"):
            generic_service = cast(GenericIngestProtocol, self._ingest_service)
            return generic_service.ingest(source_type=source_type, location=location)

        path = self._base_path / location
        typed_service = cast(TypedIngestProtocol, self._ingest_service)
        if source_type == "markdown":
            result = typed_service.ingest_markdown(
                location=location,
                markdown=path.read_text(encoding="utf-8"),
                owner="user",
            )
        elif source_type == "plain_text":
            result = typed_service.ingest_plain_text(
                location=location,
                text=path.read_text(encoding="utf-8"),
                owner="user",
            )
        elif source_type == "pdf":
            result = typed_service.ingest_pdf(
                location=location,
                pdf_path=path,
                owner="user",
            )
        elif source_type == "image":
            result = typed_service.ingest_image(
                location=location,
                image_path=path,
                owner="user",
            )
        elif source_type == "web":
            result = typed_service.ingest_web(
                location=location,
                html=path.read_text(encoding="utf-8"),
                owner="user",
            )
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        return {
            "source_id": result.source.source_id,
            "doc_id": result.document.doc_id,
            "chunk_count": len(result.chunks),
            "source_type": source_type,
            "location": location,
        }
