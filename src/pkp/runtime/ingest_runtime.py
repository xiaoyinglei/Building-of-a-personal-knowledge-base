from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from pkp.service.ingest_service import IngestResult
from pkp.types import AccessPolicy


class GenericIngestProtocol(Protocol):
    def ingest(self, source_type: str, location: str) -> dict[str, int | str]: ...


class TypedIngestProtocol(Protocol):
    def ingest_markdown(
        self,
        *,
        location: str,
        markdown: str,
        owner: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> IngestResult: ...

    def ingest_plain_text(
        self,
        *,
        location: str,
        text: str,
        owner: str,
        title: str | None = None,
        source_type: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> IngestResult: ...

    def ingest_pdf(
        self,
        *,
        location: str,
        pdf_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
    ) -> IngestResult: ...

    def ingest_image(
        self,
        *,
        location: str,
        image_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
    ) -> IngestResult: ...

    def ingest_docx(
        self,
        *,
        location: str,
        docx_path: Path,
        owner: str,
        access_policy: AccessPolicy | None = None,
        title: str | None = None,
    ) -> IngestResult: ...

    def ingest_web(
        self,
        *,
        location: str,
        html: str,
        owner: str,
        title: str | None = None,
        source_type: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> IngestResult: ...

    def ingest_web_url(
        self,
        *,
        location: str,
        owner: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> IngestResult: ...

    def ingest_file(
        self,
        *,
        location: str,
        file_path: Path,
        owner: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> IngestResult: ...


class IngestRuntime:
    def __init__(
        self,
        ingest_service: object,
        *,
        base_path: Path | None = None,
    ) -> None:
        self._ingest_service = ingest_service
        self._base_path = base_path or Path.cwd()

    def ingest_source(
        self,
        *,
        source_type: str,
        location: str | None = None,
        content: str | None = None,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> dict[str, int | str]:
        resolved_location = self._resolve_location(
            source_type=source_type,
            location=location,
            content=content,
        )
        if content is None and title is None and access_policy is None and hasattr(self._ingest_service, "ingest"):
            generic_service = cast(GenericIngestProtocol, self._ingest_service)
            return generic_service.ingest(source_type=source_type, location=resolved_location)

        typed_service = cast(TypedIngestProtocol, self._ingest_service)
        if source_type == "markdown":
            result = typed_service.ingest_markdown(
                location=resolved_location,
                markdown=self._resolve_inline_or_file_content(location=resolved_location, content=content),
                owner="user",
                title=title,
                access_policy=access_policy,
            )
        elif source_type in {"plain_text", "pasted_text"}:
            result = typed_service.ingest_plain_text(
                location=resolved_location,
                text=self._resolve_inline_or_file_content(location=resolved_location, content=content),
                owner="user",
                title=title,
                source_type=source_type,
                access_policy=access_policy,
            )
        elif source_type == "pdf":
            if content is not None:
                raise ValueError("Inline content is not supported for pdf sources")
            path = self._base_path / resolved_location
            result = typed_service.ingest_pdf(
                location=resolved_location,
                pdf_path=path,
                owner="user",
                access_policy=access_policy,
            )
        elif source_type == "docx":
            if content is not None:
                raise ValueError("Inline content is not supported for docx sources")
            path = self._base_path / resolved_location
            result = typed_service.ingest_docx(
                location=resolved_location,
                docx_path=path,
                owner="user",
                title=title,
                access_policy=access_policy,
            )
        elif source_type == "image":
            if content is not None:
                raise ValueError("Inline content is not supported for image sources")
            path = self._base_path / resolved_location
            result = typed_service.ingest_image(
                location=resolved_location,
                image_path=path,
                owner="user",
                access_policy=access_policy,
            )
        elif source_type in {"web", "browser_clip"}:
            if content is not None:
                result = typed_service.ingest_web(
                    location=resolved_location,
                    html=content,
                    owner="user",
                    title=title,
                    source_type=source_type,
                    access_policy=access_policy,
                )
            elif source_type == "web" and self._is_remote_web_location(resolved_location):
                if title is None:
                    result = typed_service.ingest_web_url(
                        location=resolved_location,
                        owner="user",
                        access_policy=access_policy,
                    )
                else:
                    result = typed_service.ingest_web_url(
                        location=resolved_location,
                        owner="user",
                        title=title,
                        access_policy=access_policy,
                    )
            elif source_type == "browser_clip" and self._is_remote_web_location(resolved_location):
                raise ValueError("browser_clip requires inline content or a local HTML file")
            else:
                result = typed_service.ingest_web(
                    location=resolved_location,
                    html=self._resolve_inline_or_file_content(location=resolved_location, content=content),
                    owner="user",
                    title=title,
                    source_type=source_type,
                    access_policy=access_policy,
                )
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        return {
            "source_id": result.source.source_id,
            "doc_id": result.document.doc_id,
            "chunk_count": len(result.chunks),
            "source_type": str(getattr(result.source, "source_type", source_type)),
            "location": resolved_location,
        }

    def process_file(
        self,
        *,
        location: str,
        title: str | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> dict[str, object]:
        typed_service = cast(TypedIngestProtocol, self._ingest_service)
        path = self._base_path / location
        result = typed_service.ingest_file(
            location=location,
            file_path=path,
            owner="user",
            title=title,
            access_policy=access_policy,
        )
        processing = getattr(result, "processing", None)
        return {
            "source_id": result.source.source_id,
            "doc_id": result.document.doc_id,
            "source_type": str(getattr(result.source, "source_type", "")),
            "location": location,
            "processing": None if processing is None else processing.model_dump(mode="json"),
        }

    def repair_indexes(self) -> dict[str, int]:
        repair = getattr(self._ingest_service, "repair_indexes", None)
        if not callable(repair):
            raise ValueError("ingest service does not support index repair")
        result = repair()
        if not isinstance(result, dict):
            raise RuntimeError("repair_indexes must return a result dictionary")
        return result

    @staticmethod
    def _is_remote_web_location(location: str) -> bool:
        parsed = urlparse(location)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _resolve_inline_or_file_content(self, *, location: str, content: str | None) -> str:
        if content is not None:
            return content
        path = self._base_path / location
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{location} is not UTF-8 text; choose the correct source type before uploading.") from exc

    @staticmethod
    def _resolve_location(*, source_type: str, location: str | None, content: str | None) -> str:
        if location is not None:
            return location
        if content is None:
            raise ValueError("location is required when inline content is not provided")
        digest = sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"inline://{source_type}/{digest}"
