from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pkp.core.options import QueryOptions
from pkp.core.pipelines.ingest_pipeline import IngestPipeline, IngestPipelineResult, IngestRequest
from pkp.core.storage_config import StorageConfig
from pkp.repo.parse.docling_parser_repo import DoclingParserRepo
from pkp.repo.parse.image_parser_repo import ImageParserRepo
from pkp.repo.parse.markdown_parser_repo import MarkdownParserRepo
from pkp.repo.parse.pdf_parser_repo import PDFParserRepo
from pkp.repo.parse.plain_text_parser_repo import PlainTextParserRepo
from pkp.repo.parse.web_fetch_repo import WebFetchRepo as HttpWebFetchRepo
from pkp.repo.parse.web_parser_repo import WebParserRepo
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.storage.file_object_store import FileObjectStore
from pkp.repo.vision.ocr_vision_repo import create_default_ocr_repo
from pkp.service.chunking_service import ChunkingService
from pkp.service.document_processing_service import DocumentProcessingService
from pkp.service.policy_resolution_service import PolicyResolutionService
from pkp.service.toc_service import TOCService
from pkp.stores import StorageBundle


@dataclass(slots=True)
class RAGCore:
    storage: StorageConfig
    stores: StorageBundle = field(init=False)
    ingest_pipeline: IngestPipeline = field(init=False)
    _fts_repo: SQLiteFTSRepo = field(init=False, repr=False)
    _object_store: FileObjectStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.stores = self.storage.build()
        ocr_repo = create_default_ocr_repo()
        self._fts_repo = SQLiteFTSRepo(self.stores.root / "fts.sqlite3")
        self._object_store = FileObjectStore(self.stores.root / "objects")
        self.ingest_pipeline = IngestPipeline(
            documents=self.stores.documents,
            chunks=self.stores.chunks,
            vectors=self.stores.vectors,
            graph=self.stores.graph,
            status=self.stores.status,
            cache=self.stores.cache,
            fts_repo=self._fts_repo,
            object_store=self._object_store,
            markdown_parser=MarkdownParserRepo(),
            pdf_parser=PDFParserRepo(),
            plain_text_parser=PlainTextParserRepo(),
            image_parser=ImageParserRepo(ocr_repo),
            web_parser=WebParserRepo(),
            web_fetch_repo=HttpWebFetchRepo(),
            docling_parser=DoclingParserRepo(ocr_repo),
            policy_resolution_service=PolicyResolutionService(),
            toc_service=TOCService(),
            chunking_service=ChunkingService(),
            document_processing_service=DocumentProcessingService(toc_service=TOCService()),
        )

    def insert(self, request: IngestRequest | None = None, /, **kwargs: Any) -> IngestPipelineResult:
        if request is None:
            normalized_kwargs = {"owner": "user", **kwargs}
            if "file_path" in normalized_kwargs and normalized_kwargs["file_path"] is not None:
                normalized_kwargs["file_path"] = Path(normalized_kwargs["file_path"])
            request = IngestRequest(**normalized_kwargs)
        return self.ingest_pipeline.run(request)

    def query(self, *args: Any, options: QueryOptions | None = None, **kwargs: Any) -> None:
        del options
        raise NotImplementedError("query pipeline is not implemented yet")

    def delete(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("delete pipeline is not implemented yet")

    def rebuild(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("rebuild pipeline is not implemented yet")
