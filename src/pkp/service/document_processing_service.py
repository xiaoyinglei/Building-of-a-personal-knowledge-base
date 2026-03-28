from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

from pkp.algorithms.chunking.multimodal_chunk_router import build_special_chunks
from pkp.algorithms.chunking.structured_chunker import build_cached_hybrid_chunker, merge_adjacent_seeds
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import PictureItem, TableItem

from pkp.repo.interfaces import ParsedDocument
from pkp.repo.parse._util import normalize_whitespace, slugify
from pkp.service.chunk_postprocessing_service import ChunkPostprocessingService
from pkp.service.chunk_routing_service import ChunkRoutingService
from pkp.service.document_feature_service import DocumentFeatureService
from pkp.service.toc_service import TOCService
from pkp.types.access import AccessPolicy
from pkp.types.content import Chunk, ChunkRole, Document, Segment, Source
from pkp.types.processing import (
    ChunkingStrategy,
    ChunkStatistics,
    DocumentProcessingPackage,
)
from pkp.types.text import text_unit_count


@dataclass(frozen=True)
class PreparedDocumentProcessing:
    segments: list[Segment]
    stored_chunks: list[Chunk]
    indexed_chunks: list[Chunk]
    package: DocumentProcessingPackage


@dataclass(frozen=True)
class ChunkSeed:
    text: str
    toc_path: list[str]
    page_numbers: list[int]


class DocumentProcessingService:
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s*")
    _DEFAULT_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    _DEFAULT_TOKENIZER_MAX_TOKENS = 512

    def __init__(
        self,
        *,
        toc_service: TOCService,
        feature_service: DocumentFeatureService | None = None,
        routing_service: ChunkRoutingService | None = None,
        postprocessing_service: ChunkPostprocessingService | None = None,
    ) -> None:
        self._toc_service = toc_service
        self._feature_service = feature_service or DocumentFeatureService()
        self._routing_service = routing_service or ChunkRoutingService()
        self._postprocessing_service = postprocessing_service or ChunkPostprocessingService()
        self._hierarchical_chunker = HierarchicalChunker()
        self._hybrid_chunker = build_cached_hybrid_chunker(
            tokenizer_cls=HuggingFaceTokenizer,
            hybrid_chunker_cls=HybridChunker,
            model_name=self._DEFAULT_TOKENIZER_MODEL,
            max_tokens=self._DEFAULT_TOKENIZER_MAX_TOKENS,
        )

    def build(
        self,
        *,
        location: str,
        source: Source,
        document: Document,
        parsed: ParsedDocument,
        access_policy: AccessPolicy,
    ) -> PreparedDocumentProcessing:
        analysis = self._feature_service.analyze(parsed)
        routing = self._routing_service.route(analysis)
        segments, parent_chunks, child_chunks = self._build_primary_chunks(
            location=location,
            document=document,
            parsed=parsed,
            access_policy=access_policy,
            strategy=routing.selected_strategy,
            local_refine=routing.local_refine,
        )
        special_chunks = self._build_special_chunks(
            location=location,
            document=document,
            parsed=parsed,
            access_policy=access_policy,
            segments=segments,
        )
        processed = self._postprocessing_service.postprocess(
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            special_chunks=special_chunks,
        )
        metadata_summary: dict[str, str | int | float | bool] = {
            "schema_version": "chunk-pipeline/v1",
            "source_type": parsed.source_type.value,
            "selected_strategy": routing.selected_strategy.value,
            "special_chunk_mode": routing.special_chunk_mode,
            "local_refine": routing.local_refine,
            "fallback": routing.fallback,
        }
        package = DocumentProcessingPackage(
            source=source,
            document=document,
            analysis=analysis,
            routing=routing,
            parent_chunks=processed.parent_chunks,
            child_chunks=processed.child_chunks,
            special_chunks=processed.special_chunks,
            metadata_summary=metadata_summary,
            stats=ChunkStatistics(
                parent_chunk_count=len(processed.parent_chunks),
                child_chunk_count=len(processed.child_chunks),
                special_chunk_count=len(processed.special_chunks),
                total_chunks=len(processed.parent_chunks) + len(processed.child_chunks) + len(processed.special_chunks),
                deduplicated_chunks=processed.deduplicated_chunks,
                merged_chunks=processed.merged_chunks,
            ),
        )
        return PreparedDocumentProcessing(
            segments=segments,
            stored_chunks=[*processed.parent_chunks, *processed.child_chunks, *processed.special_chunks],
            indexed_chunks=[*processed.child_chunks, *processed.special_chunks],
            package=package,
        )

    def _build_primary_chunks(
        self,
        *,
        location: str,
        document: Document,
        parsed: ParsedDocument,
        access_policy: AccessPolicy,
        strategy: ChunkingStrategy,
        local_refine: bool,
    ) -> tuple[list[Segment], list[Chunk], list[Chunk]]:
        if strategy is ChunkingStrategy.IMAGE:
            return self._build_image_chunks(
                location=location,
                document=document,
                parsed=parsed,
                access_policy=access_policy,
            )

        doc_model = cast(Any, parsed.doc_model)
        if doc_model is None:
            raise ValueError("Docling document is required for hierarchical or hybrid chunking")

        raw_chunks: list[Any] = list(
            self._hierarchical_chunker.chunk(doc_model)
            if strategy is ChunkingStrategy.HIERARCHICAL
            else self._hybrid_chunker.chunk(doc_model)
        )
        filtered_chunks = [
            chunk
            for chunk in raw_chunks
            if normalize_whitespace(chunk.text)
            and not any(isinstance(item, (TableItem, PictureItem)) for item in chunk.meta.doc_items)
        ]
        if not filtered_chunks:
            return self._build_fallback_section_chunks(
                location=location,
                document=document,
                parsed=parsed,
                access_policy=access_policy,
                local_refine=local_refine,
            )
        if parsed.source_type.value == "pdf" and any(
            len(set(self._page_numbers(chunk))) > 1 for chunk in filtered_chunks
        ):
            return self._build_fallback_section_chunks(
                location=location,
                document=document,
                parsed=parsed,
                access_policy=access_policy,
                local_refine=local_refine,
            )
        seeds = merge_adjacent_seeds(
            [
                ChunkSeed(
                    text=chunk.text,
                    toc_path=self._toc_path_from_chunk(parsed=parsed, chunk=chunk),
                    page_numbers=self._page_numbers(chunk),
                )
                for chunk in filtered_chunks
            ],
            source_type=parsed.source_type.value,
        )

        segments: list[Segment] = []
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []
        path_to_segment_id: dict[tuple[str, ...], str] = {}
        child_order = 0
        for index, seed in enumerate(seeds):
            toc_path = seed.toc_path
            parent_path = tuple(toc_path[:-1])
            page_numbers = seed.page_numbers
            page_range = (min(page_numbers), max(page_numbers)) if page_numbers else None
            anchor = self._toc_service.stable_anchor(
                location,
                toc_path,
                index,
                page_range=page_range,
                anchor_hint=slugify("-".join(toc_path)),
            )
            segment = Segment(
                segment_id=self._deterministic_id(document.doc_id, anchor, "segment"),
                doc_id=document.doc_id,
                parent_segment_id=path_to_segment_id.get(parent_path),
                toc_path=toc_path,
                heading_level=max(len(toc_path) - 1, 1),
                page_range=page_range,
                order_index=index,
                anchor=anchor,
                visible_text=normalize_whitespace(seed.text),
                visual_semantics=parsed.visual_semantics,
                metadata={
                    "chunk_strategy": strategy.value,
                    "location": location,
                },
            )
            segments.append(segment)
            path_to_segment_id[tuple(toc_path)] = segment.segment_id
            parent_chunk = self._make_chunk(
                location=location,
                document=document,
                segment=segment,
                text=seed.text,
                access_policy=access_policy,
                chunk_role=ChunkRole.PARENT,
                order_index=index,
                parent_chunk_id=None,
                special_chunk_type=None,
            )
            parent_chunks.append(parent_chunk)
            refined = self._refine_parent_chunk(
                parent_chunk=parent_chunk,
                access_policy=access_policy,
                local_refine=local_refine,
                starting_order=child_order,
            )
            child_chunks.extend(refined)
            child_order += len(refined)
        return segments, parent_chunks, child_chunks

    def _build_fallback_section_chunks(
        self,
        *,
        location: str,
        document: Document,
        parsed: ParsedDocument,
        access_policy: AccessPolicy,
        local_refine: bool,
    ) -> tuple[list[Segment], list[Chunk], list[Chunk]]:
        segments: list[Segment] = []
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []
        for index, section in enumerate(parsed.sections):
            anchor = self._toc_service.stable_anchor(
                location,
                section.toc_path,
                index,
                page_range=section.page_range,
                anchor_hint=section.anchor_hint,
            )
            segment = Segment(
                segment_id=self._deterministic_id(document.doc_id, anchor, "segment"),
                doc_id=document.doc_id,
                parent_segment_id=None,
                toc_path=list(section.toc_path),
                heading_level=section.heading_level,
                page_range=section.page_range,
                order_index=index,
                anchor=anchor,
                visible_text=section.text,
                visual_semantics=parsed.visual_semantics,
                metadata=section.metadata,
            )
            segments.append(segment)
            parent_chunk = self._make_chunk(
                location=location,
                document=document,
                segment=segment,
                text=section.text,
                access_policy=access_policy,
                chunk_role=ChunkRole.PARENT,
                order_index=index,
                parent_chunk_id=None,
                special_chunk_type=None,
            )
            parent_chunks.append(parent_chunk)
            child_chunks.extend(
                self._refine_parent_chunk(
                    parent_chunk=parent_chunk,
                    access_policy=access_policy,
                    local_refine=local_refine,
                    starting_order=len(child_chunks),
                )
            )
        return segments, parent_chunks, child_chunks

    def _build_image_chunks(
        self,
        *,
        location: str,
        document: Document,
        parsed: ParsedDocument,
        access_policy: AccessPolicy,
    ) -> tuple[list[Segment], list[Chunk], list[Chunk]]:
        anchor = self._toc_service.stable_anchor(
            location,
            [parsed.title],
            0,
            page_range=(1, 1),
            anchor_hint=slugify(parsed.title),
        )
        segment = Segment(
            segment_id=self._deterministic_id(document.doc_id, anchor, "segment"),
            doc_id=document.doc_id,
            parent_segment_id=None,
            toc_path=[parsed.title],
            heading_level=1,
            page_range=(1, 1),
            order_index=0,
            anchor=anchor,
            visible_text=parsed.visible_text or parsed.visual_semantics,
            visual_semantics=parsed.visual_semantics,
            metadata={"chunk_strategy": ChunkingStrategy.IMAGE.value, "location": location},
        )
        parent_text = parsed.visual_semantics or parsed.visible_text or parsed.title
        parent_chunk = self._make_chunk(
            location=location,
            document=document,
            segment=segment,
            text=parent_text,
            access_policy=access_policy,
            chunk_role=ChunkRole.PARENT,
            order_index=0,
            parent_chunk_id=None,
            special_chunk_type=None,
        )
        child_source_text = parsed.visible_text or parent_text
        child_chunks = self._refine_parent_chunk(
            parent_chunk=parent_chunk.model_copy(
                update={
                    "text": child_source_text,
                    "token_count": text_unit_count(child_source_text),
                }
            ),
            access_policy=access_policy,
            local_refine=True,
            starting_order=0,
        )
        return [segment], [parent_chunk], child_chunks

    def _build_special_chunks(
        self,
        *,
        location: str,
        document: Document,
        parsed: ParsedDocument,
        access_policy: AccessPolicy,
        segments: list[Segment],
    ) -> list[Chunk]:
        return build_special_chunks(
            location=location,
            document=document,
            parsed=parsed,
            access_policy=access_policy,
            segments=segments,
            make_chunk=self._make_chunk,
        )

    def _refine_parent_chunk(
        self,
        *,
        parent_chunk: Chunk,
        access_policy: AccessPolicy,
        local_refine: bool,
        starting_order: int,
    ) -> list[Chunk]:
        text = normalize_whitespace(parent_chunk.text)
        if not text:
            return []
        if not local_refine and text_unit_count(text) <= 140:
            return [
                parent_chunk.model_copy(
                    update={
                        "chunk_id": self._deterministic_id(parent_chunk.chunk_id, "0", "chunk"),
                        "chunk_role": ChunkRole.CHILD,
                        "parent_chunk_id": parent_chunk.chunk_id,
                        "embedding_ref": self._deterministic_id(parent_chunk.chunk_id, "0", "chunk"),
                        "order_index": starting_order,
                    }
                )
            ]

        sentences = [
            normalize_whitespace(part)
            for part in self._SENTENCE_SPLIT_RE.split(text)
            if normalize_whitespace(part)
        ]
        if not sentences:
            sentences = [text]
        groups: list[str] = []
        buffer: list[str] = []
        max_words = 90 if local_refine else 130
        max_sentences = 3 if local_refine else 5
        for sentence in sentences:
            candidate = normalize_whitespace(" ".join([*buffer, sentence]))
            should_flush = bool(buffer) and (
                text_unit_count(candidate) > max_words or len(buffer) >= max_sentences
            )
            if should_flush:
                groups.append(normalize_whitespace(" ".join(buffer)))
                buffer = [sentence]
            else:
                buffer.append(sentence)
        if buffer:
            groups.append(normalize_whitespace(" ".join(buffer)))
        refined: list[Chunk] = []
        cursor = 0
        for index, group in enumerate(groups):
            span_start = text.find(group, cursor)
            if span_start < 0:
                span_start = cursor
            span_end = span_start + len(group)
            cursor = span_end
            chunk_id = self._deterministic_id(parent_chunk.chunk_id, str(index), "chunk")
            refined.append(
                Chunk(
                    chunk_id=chunk_id,
                    segment_id=parent_chunk.segment_id,
                    doc_id=parent_chunk.doc_id,
                    text=group,
                    token_count=text_unit_count(group),
                    citation_anchor=parent_chunk.citation_anchor,
                    citation_span=(span_start, span_end),
                    effective_access_policy=access_policy,
                    extraction_quality=parent_chunk.extraction_quality,
                    embedding_ref=chunk_id,
                    order_index=starting_order + index,
                    chunk_role=ChunkRole.CHILD,
                    parent_chunk_id=parent_chunk.chunk_id,
                    metadata=parent_chunk.metadata,
                )
            )
        return refined

    def _make_chunk(
        self,
        *,
        location: str,
        document: Document,
        segment: Segment,
        text: str,
        access_policy: AccessPolicy,
        chunk_role: ChunkRole,
        order_index: int,
        parent_chunk_id: str | None,
        special_chunk_type: str | None,
        metadata: dict[str, str] | None = None,
    ) -> Chunk:
        normalized = normalize_whitespace(text)
        chunk_id_parts = [
            document.doc_id,
            segment.anchor or segment.segment_id,
            chunk_role.value,
        ]
        if chunk_role is ChunkRole.SPECIAL:
            chunk_id_parts.extend([special_chunk_type or "special", str(order_index)])
        else:
            chunk_id_parts.append(special_chunk_type or str(order_index))
        chunk_id = self._deterministic_id(*chunk_id_parts)
        return Chunk(
            chunk_id=chunk_id,
            segment_id=segment.segment_id,
            doc_id=document.doc_id,
            text=normalized,
            token_count=text_unit_count(normalized),
            citation_anchor=segment.anchor or segment.segment_id,
            citation_span=(0, len(normalized)),
            effective_access_policy=access_policy,
            extraction_quality=1.0,
            embedding_ref=None if chunk_role is ChunkRole.PARENT else chunk_id,
            order_index=order_index,
            chunk_role=chunk_role,
            special_chunk_type=special_chunk_type,
            parent_chunk_id=parent_chunk_id,
            metadata={
                "location": location,
                "source_type": metadata.get("source_type", "") if metadata else "",
                "toc_path": " > ".join(segment.toc_path),
                **(metadata or {}),
            },
        )

    @staticmethod
    def _page_numbers(chunk: Any) -> list[int]:
        page_numbers: list[int] = []
        for item in chunk.meta.doc_items:
            for provenance in getattr(item, "prov", None) or []:
                page_no = getattr(provenance, "page_no", None)
                if page_no is not None:
                    page_numbers.append(int(page_no))
        return page_numbers

    @staticmethod
    def _toc_path_from_chunk(*, parsed: ParsedDocument, chunk: Any) -> list[str]:
        headings = [
            normalize_whitespace(heading)
            for heading in chunk.meta.headings or []
            if normalize_whitespace(heading)
        ]
        if not headings:
            return [parsed.title]
        if headings[0] != parsed.title:
            return [parsed.title, *headings]
        return headings

    @staticmethod
    def _deterministic_id(*parts: str) -> str:
        digest = sha256("\0".join(parts).encode("utf-8")).hexdigest()
        return f"{parts[-1]}-{digest[:12]}"
