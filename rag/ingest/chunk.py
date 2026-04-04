from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import PictureItem, TableItem

from rag.document._parse._util import normalize_whitespace, slugify
from rag.ingest._chunking.multimodal_chunk_router import build_special_chunks, special_type_for_element
from rag.ingest._chunking.structured_chunker import ChunkSeed, build_cached_hybrid_chunker, merge_adjacent_seeds
from rag.ingest._chunking.token_chunker import chunk_by_tokens
from rag.schema._types.content import Chunk, ChunkRole, Document, Segment, Source, SourceType
from rag.schema._types.processing import (
    ChunkingStrategy,
    ChunkRoutingDecision,
    ChunkStatistics,
    DocumentFeatures,
    DocumentProcessingPackage,
)
from rag.schema._types.text import (
    DEFAULT_TOKENIZER_FALLBACK_MODEL,
    TokenAccountingService,
    TokenizerContract,
    looks_code_like,
    split_sentences,
    text_unit_count,
)
from rag.schema.document import AccessPolicy
from rag.utils._contracts import ParsedDocument


class TOCService:
    def normalize_path(self, headings: list[str] | tuple[str, ...]) -> list[str]:
        return [heading.strip() for heading in headings if heading.strip()]

    def stable_anchor(
        self,
        location: str,
        toc_path: list[str] | tuple[str, ...],
        order_index: int,
        *,
        page_range: tuple[int, int] | None = None,
        anchor_hint: str | None = None,
    ) -> str:
        normalized_path = self.normalize_path(toc_path)
        prefix = self._anchor_prefix(location)
        if page_range is not None:
            hint = anchor_hint or f"page-{page_range[0]}"
            basis = f"{prefix}|{'/'.join(normalized_path)}|{order_index}|{page_range}"
            return f"{prefix}#{hint}-{self._digest(basis)}"

        hint = anchor_hint or "-".join(slugify(part) for part in normalized_path)
        basis = f"{prefix}|{'/'.join(normalized_path)}|{order_index}"
        return f"{prefix}#{hint}-{self._digest(basis)}"

    @staticmethod
    def _anchor_prefix(location: str) -> str:
        from pathlib import Path

        path = Path(location)
        return location if not path.is_absolute() else path.name

    @staticmethod
    def _digest(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()[:10]


class DocumentFeatureService:
    def analyze(self, parsed: ParsedDocument) -> DocumentFeatures:
        word_count = len(parsed.visible_text.split())
        section_count = len([section for section in parsed.sections if section.text.strip()])
        heading_count = len([element for element in parsed.elements if element.kind == "section_header"])
        table_count = len([element for element in parsed.elements if element.kind == "table"])
        figure_count = len([element for element in parsed.elements if element.kind == "figure"])
        caption_count = len([element for element in parsed.elements if element.kind == "caption"])
        ocr_region_count = len([element for element in parsed.elements if element.kind == "ocr_region"])
        structure_depth = max((len(section.toc_path) - 1 for section in parsed.sections), default=1)
        avg_section_words = word_count / max(section_count, 1)
        heading_quality_score = self._heading_quality_score(
            section_count=section_count,
            heading_count=heading_count,
            structure_depth=structure_depth,
        )
        return DocumentFeatures(
            source_type=parsed.source_type,
            section_count=section_count,
            word_count=word_count,
            heading_count=heading_count,
            heading_quality_score=heading_quality_score,
            table_count=table_count,
            figure_count=figure_count,
            caption_count=caption_count,
            ocr_region_count=ocr_region_count,
            avg_section_words=avg_section_words,
            structure_depth=structure_depth,
            has_dense_structure=heading_count >= 2 and structure_depth >= 2,
            metadata={
                "page_count": str(parsed.page_count or 0),
                "has_visual_semantics": str(bool(parsed.visual_semantics)),
            },
        )

    @staticmethod
    def _heading_quality_score(
        *,
        section_count: int,
        heading_count: int,
        structure_depth: int,
    ) -> float:
        if heading_count <= 0 or section_count <= 0:
            return 0.0
        density = min(heading_count / section_count, 1.0)
        depth_score = min(structure_depth / 4.0, 1.0)
        return round((density * 0.7) + (depth_score * 0.3), 4)


class ChunkRoutingService:
    def route(self, features: DocumentFeatures) -> ChunkRoutingDecision:
        reasons: list[str] = []
        debug = {
            "heading_quality_score": f"{features.heading_quality_score:.3f}",
            "section_count": str(features.section_count),
            "table_count": str(features.table_count),
            "figure_count": str(features.figure_count),
            "ocr_region_count": str(features.ocr_region_count),
        }
        if features.source_type is SourceType.PDF:
            reasons.append("PDF defaults to HybridChunker for mixed layout resilience.")
            if features.table_count > 0 or features.avg_section_words >= 220:
                reasons.append("Large or mixed-content PDF enables local refine.")
            return ChunkRoutingDecision(
                source_type=features.source_type,
                selected_strategy=ChunkingStrategy.HYBRID,
                special_chunk_mode=True,
                local_refine=features.table_count > 0 or features.avg_section_words >= 220,
                fallback=False,
                reasons=reasons,
                debug=debug,
            )
        if features.source_type is SourceType.MARKDOWN:
            reasons.append("Markdown headings are usually reliable, prefer HierarchicalChunker.")
            return ChunkRoutingDecision(
                source_type=features.source_type,
                selected_strategy=ChunkingStrategy.HIERARCHICAL,
                special_chunk_mode=True,
                local_refine=False,
                fallback=False,
                reasons=reasons,
                debug=debug,
            )
        if features.source_type is SourceType.DOCX:
            if features.heading_quality_score >= 0.55 and features.heading_count >= 2:
                reasons.append("DOCX heading quality is high, prefer HierarchicalChunker.")
                strategy = ChunkingStrategy.HIERARCHICAL
                local_refine = False
            else:
                reasons.append("DOCX heading quality is low, fall back to HybridChunker.")
                strategy = ChunkingStrategy.HYBRID
                local_refine = True
            return ChunkRoutingDecision(
                source_type=features.source_type,
                selected_strategy=strategy,
                special_chunk_mode=True,
                local_refine=local_refine,
                fallback=False,
                reasons=reasons,
                debug=debug,
            )
        if features.source_type is SourceType.IMAGE:
            reasons.append("Image route uses regions, summary chunks, and caption binding.")
            return ChunkRoutingDecision(
                source_type=features.source_type,
                selected_strategy=ChunkingStrategy.IMAGE,
                special_chunk_mode=True,
                local_refine=True,
                fallback=True,
                reasons=reasons,
                debug=debug,
            )
        raise ValueError(f"Unsupported source type for routing: {features.source_type}")


@dataclass(frozen=True)
class PostprocessedChunks:
    parent_chunks: list[Chunk]
    child_chunks: list[Chunk]
    special_chunks: list[Chunk]
    merged_chunks: int
    deduplicated_chunks: int


class ChunkPostprocessingService:
    def __init__(self, *, min_words: int = 24, token_accounting: TokenAccountingService | None = None) -> None:
        self._min_words = min_words
        self._token_accounting = token_accounting

    def postprocess(
        self,
        *,
        parent_chunks: list[Chunk],
        child_chunks: list[Chunk],
        special_chunks: list[Chunk],
    ) -> PostprocessedChunks:
        normalized_parents = [
            self._clean_chunk(chunk) for chunk in sorted(parent_chunks, key=lambda item: item.order_index)
        ]
        normalized_children = [
            self._clean_chunk(chunk) for chunk in sorted(child_chunks, key=lambda item: item.order_index)
        ]
        normalized_special = [
            self._clean_chunk(chunk) for chunk in sorted(special_chunks, key=lambda item: item.order_index)
        ]

        merged_children, merged_count = self._merge_short_chunks(normalized_children)
        deduped_children, child_dedup = self._deduplicate(merged_children)
        deduped_special, special_dedup = self._deduplicate(normalized_special)
        linked_children = self._link_neighbors(deduped_children)
        linked_special = self._link_neighbors(deduped_special)
        return PostprocessedChunks(
            parent_chunks=self._assign_hashes(normalized_parents),
            child_chunks=self._assign_hashes(linked_children),
            special_chunks=self._assign_hashes(linked_special),
            merged_chunks=merged_count,
            deduplicated_chunks=child_dedup + special_dedup,
        )

    def _merge_short_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        if not chunks:
            return [], 0
        merged: list[Chunk] = []
        merge_count = 0
        index = 0
        while index < len(chunks):
            current = chunks[index]
            word_count = self._count_tokens(current.text)
            if word_count >= self._min_words:
                merged.append(current)
                index += 1
                continue

            if index + 1 < len(chunks) and chunks[index + 1].parent_chunk_id == current.parent_chunk_id:
                next_chunk = chunks[index + 1]
                merged_text = normalize_whitespace(f"{current.text} {next_chunk.text}")
                merged.append(
                    next_chunk.model_copy(
                        update={
                            "text": merged_text,
                            "token_count": self._count_tokens(merged_text),
                            "citation_span": (0, len(merged_text)),
                            "order_index": current.order_index,
                        }
                    )
                )
                merge_count += 1
                index += 2
                continue

            if merged and merged[-1].parent_chunk_id == current.parent_chunk_id:
                previous = merged[-1]
                merged_text = normalize_whitespace(f"{previous.text} {current.text}")
                merged[-1] = previous.model_copy(
                    update={
                        "text": merged_text,
                        "token_count": self._count_tokens(merged_text),
                        "citation_span": (0, len(merged_text)),
                    }
                )
                merge_count += 1
            else:
                merged.append(current)
            index += 1
        return merged, merge_count

    @staticmethod
    def _deduplicate(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        seen: set[tuple[str | None, str, str]] = set()
        results: list[Chunk] = []
        deduped = 0
        for chunk in chunks:
            key = (chunk.parent_chunk_id, chunk.special_chunk_type or "", chunk.text)
            if key in seen:
                deduped += 1
                continue
            seen.add(key)
            results.append(chunk)
        return results, deduped

    @staticmethod
    def _link_neighbors(chunks: list[Chunk]) -> list[Chunk]:
        linked: list[Chunk] = []
        for index, chunk in enumerate(chunks):
            previous = chunks[index - 1].chunk_id if index > 0 else None
            next_chunk = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
            linked.append(
                chunk.model_copy(
                    update={
                        "prev_chunk_id": previous,
                        "next_chunk_id": next_chunk,
                    }
                )
            )
        return linked

    @staticmethod
    def _assign_hashes(chunks: list[Chunk]) -> list[Chunk]:
        results: list[Chunk] = []
        for chunk in chunks:
            content_hash = sha256(chunk.text.encode("utf-8")).hexdigest()
            results.append(chunk.model_copy(update={"content_hash": content_hash}))
        return results

    def _clean_chunk(self, chunk: Chunk) -> Chunk:
        cleaned = normalize_whitespace(chunk.text)
        return chunk.model_copy(
            update={
                "text": cleaned,
                "token_count": self._count_tokens(cleaned),
                "citation_span": (0, len(cleaned)),
            }
        )

    def _count_tokens(self, text: str) -> int:
        if self._token_accounting is not None:
            return self._token_accounting.count(text)
        return text_unit_count(text)


class ChunkingService:
    def __init__(
        self,
        *,
        max_words: int = 160,
        token_accounting: TokenAccountingService | None = None,
    ) -> None:
        self._max_words = max_words
        self._token_accounting = token_accounting

    def chunk_section(
        self,
        *,
        location: str,
        doc_id: str,
        segment: Segment,
        text: str,
        access_policy: AccessPolicy,
    ) -> list[Chunk]:
        normalized = normalize_whitespace(text)
        if not normalized:
            return []

        effective_size = self._max_chunk_tokens()
        effective_overlap = self._chunk_overlap_tokens(text=normalized)
        chunks = self._chunk_text(
            normalized,
            chunk_token_size=effective_size,
            chunk_overlap_tokens=effective_overlap,
        )
        if not chunks:
            chunks = [normalized]

        results: list[Chunk] = []
        cursor = 0
        for index, chunk_text in enumerate(chunks):
            span_start = normalized.find(chunk_text, cursor)
            if span_start < 0:
                span_start = cursor
            span_end = span_start + len(chunk_text)
            cursor = span_end
            chunk_id = self._chunk_id(location, segment.anchor or segment.segment_id, index)
            results.append(
                Chunk(
                    chunk_id=chunk_id,
                    segment_id=segment.segment_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    token_count=self._count_tokens(chunk_text),
                    citation_anchor=segment.anchor or segment.segment_id,
                    citation_span=(span_start, span_end),
                    effective_access_policy=access_policy,
                    extraction_quality=1.0 if len(chunks) == 1 else 0.95,
                    embedding_ref=chunk_id,
                    order_index=index,
                    metadata={"order_index": str(index)},
                )
            )
        return results

    @staticmethod
    def _chunk_id(location: str, anchor: str, index: int) -> str:
        digest = sha256(f"{location}|{anchor}|{index}".encode()).hexdigest()
        return f"chunk-{digest[:16]}"

    def _count_tokens(self, text: str) -> int:
        if self._token_accounting is not None:
            return self._token_accounting.count(text)
        return text_unit_count(text)

    def _chunk_text(
        self,
        text: str,
        *,
        chunk_token_size: int,
        chunk_overlap_tokens: int,
    ) -> list[str]:
        if self._token_accounting is not None:
            return self._token_accounting.chunk_text(
                text,
                chunk_token_size=chunk_token_size,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        windows = chunk_by_tokens(
            text,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_tokens,
        )
        return [window.text for window in windows]

    def _max_chunk_tokens(self) -> int:
        if self._token_accounting is not None:
            return self._token_accounting.contract.chunk_token_size
        return self._max_words

    def _chunk_overlap_tokens(self, *, text: str) -> int:
        if self._token_accounting is None or looks_code_like(text):
            return 0
        return self._token_accounting.contract.normalized_chunk_overlap_tokens()


@dataclass(frozen=True)
class PreparedDocumentProcessing:
    segments: list[Segment]
    stored_chunks: list[Chunk]
    indexed_chunks: list[Chunk]
    package: DocumentProcessingPackage


class DocumentProcessingService:
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s*")
    _DEFAULT_TOKENIZER_MODEL = DEFAULT_TOKENIZER_FALLBACK_MODEL
    _DEFAULT_TOKENIZER_MAX_TOKENS = 512

    def __init__(
        self,
        *,
        toc_service: TOCService,
        feature_service: DocumentFeatureService | None = None,
        routing_service: ChunkRoutingService | None = None,
        postprocessing_service: ChunkPostprocessingService | None = None,
        token_accounting: TokenAccountingService | None = None,
        tokenizer_contract: TokenizerContract | None = None,
    ) -> None:
        contract = tokenizer_contract or TokenizerContract.from_env(
            embedding_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            default_chunk_token_size=self._DEFAULT_TOKENIZER_MAX_TOKENS,
        )
        self._token_accounting = token_accounting or TokenAccountingService(contract)
        self._tokenizer_contract = contract
        self._toc_service = toc_service
        self._feature_service = feature_service or DocumentFeatureService()
        self._routing_service = routing_service or ChunkRoutingService()
        self._postprocessing_service = postprocessing_service or ChunkPostprocessingService(
            token_accounting=self._token_accounting
        )
        self._hierarchical_chunker = HierarchicalChunker()
        self._resolved_chunking_tokenizer_model = (
            contract.chunking_tokenizer_model_name or self._DEFAULT_TOKENIZER_MODEL
        )
        self._hybrid_chunker = self._build_hybrid_chunker(contract)

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
            "tokenizer_backend": self._token_accounting.backend_descriptor()[0],
            "tokenizer_model": self._tokenizer_contract.tokenizer_model_name,
            "chunking_tokenizer_model": self._resolved_chunking_tokenizer_model,
            "chunk_token_size": self._tokenizer_contract.chunk_token_size,
            "chunk_overlap_tokens": self._tokenizer_contract.normalized_chunk_overlap_tokens(),
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

    def _build_hybrid_chunker(self, contract: TokenizerContract) -> Any:
        attempted: list[tuple[str, bool]] = []
        for model_name, local_files_only in (
            (self._resolved_chunking_tokenizer_model, contract.local_files_only),
            (self._DEFAULT_TOKENIZER_MODEL, contract.local_files_only),
            (self._DEFAULT_TOKENIZER_MODEL, False),
        ):
            if (model_name, local_files_only) in attempted:
                continue
            attempted.append((model_name, local_files_only))
            try:
                self._resolved_chunking_tokenizer_model = model_name
                return build_cached_hybrid_chunker(
                    tokenizer_cls=HuggingFaceTokenizer,
                    hybrid_chunker_cls=HybridChunker,
                    model_name=model_name,
                    max_tokens=contract.chunk_token_size,
                    local_files_only=local_files_only,
                )
            except Exception:
                continue
        raise RuntimeError("Unable to initialize Docling chunk tokenizer for the current tokenizer contract")

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
            return self._build_fallback_section_chunks(
                location=location,
                document=document,
                parsed=parsed,
                access_policy=access_policy,
                local_refine=local_refine,
            )

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
            unit_counter=self._token_accounting.count,
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
                    "token_count": self._count_tokens(child_source_text),
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
        max_tokens = self._max_chunk_tokens(local_refine=local_refine)
        if not local_refine and self._count_tokens(text) <= max_tokens:
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

        overlap_tokens = self._overlap_tokens(text=text)
        groups = self._coarse_to_fine_groups(
            text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        refined: list[Chunk] = []
        cursor = 0
        for index, group in enumerate(groups):
            search_start = max(0, cursor - len(group))
            span_start = text.find(group, search_start)
            if span_start < 0:
                span_start = cursor
            span_end = span_start + len(group)
            cursor = max(cursor, span_end)
            chunk_id = self._deterministic_id(parent_chunk.chunk_id, str(index), "chunk")
            refined.append(
                Chunk(
                    chunk_id=chunk_id,
                    segment_id=parent_chunk.segment_id,
                    doc_id=parent_chunk.doc_id,
                    text=group,
                    token_count=self._count_tokens(group),
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
            token_count=self._count_tokens(normalized),
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

    def _coarse_to_fine_groups(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        units = [normalize_whitespace(part) for part in split_sentences(text) if normalize_whitespace(part)]
        if not units:
            units = [text]
        groups: list[str] = []
        buffer = ""
        allow_overlap = overlap_tokens > 0 and not looks_code_like(text)
        for unit in units:
            unit_tokens = self._count_tokens(unit)
            if unit_tokens > max_tokens:
                if buffer:
                    groups.append(buffer)
                    buffer = ""
                groups.extend(
                    self._chunk_text(
                        unit,
                        chunk_token_size=max_tokens,
                        chunk_overlap_tokens=overlap_tokens if allow_overlap else 0,
                    )
                )
                continue
            candidate = normalize_whitespace(f"{buffer} {unit}") if buffer else unit
            if buffer and self._count_tokens(candidate) > max_tokens:
                groups.append(buffer)
                overlap_seed = self._tail_text(buffer, overlap_tokens) if allow_overlap else ""
                candidate = normalize_whitespace(f"{overlap_seed} {unit}") if overlap_seed else unit
                if overlap_seed and self._count_tokens(candidate) > max_tokens:
                    candidate = unit
                buffer = candidate
            else:
                buffer = candidate
        if buffer:
            groups.append(buffer)
        return groups or [text]

    def _count_tokens(self, text: str) -> int:
        return self._token_accounting.count(text)

    def _chunk_text(
        self,
        text: str,
        *,
        chunk_token_size: int,
        chunk_overlap_tokens: int,
    ) -> list[str]:
        chunks = self._token_accounting.chunk_text(
            text,
            chunk_token_size=chunk_token_size,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        return chunks or [text]

    def _tail_text(self, text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        return self._token_accounting.tail(text, token_budget)

    def _overlap_tokens(self, *, text: str) -> int:
        if looks_code_like(text):
            return 0
        return self._tokenizer_contract.normalized_chunk_overlap_tokens()

    def _max_chunk_tokens(self, *, local_refine: bool) -> int:
        base = max(self._tokenizer_contract.chunk_token_size, 32)
        if not local_refine:
            return base
        return max(int(base * 0.72), 64)

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
            normalize_whitespace(heading) for heading in chunk.meta.headings or [] if normalize_whitespace(heading)
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


__all__ = [
    "ChunkPostprocessingService",
    "ChunkRoutingService",
    "ChunkSeed",
    "ChunkingService",
    "DocumentFeatureService",
    "DocumentProcessingService",
    "PostprocessedChunks",
    "PreparedDocumentProcessing",
    "TOCService",
    "build_special_chunks",
    "chunk_by_tokens",
    "merge_adjacent_seeds",
    "special_type_for_element",
]
