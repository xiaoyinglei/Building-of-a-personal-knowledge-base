from __future__ import annotations

from collections.abc import Callable

from pkp.repo.interfaces import ParsedDocument, ParsedElement
from pkp.repo.parse._util import normalize_whitespace
from pkp.types.access import AccessPolicy
from pkp.types.content import Chunk, ChunkRole, Document, Segment


def special_type_for_element(element: ParsedElement) -> str | None:
    if element.kind == "table":
        return "table"
    if element.kind == "figure":
        return "figure"
    if element.kind == "caption":
        return "caption"
    if element.kind == "ocr_region":
        return "ocr_region"
    return None


def build_special_chunks(
    *,
    location: str,
    document: Document,
    parsed: ParsedDocument,
    access_policy: AccessPolicy,
    segments: list[Segment],
    make_chunk: Callable[..., Chunk],
) -> list[Chunk]:
    segment_by_path = {tuple(segment.toc_path): segment for segment in segments}
    fallback_segment = segments[0] if segments else None
    special_chunks: list[Chunk] = []

    for index, element in enumerate(parsed.elements):
        special_type = special_type_for_element(element)
        if special_type is None:
            continue
        target_segment = segment_by_path.get(tuple(element.toc_path), fallback_segment)
        if target_segment is None:
            continue
        special_chunks.append(
            make_chunk(
                location=location,
                document=document,
                segment=target_segment,
                text=element.text,
                access_policy=access_policy,
                chunk_role=ChunkRole.SPECIAL,
                order_index=index,
                parent_chunk_id=None,
                special_chunk_type=special_type,
                metadata={
                    "bbox": "" if element.bbox is None else ",".join(f"{value:.2f}" for value in element.bbox),
                    "page_no": "" if element.page_no is None else str(element.page_no),
                    **element.metadata,
                },
            )
        )

    if parsed.source_type.value == "image" and fallback_segment is not None:
        summary_text = normalize_whitespace(parsed.visual_semantics or parsed.visible_text or parsed.title)
        if summary_text:
            special_chunks.append(
                make_chunk(
                    location=location,
                    document=document,
                    segment=fallback_segment,
                    text=summary_text,
                    access_policy=access_policy,
                    chunk_role=ChunkRole.SPECIAL,
                    order_index=len(special_chunks),
                    parent_chunk_id=None,
                    special_chunk_type="image_summary",
                    metadata=None,
                )
            )

    return special_chunks
