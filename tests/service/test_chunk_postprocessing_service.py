from rag.ingest.chunking import ChunkPostprocessingService
from rag.schema.core import Chunk, ChunkRole
from rag.schema.runtime import AccessPolicy


def _chunk(
    chunk_id: str,
    text: str,
    *,
    parent_chunk_id: str | None = None,
    order_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        segment_id="seg-1",
        doc_id="doc-1",
        text=text,
        token_count=len(text.split()),
        citation_anchor="doc-1#sec-1",
        citation_span=(0, len(text)),
        effective_access_policy=AccessPolicy.default(),
        extraction_quality=0.9,
        embedding_ref=chunk_id,
        chunk_role=ChunkRole.CHILD,
        parent_chunk_id=parent_chunk_id,
        order_index=order_index,
    )


def test_postprocessing_merges_short_chunks_deduplicates_and_links_neighbors() -> None:
    service = ChunkPostprocessingService(min_words=5)
    parent = _chunk(
        "parent-1",
        "Executive summary revenue growth customer expansion and margin improvement.",
        order_index=0,
    ).model_copy(update={"chunk_role": ChunkRole.PARENT})
    chunks = [
        _chunk("child-1", "Short note.", parent_chunk_id=parent.chunk_id, order_index=0),
        _chunk(
            "child-2",
            "Revenue increased strongly in the quarter because enterprise usage expanded quickly.",
            parent_chunk_id=parent.chunk_id,
            order_index=1,
        ),
        _chunk(
            "child-3",
            "Revenue increased strongly in the quarter because enterprise usage expanded quickly.",
            parent_chunk_id=parent.chunk_id,
            order_index=2,
        ),
        _chunk("child-4", "Tiny.", parent_chunk_id=parent.chunk_id, order_index=3),
    ]

    processed = service.postprocess(parent_chunks=[parent], child_chunks=chunks, special_chunks=[])

    assert len(processed.child_chunks) == 2
    assert all(chunk.content_hash for chunk in processed.child_chunks)
    assert processed.child_chunks[0].next_chunk_id == processed.child_chunks[1].chunk_id
    assert processed.child_chunks[1].prev_chunk_id == processed.child_chunks[0].chunk_id
    assert processed.child_chunks[0].text.startswith("Short note.")
    assert processed.child_chunks[1].text.endswith("Tiny.")
