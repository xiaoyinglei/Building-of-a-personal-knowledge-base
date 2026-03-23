from pkp.repo.search.in_memory_vector_repo import InMemoryVectorRepo
from pkp.types import AccessPolicy, Chunk


def _chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        segment_id=f"seg-{chunk_id}",
        doc_id=doc_id,
        text=text,
        token_count=len(text.split()),
        citation_anchor=chunk_id,
        citation_span=(0, len(text)),
        effective_access_policy=AccessPolicy.default(),
        extraction_quality=1.0,
        embedding_ref=None,
    )


def test_in_memory_vector_repo_returns_most_similar_chunks() -> None:
    repo = InMemoryVectorRepo()
    repo.index_chunks(
        [
            _chunk("chunk-a", "doc-a", "fast path direct answer"),
            _chunk("chunk-b", "doc-b", "deep research comparison synthesis"),
        ]
    )

    results = repo.search("comparison synthesis", limit=2, doc_ids=["doc-a", "doc-b"])

    assert [item.chunk_id for item in results][0] == "chunk-b"
    assert results[0].score >= results[1].score
