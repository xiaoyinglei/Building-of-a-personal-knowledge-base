from pathlib import Path

from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
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


def test_sqlite_fts_repo_indexes_chunks_and_returns_ranked_matches(tmp_path: Path) -> None:
    repo = SQLiteFTSRepo(tmp_path / "fts.sqlite3")
    repo.index_chunks(
        [
            _chunk("chunk-a", "doc-a", "fast path answer with citation"),
            _chunk("chunk-b", "doc-b", "deep path research comparison"),
        ]
    )

    results = repo.search("citation answer", limit=5, doc_ids=["doc-a", "doc-b"])

    assert [item.chunk_id for item in results] == ["chunk-a"]
    assert results[0].score > 0
