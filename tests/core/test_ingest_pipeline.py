from __future__ import annotations

from pkp.algorithms.chunking.multimodal_chunk_router import special_type_for_element
from pkp.algorithms.chunking.structured_chunker import ChunkSeed, merge_adjacent_seeds
from pkp.algorithms.chunking.token_chunker import chunk_by_tokens
from pkp.repo.interfaces import ParsedElement


def test_token_chunker_produces_stable_child_chunks() -> None:
    chunks = chunk_by_tokens(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        chunk_token_size=4,
        chunk_overlap_token_size=1,
    )

    assert len(chunks) >= 3
    assert chunks[0].order_index == 0
    assert all(chunk.text for chunk in chunks)


def test_structured_chunker_merges_adjacent_markdown_seeds_with_same_path() -> None:
    seeds = [
        ChunkSeed(text="Overview", toc_path=["Quarterly Review", "Revenue"], page_numbers=[1]),
        ChunkSeed(text="expanded", toc_path=["Quarterly Review", "Revenue"], page_numbers=[1]),
    ]

    merged = merge_adjacent_seeds(seeds, source_type="markdown")

    assert len(merged) == 1
    assert merged[0].text == "Overview expanded"


def test_multimodal_router_classifies_special_elements() -> None:
    table = ParsedElement(element_id="table-1", kind="table", text="q1,q2")
    paragraph = ParsedElement(element_id="para-1", kind="paragraph", text="plain text")

    assert special_type_for_element(table) == "table"
    assert special_type_for_element(paragraph) is None
