from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from rag.document._parse._util import normalize_whitespace
from rag.schema._types.text import text_unit_count

DEFAULT_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOKENIZER_MAX_TOKENS = 512


@dataclass(frozen=True)
class ChunkSeed:
    text: str
    toc_path: list[str]
    page_numbers: list[int]


def build_cached_tokenizer(
    *,
    tokenizer_cls: Any = HuggingFaceTokenizer,
    model_name: str = DEFAULT_TOKENIZER_MODEL,
    max_tokens: int = DEFAULT_TOKENIZER_MAX_TOKENS,
) -> Any:
    return tokenizer_cls.from_pretrained(
        model_name=model_name,
        max_tokens=max_tokens,
        local_files_only=True,
    )


def build_cached_hybrid_chunker(
    *,
    tokenizer_cls: Any = HuggingFaceTokenizer,
    hybrid_chunker_cls: Any = HybridChunker,
    model_name: str = DEFAULT_TOKENIZER_MODEL,
    max_tokens: int = DEFAULT_TOKENIZER_MAX_TOKENS,
) -> Any:
    tokenizer = build_cached_tokenizer(
        tokenizer_cls=tokenizer_cls,
        model_name=model_name,
        max_tokens=max_tokens,
    )
    return hybrid_chunker_cls(tokenizer=tokenizer)


def merge_adjacent_seeds(seeds: list[ChunkSeed], *, source_type: str) -> list[ChunkSeed]:
    if not seeds:
        return []

    merged: list[ChunkSeed] = [seeds[0]]
    for seed in seeds[1:]:
        previous = merged[-1]
        should_merge = previous.toc_path == seed.toc_path and (
            source_type in {"markdown", "docx"}
            or text_unit_count(previous.text) < 12
            or text_unit_count(seed.text) < 12
        )
        if not should_merge:
            merged.append(seed)
            continue
        merged[-1] = ChunkSeed(
            text=normalize_whitespace(f"{previous.text} {seed.text}"),
            toc_path=previous.toc_path,
            page_numbers=[*previous.page_numbers, *seed.page_numbers],
        )
    return merged
