from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from pkp.repo.parse._util import normalize_whitespace


@dataclass(frozen=True, slots=True)
class TokenChunkWindow:
    text: str
    token_count: int
    order_index: int
    token_span: tuple[int, int]


def _default_tokenize(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    return normalized.split() if normalized else []


def chunk_by_tokens(
    text: str,
    *,
    chunk_token_size: int = 1200,
    chunk_overlap_token_size: int = 100,
    tokenizer: Callable[[str], Sequence[str]] | None = None,
) -> list[TokenChunkWindow]:
    if chunk_token_size <= 0:
        raise ValueError("chunk_token_size must be positive")
    if chunk_overlap_token_size < 0:
        raise ValueError("chunk_overlap_token_size cannot be negative")
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError("chunk_overlap_token_size must be smaller than chunk_token_size")

    tokenize = tokenizer or _default_tokenize
    tokens = [token for token in tokenize(text) if token]
    if not tokens:
        return []

    step = chunk_token_size - chunk_overlap_token_size
    chunks: list[TokenChunkWindow] = []
    for order_index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + chunk_token_size]
        if not window:
            continue
        chunks.append(
            TokenChunkWindow(
                text=" ".join(window),
                token_count=len(window),
                order_index=order_index,
                token_span=(start, start + len(window)),
            )
        )
        if start + chunk_token_size >= len(tokens):
            break
    return chunks
