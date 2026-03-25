from __future__ import annotations

import re
from collections.abc import Iterable

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;.\n])")
_COMMAND_FLAG_RE = re.compile(r"(^|\s)-{1,2}[a-zA-Z][a-zA-Z-]*")
_COMMAND_MARKERS = ("uv run", "curl -x", "--query", "--mode", "pkp_", "```", "ollama ", "python -m")
_DEFINITION_QUERY_MARKERS = ("做什么", "是什么", "作用", "用途", "what is", "what does")
_DEFINITION_TEXT_MARKERS = ("是一个", "用于", "用来", "负责", "平台", "system", "designed to", "used to")


def search_terms(text: str) -> tuple[str, ...]:
    normalized = text.strip().lower()
    if not normalized:
        return ()

    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        value = term.strip().lower()
        if not value or value in seen:
            return
        seen.add(value)
        terms.append(value)

    for token in _ASCII_TOKEN_RE.findall(normalized):
        add(token)

    for run in _CJK_RUN_RE.findall(normalized):
        add(run)
        if len(run) == 1:
            continue
        for index in range(len(run) - 1):
            add(run[index : index + 2])

    return tuple(terms)


def build_fts_query(text: str) -> str:
    terms = search_terms(text)
    if not terms:
        return ""
    escaped_terms = [term.replace('"', '""') for term in terms]
    return " OR ".join(f'"{term}"' for term in escaped_terms)


def split_sentences(text: str) -> tuple[str, ...]:
    normalized = text.strip()
    if not normalized:
        return ()
    chunks = [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    return tuple(chunks or [normalized])


def keyword_overlap(query_terms: Iterable[str], text: str) -> int:
    term_set = set(query_terms)
    if not term_set:
        return 0
    return sum(1 for term in search_terms(text) if term in term_set)


def looks_command_like(text: str) -> bool:
    lowered = text.lower()
    return (
        any(marker in lowered for marker in _COMMAND_MARKERS)
        or "http://" in lowered
        or "https://" in lowered
        or "127.0.0.1" in lowered
        or "content-type" in lowered
        or '{"query"' in lowered
        or bool(_COMMAND_FLAG_RE.search(text))
    )


def looks_definition_query(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _DEFINITION_QUERY_MARKERS)


def looks_definition_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _DEFINITION_TEXT_MARKERS)
