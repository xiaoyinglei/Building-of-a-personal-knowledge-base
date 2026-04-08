from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;.\n])")
_COMMAND_FLAG_RE = re.compile(r"(^|\s)-{1,2}[a-zA-Z][a-zA-Z-]*")
_COMMAND_MARKERS = ("uv run", "curl -x", "--query", "--mode", "rag_", "```", "ollama ", "python -m")
_CODE_FENCE_MARKERS = ("```", "<code>", "</code>")
_CODE_LINE_RE = re.compile(
    r"^\s*(?:def |class |function |SELECT |INSERT |UPDATE |DELETE |FROM |WHERE "
    r"|if\s*\(|for\s*\(|while\s*\(|return\b|import\b|from\b|const\b|let\b|var\b"
    r"|public\b|private\b|protected\b|#include\b)"
)

DEFAULT_TOKENIZER_FALLBACK_MODEL = "BAAI/bge-m3"


def load_env_file(path: Path | str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key or normalized_key in os.environ:
            continue
        normalized_value = value.strip()
        if (
            len(normalized_value) >= 2
            and normalized_value[0] == normalized_value[-1]
            and normalized_value[0] in {'"', "'"}
        ):
            normalized_value = normalized_value[1:-1]
        os.environ[normalized_key] = normalized_value


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


def text_unit_count(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    ascii_count = len(_ASCII_TOKEN_RE.findall(normalized))
    cjk_count = len(_CJK_CHAR_RE.findall(normalized))
    return ascii_count + cjk_count


def looks_code_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(marker in lowered for marker in _CODE_FENCE_MARKERS):
        return True
    lines = [line.rstrip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    code_like_lines = 0
    for line in lines[:12]:
        if _CODE_LINE_RE.match(line):
            code_like_lines += 1
            continue
        if any(symbol in line for symbol in ("{", "}", "();", "=>", "::", "</", "/>", "SELECT ", "WHERE ", "FROM ")):
            code_like_lines += 1
            continue
        if line.startswith(("    ", "\t", "$ ", ">>> ")):
            code_like_lines += 1
    return code_like_lines >= max(2, len(lines) // 2)


def _token_unit_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _TOKEN_RE.finditer(text)]


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


__all__ = [
    "DEFAULT_TOKENIZER_FALLBACK_MODEL",
    "_token_unit_spans",
    "build_fts_query",
    "keyword_overlap",
    "load_env_file",
    "looks_code_like",
    "looks_command_like",
    "search_terms",
    "split_sentences",
    "text_unit_count",
]
