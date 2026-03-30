from __future__ import annotations

import importlib
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;.\n])")
_COMMAND_FLAG_RE = re.compile(r"(^|\s)-{1,2}[a-zA-Z][a-zA-Z-]*")
_COMMAND_MARKERS = ("uv run", "curl -x", "--query", "--mode", "rag_", "```", "ollama ", "python -m")
_DEFINITION_QUERY_MARKERS = ("做什么", "是什么", "作用", "用途", "what is", "what does")
_DEFINITION_TEXT_MARKERS = ("是一个", "用于", "用来", "负责", "平台", "system", "designed to", "used to")
_OPERATION_QUERY_MARKERS = (
    "如何使用",
    "怎么使用",
    "运行方式",
    "怎么运行",
    "如何运行",
    "如何配置",
    "怎么配置",
    "怎么接入",
    "如何接入",
    "how to use",
    "how to run",
    "setup",
    "configure",
)
_OPERATION_TEXT_MARKERS = (
    "ollama",
    "openai",
    "local_only",
    "cloud_first",
    "uv sync",
    ".env",
    "安装依赖",
    "本地模式",
    "云端模式",
    "如何使用",
    "接入",
    "配置",
)
_STRUCTURE_QUERY_MARKERS = (
    "架构",
    "architecture",
    "结构",
    "分层",
    "层级",
    "模块",
    "module",
    "modules",
    "组件",
    "component",
    "components",
    "组成",
    "layer",
    "layers",
)
_GENERIC_QUERY_TERMS = frozenset(
    {
        "这个",
        "那个",
        "什么",
        "项目",
        "这个项目",
        "一下",
        "请问",
        "如何",
        "的是",
        "什么是",
        "what",
        "does",
        "this",
        "project",
        "is",
        "the",
        "a",
        "an",
        "of",
    }
)
DEFAULT_TOKENIZER_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_OPENAI_TOKENIZER_MARKERS = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "text-embedding-",
    "omni-",
)
_CODE_FENCE_MARKERS = ("```", "<code>", "</code>")
_CODE_LINE_RE = re.compile(
    r"^\s*(?:def |class |function |SELECT |INSERT |UPDATE |DELETE |FROM |WHERE "
    r"|if\s*\(|for\s*\(|while\s*\(|return\b|import\b|from\b|const\b|let\b|var\b"
    r"|public\b|private\b|protected\b|#include\b)"
)


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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True, slots=True)
class TokenizerContract:
    embedding_model_name: str
    tokenizer_model_name: str
    chunking_tokenizer_model_name: str
    tokenizer_backend: str = "auto"
    chunk_token_size: int = 480
    chunk_overlap_tokens: int = 64
    max_context_tokens: int = 1200
    prompt_reserved_tokens: int = 256
    local_files_only: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        embedding_model_name: str,
        default_context_tokens: int = 1200,
        default_chunk_token_size: int = 480,
        default_chunk_overlap_tokens: int = 64,
        default_prompt_reserved_tokens: int = 256,
    ) -> TokenizerContract:
        load_env_file()
        locked_embedding_model = (
            os.environ.get("RAG_EMBEDDING_MODEL")
            or os.environ.get("RAG_INDEX_EMBEDDING_MODEL")
            or embedding_model_name
            or DEFAULT_TOKENIZER_FALLBACK_MODEL
        )
        tokenizer_model_name = (
            os.environ.get("RAG_TOKENIZER_MODEL")
            or os.environ.get("RAG_BUDGET_TOKENIZER_MODEL")
            or locked_embedding_model
            or DEFAULT_TOKENIZER_FALLBACK_MODEL
        )
        chunking_tokenizer_model_name = (
            os.environ.get("RAG_CHUNKING_TOKENIZER_MODEL")
            or os.environ.get("RAG_DOCLING_TOKENIZER_MODEL")
            or tokenizer_model_name
            or DEFAULT_TOKENIZER_FALLBACK_MODEL
        )
        return cls(
            embedding_model_name=locked_embedding_model,
            tokenizer_model_name=tokenizer_model_name,
            chunking_tokenizer_model_name=chunking_tokenizer_model_name,
            tokenizer_backend=os.environ.get("RAG_TOKENIZER_BACKEND", "auto").strip().lower() or "auto",
            chunk_token_size=max(32, _env_int("RAG_CHUNK_TOKEN_SIZE", default_chunk_token_size)),
            chunk_overlap_tokens=max(0, _env_int("RAG_CHUNK_OVERLAP_TOKENS", default_chunk_overlap_tokens)),
            max_context_tokens=max(64, _env_int("RAG_MAX_CONTEXT_TOKENS", default_context_tokens)),
            prompt_reserved_tokens=max(32, _env_int("RAG_PROMPT_RESERVED_TOKENS", default_prompt_reserved_tokens)),
            local_files_only=_env_bool("RAG_TOKENIZER_LOCAL_FILES_ONLY", True),
        )

    def normalized_chunk_overlap_tokens(self) -> int:
        return min(self.chunk_overlap_tokens, max(self.chunk_token_size - 1, 0))


@dataclass(slots=True)
class TokenAccountingService:
    contract: TokenizerContract
    _backend_kind: str | None = None
    _backend: Any | None = None

    def count(self, text: str) -> int:
        normalized = text.strip()
        if not normalized:
            return 0
        encoded = self._encode(normalized)
        return len(encoded) if encoded is not None else text_unit_count(normalized)

    def clip(self, text: str, token_budget: int, *, add_ellipsis: bool = False) -> str:
        normalized_budget = max(token_budget, 1)
        normalized = text.strip()
        if not normalized:
            return ""
        spans = self._offset_spans(normalized)
        if spans is not None:
            clipped = self._clip_with_spans(normalized, spans, normalized_budget)
        else:
            encoded = self._encode(normalized)
            if encoded is None:
                clipped = self._clip_with_units(normalized, normalized_budget)
            elif len(encoded) <= normalized_budget:
                clipped = normalized
            else:
                clipped = self._decode(encoded[:normalized_budget]).strip()
        if not clipped:
            return ""
        if add_ellipsis and self.count(clipped) < self.count(normalized):
            return f"{clipped} ..."
        return clipped

    def tail(self, text: str, token_budget: int) -> str:
        normalized_budget = max(token_budget, 0)
        normalized = text.strip()
        if normalized_budget <= 0 or not normalized:
            return ""
        spans = self._offset_spans(normalized)
        if spans is not None:
            return self._tail_with_spans(normalized, spans, normalized_budget)
        encoded = self._encode(normalized)
        if encoded is None:
            unit_spans = _token_unit_spans(normalized)
            if not unit_spans:
                return ""
            start = unit_spans[max(len(unit_spans) - normalized_budget, 0)][0]
            return normalized[start:].strip()
        if len(encoded) <= normalized_budget:
            clipped = normalized
        else:
            clipped = self._decode(encoded[-normalized_budget:]).strip()
        return clipped

    def chunk_text(
        self,
        text: str,
        *,
        chunk_token_size: int | None = None,
        chunk_overlap_tokens: int | None = None,
    ) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        size = max(chunk_token_size or self.contract.chunk_token_size, 1)
        resolved_overlap = (
            chunk_overlap_tokens
            if chunk_overlap_tokens is not None
            else self.contract.normalized_chunk_overlap_tokens()
        )
        overlap = max(resolved_overlap, 0)
        overlap = min(overlap, max(size - 1, 0))
        spans = self._offset_spans(normalized)
        if spans is not None:
            if not spans:
                return []
            step = max(size - overlap, 1)
            chunks: list[str] = []
            for start_index in range(0, len(spans), step):
                span_window = spans[start_index : start_index + size]
                if not span_window:
                    continue
                chunks.append(normalized[span_window[0][0] : span_window[-1][1]].strip())
                if start_index + size >= len(spans):
                    break
            return [chunk for chunk in chunks if chunk]
        encoded = self._encode(normalized)
        if encoded is None:
            spans = _token_unit_spans(normalized)
            if not spans:
                return []
            step = max(size - overlap, 1)
            chunks = []
            for start_index in range(0, len(spans), step):
                span_window = spans[start_index : start_index + size]
                if not span_window:
                    continue
                chunks.append(normalized[span_window[0][0] : span_window[-1][1]].strip())
                if start_index + size >= len(spans):
                    break
            return [chunk for chunk in chunks if chunk]
        step = max(size - overlap, 1)
        chunks = []
        for start in range(0, len(encoded), step):
            token_window = encoded[start : start + size]
            if not token_window:
                continue
            chunk = self._decode(token_window).strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(encoded):
                break
        return chunks

    def prompt_budget(self, total_budget: int | None = None) -> int:
        resolved_budget = total_budget or self.contract.max_context_tokens
        return max(resolved_budget - self.contract.prompt_reserved_tokens, 1)

    def backend_descriptor(self) -> tuple[str, str]:
        self._ensure_backend()
        return self._backend_kind or "heuristic", self.contract.tokenizer_model_name

    def _encode(self, text: str) -> list[int] | None:
        self._ensure_backend()
        if self._backend_kind == "tiktoken":
            backend = self._backend
            return list(backend.encode(text)) if backend is not None else None
        if self._backend_kind == "transformers":
            backend = self._backend
            return list(backend.encode(text, add_special_tokens=False)) if backend is not None else None
        return None

    def _decode(self, tokens: Sequence[int]) -> str:
        self._ensure_backend()
        if self._backend_kind == "tiktoken":
            backend = self._backend
            return "" if backend is None else str(backend.decode(list(tokens)))
        if self._backend_kind == "transformers":
            backend = self._backend
            if backend is None:
                return ""
            return str(
                backend.decode(
                    list(tokens),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return ""

    def _ensure_backend(self) -> None:
        if self._backend is not None or self._backend_kind is not None:
            return
        backend_kind, backend = _build_tokenizer_backend(
            self.contract.tokenizer_model_name,
            backend=self.contract.tokenizer_backend,
            local_files_only=self.contract.local_files_only,
        )
        self._backend_kind = backend_kind
        self._backend = backend

    @staticmethod
    def _clip_with_units(text: str, token_budget: int) -> str:
        spans = _token_unit_spans(text)
        if len(spans) <= token_budget:
            return text
        return text[: spans[token_budget - 1][1]].strip()

    def _offset_spans(self, text: str) -> list[tuple[int, int]] | None:
        self._ensure_backend()
        if self._backend_kind == "heuristic":
            return _token_unit_spans(text)
        if self._backend_kind != "transformers":
            return None
        backend = self._backend
        if backend is None:
            return None
        try:
            payload = backend(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except Exception:
            return None
        offsets = getattr(payload, "offset_mapping", None)
        if offsets is None and isinstance(payload, dict):
            offsets = payload.get("offset_mapping")
        if not isinstance(offsets, list):
            return None
        return [
            (int(start), int(end))
            for start, end in offsets
            if isinstance(start, int) and isinstance(end, int) and end > start
        ]

    @staticmethod
    def _clip_with_spans(text: str, spans: Sequence[tuple[int, int]], token_budget: int) -> str:
        if len(spans) <= token_budget:
            return text
        return text[: spans[token_budget - 1][1]].strip()

    @staticmethod
    def _tail_with_spans(text: str, spans: Sequence[tuple[int, int]], token_budget: int) -> str:
        if len(spans) <= token_budget:
            return text
        start = spans[max(len(spans) - token_budget, 0)][0]
        return text[start:].strip()


@lru_cache(maxsize=12)
def _build_tokenizer_backend(
    model_name: str,
    *,
    backend: str,
    local_files_only: bool,
) -> tuple[str, Any | None]:
    preferred = _preferred_backend(model_name=model_name, backend=backend)
    for candidate in preferred:
        built = _try_build_tokenizer(model_name=model_name, backend=candidate, local_files_only=local_files_only)
        if built is not None:
            return candidate, built
    return "heuristic", None


def _preferred_backend(*, model_name: str, backend: str) -> tuple[str, ...]:
    normalized_backend = backend.strip().lower()
    if normalized_backend in {"tiktoken", "transformers", "heuristic"}:
        return (normalized_backend,)
    lowered_model = model_name.strip().lower()
    if any(marker in lowered_model for marker in _OPENAI_TOKENIZER_MARKERS):
        return ("tiktoken", "transformers", "heuristic")
    return ("transformers", "tiktoken", "heuristic")


def _try_build_tokenizer(*, model_name: str, backend: str, local_files_only: bool) -> Any | None:
    if backend == "heuristic":
        return None
    if backend == "tiktoken":
        spec = importlib.util.find_spec("tiktoken")
        if spec is None:
            return None
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    if backend == "transformers":
        spec = importlib.util.find_spec("transformers")
        if spec is None:
            return None
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_fast=True,
                trust_remote_code=True,
            )  # type: ignore[no-untyped-call]
        except Exception:
            return None
    return None


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


def looks_definition_query(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _DEFINITION_QUERY_MARKERS)


def looks_definition_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _DEFINITION_TEXT_MARKERS)


def looks_operation_query(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _OPERATION_QUERY_MARKERS)


def looks_operation_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _OPERATION_TEXT_MARKERS)


def looks_structure_query(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _STRUCTURE_QUERY_MARKERS)


def looks_structure_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _STRUCTURE_QUERY_MARKERS)


def focus_terms(text: str) -> tuple[str, ...]:
    filtered = tuple(term for term in search_terms(text) if term not in _GENERIC_QUERY_TERMS)
    return filtered or search_terms(text)
