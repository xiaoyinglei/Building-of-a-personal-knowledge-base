from __future__ import annotations

from collections.abc import MutableMapping

_FAST_TOKENIZER_PADDING_WARNING = "Asking-to-pad-a-fast-tokenizer"


def suppress_backend_fast_tokenizer_padding_warning(backend: object) -> object:
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        return backend
    if not _looks_like_fast_tokenizer(tokenizer):
        return backend

    deprecation_warnings = getattr(tokenizer, "deprecation_warnings", None)
    if isinstance(deprecation_warnings, MutableMapping):
        deprecation_warnings[_FAST_TOKENIZER_PADDING_WARNING] = True
    return backend


def _looks_like_fast_tokenizer(tokenizer: object) -> bool:
    if bool(getattr(tokenizer, "is_fast", False)):
        return True
    return tokenizer.__class__.__name__.endswith("Fast")
