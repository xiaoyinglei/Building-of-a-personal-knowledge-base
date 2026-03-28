from __future__ import annotations

from enum import StrEnum


class QueryMode(StrEnum):
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"


def normalize_query_mode(mode: QueryMode | str | None) -> QueryMode:
    if mode is None:
        return QueryMode.MIX
    if isinstance(mode, QueryMode):
        return mode
    return QueryMode(mode)
