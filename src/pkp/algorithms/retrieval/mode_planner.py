from __future__ import annotations

from dataclasses import dataclass, field

from pkp.query.query import QueryMode, normalize_query_mode
from pkp.schema.query import QueryUnderstanding


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    mode: QueryMode
    internal_branches: tuple[str, ...]
    allow_special: bool = False
    allow_structure: bool = False
    allow_metadata: bool = False
    allow_web: bool = True
    allow_graph_expansion: bool = True


@dataclass(slots=True)
class RetrievalPlanBuilder:
    def build(
        self,
        *,
        query_understanding: QueryUnderstanding,
        requested_mode: QueryMode | str | None = None,
    ) -> RetrievalPlan:
        mode = normalize_query_mode(requested_mode)
        branches: list[str]
        if mode is QueryMode.NAIVE:
            branches = ["vector"]
        elif mode is QueryMode.LOCAL:
            branches = ["local"]
        elif mode is QueryMode.GLOBAL:
            branches = ["global"]
        elif mode is QueryMode.HYBRID:
            branches = ["local", "global"]
        else:
            branches = ["local", "global", "vector"]
            if query_understanding.needs_sparse:
                branches.append("full_text")

        if query_understanding.needs_structure and mode is QueryMode.MIX:
            branches.append("section")
        if query_understanding.needs_special and mode is QueryMode.MIX:
            branches.append("special")
        if query_understanding.needs_metadata and mode is QueryMode.MIX:
            branches.append("metadata")

        deduped = tuple(dict.fromkeys(branches))
        return RetrievalPlan(
            mode=mode,
            internal_branches=deduped,
            allow_special=query_understanding.needs_special and mode is QueryMode.MIX,
            allow_structure=query_understanding.needs_structure and mode is QueryMode.MIX,
            allow_metadata=query_understanding.needs_metadata and mode is QueryMode.MIX,
            allow_web=True,
            allow_graph_expansion=mode is not QueryMode.NAIVE,
        )
