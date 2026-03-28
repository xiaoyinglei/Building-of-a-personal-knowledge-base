from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class QueryOptions:
    mode: Literal["naive", "local", "global", "hybrid", "mix"] = "mix"
