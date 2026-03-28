from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pkp.core.options import QueryOptions
from pkp.core.storage_config import StorageConfig
from pkp.stores import StorageBundle


@dataclass(slots=True)
class RAGCore:
    storage: StorageConfig
    stores: StorageBundle = field(init=False)

    def __post_init__(self) -> None:
        self.stores = self.storage.build()

    def insert(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("insert pipeline is not implemented yet")

    def query(self, *args: Any, options: QueryOptions | None = None, **kwargs: Any) -> None:
        del options
        raise NotImplementedError("query pipeline is not implemented yet")

    def delete(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("delete pipeline is not implemented yet")

    def rebuild(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("rebuild pipeline is not implemented yet")
