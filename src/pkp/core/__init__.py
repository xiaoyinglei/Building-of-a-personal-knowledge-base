from pkp.core.options import QueryOptions
from pkp.core.storage_config import StorageConfig

__all__ = ["QueryOptions", "RAGCore", "StorageConfig"]


def __getattr__(name: str) -> object:
    if name == "RAGCore":
        from pkp.core.rag_core import RAGCore

        return RAGCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
