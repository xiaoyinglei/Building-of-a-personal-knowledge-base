from pkp.query.query import QueryOptions
from pkp.storage import StorageConfig

__all__ = ["QueryOptions", "RAGCore", "StorageConfig"]


def __getattr__(name: str) -> object:
    if name == "RAGCore":
        from pkp.engine import RAGCore

        return RAGCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
