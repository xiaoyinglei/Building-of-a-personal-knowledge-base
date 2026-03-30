from __future__ import annotations

from hashlib import sha256
from pathlib import Path


class FileObjectStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, content: bytes, *, suffix: str = "") -> str:
        digest = sha256(content).hexdigest()
        safe_suffix = suffix if suffix.startswith(".") or not suffix else f".{suffix}"
        key = f"{digest}{safe_suffix}"
        path = self._root / key
        if not path.exists():
            path.write_bytes(content)
        return key

    def read_bytes(self, key: str) -> bytes:
        return (self._root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self._root / key).exists()

    def path_for_key(self, key: str) -> Path:
        return self._root / key
