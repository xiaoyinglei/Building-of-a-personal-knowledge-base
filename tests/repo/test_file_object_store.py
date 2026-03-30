from pathlib import Path

from pkp.storage._repo.file_object_store import FileObjectStore


def test_file_object_store_round_trips_bytes(tmp_path: Path) -> None:
    store = FileObjectStore(tmp_path / "objects")

    key = store.put_bytes(b"hello world", suffix=".txt")

    assert store.exists(key)
    assert store.read_bytes(key) == b"hello world"
    assert (tmp_path / "objects" / key).exists()
