from __future__ import annotations

from pathlib import Path


def expand_optional_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    if isinstance(raw, Path):
        return raw.expanduser()
    normalized = raw.strip()
    if not normalized:
        return None
    return Path(normalized).expanduser()


def resolve_local_model_reference(model_name: str, model_path: str | Path | None) -> str:
    expanded = expand_optional_path(model_path)
    if expanded is None:
        return model_name
    return str(resolve_huggingface_snapshot_path(expanded))


def resolve_huggingface_snapshot_path(model_root: str | Path) -> Path:
    path = Path(model_root).expanduser()
    if _looks_like_model_dir(path):
        return path

    main_ref = path / "refs" / "main"
    if main_ref.exists():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = path / "snapshots" / revision
        if _looks_like_model_dir(snapshot):
            return snapshot

    snapshots_root = path / "snapshots"
    if snapshots_root.exists():
        candidates = sorted(
            candidate
            for candidate in snapshots_root.iterdir()
            if candidate.is_dir() and _looks_like_model_dir(candidate)
        )
        if len(candidates) == 1:
            return candidates[0]

    return path


def _looks_like_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "tokenizer_config.json").exists()
