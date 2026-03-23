from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image

from pkp.repo.interfaces import OcrResult, OcrVisionRepo


class DeterministicOcrVisionRepo(OcrVisionRepo):
    def __init__(self, mapping: dict[str, OcrResult] | None = None) -> None:
        self._mapping = mapping or {}

    def extract(self, image_path: Path) -> OcrResult:
        if image_path.as_posix() in self._mapping:
            return self._mapping[image_path.as_posix()]

        with Image.open(image_path) as image:
            semantics = f"{image.width}x{image.height} {image.mode} image"
        return OcrResult(visible_text="", visual_semantics=semantics, regions=[])


class CallableOcrVisionRepo(OcrVisionRepo):
    def __init__(self, extract_fn: Callable[[Path], OcrResult]) -> None:
        self._extract_fn = extract_fn

    def extract(self, image_path: Path) -> OcrResult:
        return self._extract_fn(image_path)
