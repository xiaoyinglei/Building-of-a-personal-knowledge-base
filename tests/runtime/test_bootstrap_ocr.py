from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import pkp.bootstrap as bootstrap_module
from pkp.bootstrap import build_runtime_container
from pkp.config import AppSettings


@pytest.mark.skipif(sys.platform != "darwin", reason="default runtime OCR is macOS-specific")
def test_build_runtime_container_uses_real_mac_ocr_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_container(**kwargs: object) -> object:
        captured["ocr_repo"] = kwargs["ocr_repo"]
        return SimpleNamespace()

    monkeypatch.setattr(bootstrap_module, "_build_runtime_container", fake_build_container)

    settings = AppSettings.model_validate(
        {
            "runtime": {
                "data_dir": str(tmp_path / "runtime"),
                "object_store_dir": str(tmp_path / "runtime" / "objects"),
            }
        }
    )

    result = build_runtime_container(settings)

    assert isinstance(result, SimpleNamespace)
    assert captured["ocr_repo"].__class__.__name__ == "OCRMacVisionRepo"
