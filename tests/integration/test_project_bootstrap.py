from pathlib import Path


def test_bootstrap_files_exist() -> None:
    required = [
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "importlinter.ini",
        ".env.example",
    ]

    missing = [relative for relative in required if not Path(relative).exists()]
    assert not missing, missing
