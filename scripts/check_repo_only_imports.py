from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "pkp"
ALLOWED_ROOT = SRC_ROOT / "repo"
FORBIDDEN_IMPORTS = {
    "bs4",
    "fitz",
    "openai",
    "PIL",
    "pymupdf",
    "trafilatura",
}


def iter_python_files() -> list[Path]:
    return [path for path in SRC_ROOT.rglob("*.py") if not path.is_relative_to(ALLOWED_ROOT)]


def imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def main() -> None:
    violations: list[str] = []
    for path in iter_python_files():
        imported = imported_top_levels(path)
        blocked = sorted(imported & FORBIDDEN_IMPORTS)
        if blocked:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {', '.join(blocked)}")

    if violations:
        joined = "\n".join(violations)
        raise SystemExit(f"Provider/parser imports must stay inside src/pkp/repo:\n{joined}")


if __name__ == "__main__":
    main()
