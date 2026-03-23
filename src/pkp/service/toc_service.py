from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from pkp.repo.parse._util import slugify


class TOCService:
    def normalize_path(self, headings: Sequence[str]) -> list[str]:
        return [heading.strip() for heading in headings if heading.strip()]

    def stable_anchor(
        self,
        location: str,
        toc_path: Sequence[str],
        order_index: int,
        *,
        page_range: tuple[int, int] | None = None,
        anchor_hint: str | None = None,
    ) -> str:
        normalized_path = self.normalize_path(toc_path)
        prefix = self._anchor_prefix(location)
        if page_range is not None:
            hint = anchor_hint or f"page-{page_range[0]}"
            basis = f"{prefix}|{'/'.join(normalized_path)}|{order_index}|{page_range}"
            return f"{prefix}#{hint}-{self._digest(basis)}"

        hint = anchor_hint or "-".join(slugify(part) for part in normalized_path)
        basis = f"{prefix}|{'/'.join(normalized_path)}|{order_index}"
        return f"{prefix}#{hint}-{self._digest(basis)}"

    @staticmethod
    def _anchor_prefix(location: str) -> str:
        path = Path(location)
        return location if not path.is_absolute() else path.name

    @staticmethod
    def _digest(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()[:10]
