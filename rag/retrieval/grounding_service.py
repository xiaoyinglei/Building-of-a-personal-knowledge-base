from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from threading import BoundedSemaphore
from typing import Any, Protocol

from rag.assembly import RerankCapabilityBinding, TokenAccountingService, TokenizerContract
from rag.schema.core import AssetRecord, LayoutMetaCacheRecord, SectionRecord
from rag.schema.query import EvidenceItem, GroundingTarget
from rag.utils.text import DEFAULT_TOKENIZER_FALLBACK_MODEL, keyword_overlap, search_terms


def _default_token_accounting() -> TokenAccountingService:
    contract = TokenizerContract(
        embedding_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
        tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
        chunking_tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
    )
    return TokenAccountingService(contract)


@dataclass(frozen=True, slots=True)
class GroundingBudgets:
    max_targets_to_read: int = 3
    max_output_tokens: int = 8_000
    local_chunk_tokens: int = 220
    local_chunk_overlap_tokens: int = 30
    local_chunk_top_k: int = 6
    max_neighbor_assets: int = 2
    read_timeout_seconds: float = 1.5
    max_parallel_reads: int = 2
    max_asset_preview_bytes: int = 4096
    max_document_sections: int = 2


class _GroundingMetadataRepo(Protocol):
    def get_section(self, section_id: int) -> SectionRecord | None: ...

    def list_sections(self, *, doc_id: int | None = None, source_id: int | None = None) -> list[SectionRecord]: ...

    def get_asset(self, asset_id: int) -> AssetRecord | None: ...

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]: ...

    def get_layout_meta_cache(self, doc_id: int) -> LayoutMetaCacheRecord | None: ...


class _RangeReadableObjectStore(Protocol):
    def read_byte_range(self, key: str, start: int, end: int) -> bytes: ...


@dataclass(slots=True)
class _GroundingSession:
    executor: ThreadPoolExecutor
    semaphore: BoundedSemaphore


@dataclass(slots=True)
class GroundingService:
    metadata_repo: object
    object_store: object
    token_accounting: TokenAccountingService = field(default_factory=_default_token_accounting)
    budgets: GroundingBudgets = field(default_factory=GroundingBudgets)
    rerank_binding: RerankCapabilityBinding | object | None = None

    def ground(
        self,
        *,
        query: str,
        evidence: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        grounded_targets = [item for item in evidence if item.grounding_target is not None]
        if not grounded_targets:
            return list(evidence)

        query_terms = search_terms(query)
        grounded_items: list[EvidenceItem] = []
        output_tokens = 0

        with ThreadPoolExecutor(max_workers=max(self.budgets.max_parallel_reads, 1)) as executor:
            session = _GroundingSession(
                executor=executor,
                semaphore=BoundedSemaphore(value=max(self.budgets.max_parallel_reads, 1)),
            )
            for item in grounded_targets[: self.budgets.max_targets_to_read]:
                for grounded in self._ground_item(item, query=query, query_terms=query_terms, session=session):
                    token_count = self.token_accounting.count(grounded.text)
                    if token_count <= 0:
                        continue
                    remaining_budget = self.budgets.max_output_tokens - output_tokens
                    if remaining_budget <= 0:
                        break
                    if token_count > remaining_budget:
                        clipped = self.token_accounting.clip(grounded.text, remaining_budget, add_ellipsis=True)
                        clipped_token_count = self.token_accounting.count(clipped)
                        if clipped_token_count <= 0:
                            break
                        grounded = grounded.model_copy(update={"text": clipped})
                        token_count = clipped_token_count
                    grounded_items.append(grounded)
                    output_tokens += token_count
                    if output_tokens >= self.budgets.max_output_tokens:
                        break
                if output_tokens >= self.budgets.max_output_tokens:
                    break

        return grounded_items or list(evidence[: self.budgets.max_targets_to_read])

    def _ground_item(
        self,
        item: EvidenceItem,
        *,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        target = item.grounding_target
        if target is None:
            return [item]
        if target.kind == "section":
            return self._ground_section_item(item, query=query, query_terms=query_terms, session=session)
        if target.kind == "asset":
            return self._ground_asset_item(item, query=query, query_terms=query_terms, session=session)
        if target.kind == "document":
            return self._ground_document_item(item, query=query, query_terms=query_terms, session=session)
        return [item]

    def _ground_section_item(
        self,
        item: EvidenceItem,
        *,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        target = item.grounding_target
        if target is None:
            return [item]
        section = self._get_section(self._safe_int(target.section_id))
        if section is None:
            return [item]
        raw_text = self._read_section_text(section, session=session)
        local_items = self._section_local_chunks(
            item=item,
            target=target,
            section=section,
            raw_text=raw_text,
            query=query,
            query_terms=query_terms,
        )
        if target.kind == "section":
            local_items.extend(
                self._neighbor_asset_items(item=item, section=section, query=query, query_terms=query_terms, session=session)
            )
        ranked = self._rank_local_items(local_items, query=query, query_terms=query_terms)
        return ranked[: max(self.budgets.local_chunk_top_k + self.budgets.max_neighbor_assets, 1)]

    def _ground_document_item(
        self,
        item: EvidenceItem,
        *,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        target = item.grounding_target
        if target is None:
            return [item]
        list_sections = getattr(self.metadata_repo, "list_sections", None)
        if not callable(list_sections):
            return [item]
        doc_id = self._safe_int(target.doc_id)
        if doc_id is None:
            return [item]
        sections = list_sections(doc_id=doc_id)
        if not sections:
            return [item]
        ranked_sections = sorted(
            sections,
            key=lambda section: (
                -self._section_candidate_overlap(section, query_terms=query_terms),
                section.order_index,
                section.section_id,
            ),
        )
        grounded: list[EvidenceItem] = []
        for section in ranked_sections[: self.budgets.max_document_sections]:
            section_target = target.model_copy(
                update={
                    "kind": "section",
                    "section_id": str(section.section_id),
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "section_path": list(section.toc_path),
                }
            )
            section_item = item.model_copy(
                update={
                    "citation_anchor": " / ".join(section.toc_path) or item.citation_anchor,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "grounding_target": section_target,
                }
            )
            grounded.extend(self._ground_section_item(section_item, query=query, query_terms=query_terms, session=session))
        return grounded or [item]

    def _ground_asset_item(
        self,
        item: EvidenceItem,
        *,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        target = item.grounding_target
        if target is None:
            return [item]
        asset = self._get_asset(self._safe_int(target.asset_id))
        if asset is None:
            return [item]
        text = self._asset_text(asset, session=session).strip()
        if not text:
            return [item]
        grounded = item.model_copy(
            update={
                "chunk_id": f"grounded:asset:{asset.asset_id}",
                "text": text,
                "citation_anchor": item.citation_anchor or f"{asset.asset_type}@p{asset.page_no}",
                "special_chunk_type": asset.asset_type,
                "page_start": asset.page_no,
                "page_end": asset.page_no,
                "grounding_target": GroundingTarget(
                    kind="asset",
                    doc_id=str(asset.doc_id),
                    source_id=str(asset.source_id),
                    section_id=(str(asset.section_id) if asset.section_id is not None else None),
                    asset_id=str(asset.asset_id),
                    page_start=asset.page_no,
                    page_end=asset.page_no,
                    raw_locator={key: str(value) for key, value in asset.raw_locator.items()},
                ),
                "retrieval_channels": [*item.retrieval_channels, "grounding"],
            }
        )
        return self._rank_local_items([grounded], query=query, query_terms=query_terms)

    def _section_local_chunks(
        self,
        *,
        item: EvidenceItem,
        target: GroundingTarget,
        section: SectionRecord,
        raw_text: str,
        query: str,
        query_terms: tuple[str, ...],
    ) -> list[EvidenceItem]:
        text = raw_text.strip()
        if not text:
            return []
        section_path = list(target.section_path or section.toc_path)
        chunks = self.token_accounting.chunk_text(
            text,
            chunk_token_size=self.budgets.local_chunk_tokens,
            chunk_overlap_tokens=self.budgets.local_chunk_overlap_tokens,
        )
        if not chunks:
            chunks = [text]
        local_items: list[EvidenceItem] = []
        for index, chunk_text in enumerate(chunks, start=1):
            normalized = chunk_text.strip().rstrip(".")
            if not normalized:
                continue
            local_items.append(
                item.model_copy(
                    update={
                        "chunk_id": f"grounded:section:{section.section_id}:{index}",
                        "text": normalized,
                        "citation_anchor": " / ".join(section_path) or item.citation_anchor,
                        "section_path": section_path,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        "grounding_target": GroundingTarget(
                            kind="section",
                            doc_id=str(section.doc_id),
                            source_id=str(section.source_id),
                            section_id=str(section.section_id),
                            page_start=section.page_start,
                            page_end=section.page_end,
                            section_path=section_path,
                            raw_locator={key: str(value) for key, value in section.raw_locator.items()},
                        ),
                        "retrieval_channels": [*item.retrieval_channels, "grounding"],
                    }
                )
            )
        return local_items

    def _neighbor_asset_items(
        self,
        *,
        item: EvidenceItem,
        section: SectionRecord,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        list_assets = getattr(self.metadata_repo, "list_assets", None)
        if not callable(list_assets):
            return []
        assets = self._layout_neighbor_assets(section)
        if not assets:
            assets = list_assets(doc_id=section.doc_id, section_id=section.section_id)
        grounded_assets: list[EvidenceItem] = []
        for asset in assets[: self.budgets.max_neighbor_assets]:
            text = self._asset_text(asset, session=session).strip()
            if not text:
                continue
            grounded_assets.append(
                item.model_copy(
                    update={
                        "chunk_id": f"grounded:asset:{asset.asset_id}",
                        "text": text.rstrip("."),
                        "citation_anchor": f"{asset.asset_type}@p{asset.page_no}",
                        "special_chunk_type": asset.asset_type,
                        "page_start": asset.page_no,
                        "page_end": asset.page_no,
                        "grounding_target": GroundingTarget(
                            kind="asset",
                            doc_id=str(asset.doc_id),
                            source_id=str(asset.source_id),
                            section_id=(str(asset.section_id) if asset.section_id is not None else None),
                            asset_id=str(asset.asset_id),
                            page_start=asset.page_no,
                            page_end=asset.page_no,
                            raw_locator={key: str(value) for key, value in asset.raw_locator.items()},
                        ),
                        "retrieval_channels": [*item.retrieval_channels, "grounding"],
                    }
                )
            )
        return self._rank_local_items(grounded_assets, query=query, query_terms=query_terms)

    def _read_section_text(self, section: SectionRecord, *, session: _GroundingSession) -> str:
        key = section.visible_text_key or self._locator_key(section.raw_locator)
        start = section.byte_range_start
        end = section.byte_range_end
        if not key or start is None or end is None or end <= start:
            return ""
        return self._read_range(key, start, end, session=session).decode("utf-8", errors="ignore")

    def _asset_text(self, asset: AssetRecord, *, session: _GroundingSession) -> str:
        if asset.caption and asset.caption.strip():
            return asset.caption.strip()
        key = asset.storage_key or self._locator_key(asset.raw_locator)
        if not key:
            return ""
        preview = self._read_range(key, 0, self.budgets.max_asset_preview_bytes, session=session)
        return preview.decode("utf-8", errors="ignore").strip()

    def _read_range(self, key: str, start: int, end: int, *, session: _GroundingSession) -> bytes:
        if end <= start:
            return b""
        reader = getattr(self.object_store, "read_byte_range", None)
        if not callable(reader):
            return b""
        with session.semaphore:
            future = session.executor.submit(reader, key, start, end)
            try:
                return future.result(timeout=self.budgets.read_timeout_seconds)
            except FuturesTimeoutError:
                return b""

    @staticmethod
    def _locator_key(raw_locator: dict[str, Any]) -> str | None:
        for field_name in ("object_key", "visible_text_key", "storage_key"):
            value = raw_locator.get(field_name)
            if value is None:
                continue
            normalized = str(value).strip()
            if normalized:
                return normalized
        return None

    @staticmethod
    def _safe_int(value: object | None) -> int | None:
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _get_section(self, section_id: int | None) -> SectionRecord | None:
        if section_id is None:
            return None
        getter = getattr(self.metadata_repo, "get_section", None)
        if not callable(getter):
            return None
        return getter(section_id)

    def _get_asset(self, asset_id: int | None) -> AssetRecord | None:
        if asset_id is None:
            return None
        getter = getattr(self.metadata_repo, "get_asset", None)
        if not callable(getter):
            return None
        return getter(asset_id)

    def _get_layout_meta_cache(self, doc_id: int) -> LayoutMetaCacheRecord | None:
        getter = getattr(self.metadata_repo, "get_layout_meta_cache", None)
        if not callable(getter):
            return None
        return getter(doc_id)

    def _layout_neighbor_assets(self, section: SectionRecord) -> list[AssetRecord]:
        layout_cache = self._get_layout_meta_cache(section.doc_id)
        list_assets = getattr(self.metadata_repo, "list_assets", None)
        if layout_cache is None or not callable(list_assets):
            return []
        assets = list_assets(doc_id=section.doc_id)
        if not assets:
            return []
        elements = layout_cache.layout_json.get("elements")
        if not isinstance(elements, list):
            return []
        assets_by_ref = {
            str(asset.element_ref): asset
            for asset in assets
            if asset.element_ref is not None and str(asset.element_ref).strip()
        }
        if not assets_by_ref:
            return []
        section_path = tuple(str(part) for part in section.toc_path)
        text_blocks = [
            element
            for element in elements
            if self._layout_toc_path(element) == section_path
            and self._layout_page_no(element) in self._section_pages(section)
            and not self._layout_is_asset(element)
        ]
        ranked: list[tuple[tuple[int, float, int, int], AssetRecord]] = []
        for element in elements:
            if not self._layout_is_asset(element):
                continue
            asset = assets_by_ref.get(str(element.get("element_id", "")))
            if asset is None:
                continue
            page_no = self._layout_page_no(element)
            if page_no is None:
                continue
            same_path = self._layout_toc_path(element) == section_path
            near_page = page_no in self._section_pages(section)
            if not same_path and not near_page:
                continue
            y_distance = self._layout_vertical_distance(text_blocks, element)
            ranked.append(
                (
                    (
                        0 if same_path else 1,
                        y_distance,
                        abs(page_no - min(self._section_pages(section) or {page_no})),
                        asset.asset_id,
                    ),
                    asset,
                )
            )
        ranked.sort(key=lambda item: item[0])
        return [asset for _score, asset in ranked]

    @staticmethod
    def _layout_is_asset(element: object) -> bool:
        if not isinstance(element, dict):
            return False
        kind = str(element.get("kind", "") or "").strip().lower()
        return kind in {"table", "figure", "image", "chart", "image_summary"}

    @staticmethod
    def _layout_page_no(element: object) -> int | None:
        if not isinstance(element, dict):
            return None
        value = element.get("page_no")
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _layout_toc_path(element: object) -> tuple[str, ...]:
        if not isinstance(element, dict):
            return ()
        toc_path = element.get("toc_path")
        if not isinstance(toc_path, list):
            return ()
        return tuple(str(part) for part in toc_path if str(part).strip())

    @staticmethod
    def _layout_bbox(element: object) -> tuple[float, float, float, float] | None:
        if not isinstance(element, dict):
            return None
        bbox = element.get("bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        return (x1, y1, x2, y2)

    def _layout_vertical_distance(self, text_blocks: list[object], asset_element: object) -> float:
        asset_bbox = self._layout_bbox(asset_element)
        asset_page = self._layout_page_no(asset_element)
        if asset_bbox is None or asset_page is None:
            return float("inf")
        top = asset_bbox[1]
        distances = []
        for text_block in text_blocks:
            if self._layout_page_no(text_block) != asset_page:
                continue
            bbox = self._layout_bbox(text_block)
            if bbox is None:
                continue
            distances.append(abs(top - bbox[3]))
        return min(distances, default=float("inf"))

    @staticmethod
    def _section_pages(section: SectionRecord) -> set[int]:
        if section.page_start is None and section.page_end is None:
            return set()
        start = section.page_start or section.page_end or 0
        end = section.page_end or section.page_start or start
        return {page for page in range(min(start, end), max(start, end) + 1) if page > 0}

    def _rank_local_items(
        self,
        items: list[EvidenceItem],
        *,
        query: str,
        query_terms: tuple[str, ...],
    ) -> list[EvidenceItem]:
        if not items:
            return []
        lexical_scores = {
            item.chunk_id: float(item.score) + 0.05 * keyword_overlap(query_terms, item.text)
            for item in items
        }
        rerank_bonus = self._rerank_bonus(query, items)
        return sorted(
            items,
            key=lambda item: (
                -(lexical_scores[item.chunk_id] + rerank_bonus.get(item.chunk_id, 0.0)),
                item.chunk_id,
            ),
        )

    def _rerank_bonus(self, query: str, items: list[EvidenceItem]) -> dict[str, float]:
        binding = self.rerank_binding
        rerank = getattr(binding, "rerank", None)
        if not callable(rerank) or len(items) <= 1:
            return {}
        try:
            ranking = list(rerank(query, [item.text for item in items]))
        except RuntimeError:
            return {}
        bonuses: dict[str, float] = {}
        max_rank = max(len(ranking), 1)
        for rank, item_index in enumerate(ranking):
            if not isinstance(item_index, int) or item_index < 0 or item_index >= len(items):
                continue
            bonuses[items[item_index].chunk_id] = 0.25 * (max_rank - rank) / max_rank
        return bonuses

    @staticmethod
    def _section_candidate_overlap(section: SectionRecord, *, query_terms: tuple[str, ...]) -> int:
        summary_text = str(section.metadata_json.get("summary_text", "") or "")
        toc_text = " / ".join(section.toc_path)
        return keyword_overlap(query_terms, f"{toc_text} {summary_text}".strip())


__all__ = ["GroundingBudgets", "GroundingService"]
