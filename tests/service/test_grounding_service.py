from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.assembly import TokenAccountingService, TokenizerContract
import rag.retrieval.grounding_service as grounding_module
from rag.retrieval.grounding_service import GroundingBudgets, GroundingService
from rag.schema.core import AssetRecord, DocumentType, LayoutMetaCacheRecord, SectionRecord, SourceType
from rag.schema.query import EvidenceItem, GroundingTarget
from rag.utils.text import DEFAULT_TOKENIZER_FALLBACK_MODEL


def _token_accounting() -> TokenAccountingService:
    return TokenAccountingService(
        TokenizerContract(
            embedding_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            chunking_tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
        )
    )


@dataclass
class _MetadataRepo:
    sections: dict[int, SectionRecord] = field(default_factory=dict)
    assets: dict[int, AssetRecord] = field(default_factory=dict)
    layouts: dict[int, LayoutMetaCacheRecord] = field(default_factory=dict)

    def get_section(self, section_id: int) -> SectionRecord | None:
        return self.sections.get(section_id)

    def list_sections(self, *, doc_id: int | None = None, source_id: int | None = None) -> list[SectionRecord]:
        del source_id
        sections = list(self.sections.values())
        if doc_id is not None:
            sections = [section for section in sections if section.doc_id == doc_id]
        return sorted(sections, key=lambda item: (item.order_index, item.section_id))

    def get_asset(self, asset_id: int) -> AssetRecord | None:
        return self.assets.get(asset_id)

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]:
        del source_id
        assets = list(self.assets.values())
        if doc_id is not None:
            assets = [asset for asset in assets if asset.doc_id == doc_id]
        if section_id is not None:
            assets = [asset for asset in assets if asset.section_id == section_id]
        return sorted(assets, key=lambda item: (item.page_no, item.asset_id))

    def get_layout_meta_cache(self, doc_id: int) -> LayoutMetaCacheRecord | None:
        return self.layouts.get(doc_id)


@dataclass
class _ObjectStore:
    payloads: dict[str, bytes]
    range_calls: list[tuple[str, int, int]] = field(default_factory=list)
    read_calls: list[str] = field(default_factory=list)

    def read_byte_range(self, key: str, start: int, end: int) -> bytes:
        self.range_calls.append((key, start, end))
        return self.payloads[key][start:end]

    def read_bytes(self, key: str) -> bytes:
        self.read_calls.append(key)
        raise AssertionError("L5 should not fall back to full-file reads when byte ranges are available")


@dataclass
class _RerankBinding:
    ranking: list[int]

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        del query, candidates
        return list(self.ranking)


@dataclass
class _TokenAccountingStub:
    chunks: list[str]

    def count(self, text: str) -> int:
        return len(text.split())

    def clip(self, text: str, budget: int, *, add_ellipsis: bool = False) -> str:
        words = text.split()
        clipped = " ".join(words[:budget])
        if add_ellipsis and len(words) > budget:
            return f"{clipped} ..."
        return clipped

    def chunk_text(self, text: str, *, chunk_token_size: int, chunk_overlap_tokens: int) -> list[str]:
        del text, chunk_token_size, chunk_overlap_tokens
        return list(self.chunks)


def test_grounding_service_reads_section_byte_range_and_includes_neighbor_asset() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: SectionRecord(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture", "Alpha"],
                order_index=1,
                page_start=2,
                page_end=2,
                byte_range_start=0,
                byte_range_end=23,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        },
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=7,
                asset_type="table",
                page_no=2,
                caption="Alpha capacity table",
                content_hash="asset-hash",
                storage_key="asset-11.txt",
            )
        },
    )
    object_store = _ObjectStore(
        payloads={
            "doc-42.txt": b"Alpha architecture body.",
            "asset-11.txt": b"unused",
        }
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=object_store,
        token_accounting=_token_accounting(),
    )

    grounded = service.ground(
        query="What is Alpha architecture?",
        evidence=[
            EvidenceItem(
                chunk_id="summary:section_summary:7",
                doc_id="42",
                source_id="9",
                citation_anchor="Architecture / Alpha",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(
                    kind="section",
                    doc_id="42",
                    source_id="9",
                    section_id="7",
                    page_start=2,
                    page_end=2,
                    section_path=["Architecture", "Alpha"],
                    raw_locator={"summary_item_id": "7"},
                ),
            )
        ],
    )

    assert object_store.range_calls == [("doc-42.txt", 0, 23)]
    assert object_store.read_calls == []
    assert grounded[0].text == "Alpha architecture body"
    assert grounded[0].grounding_target is not None
    assert grounded[0].grounding_target.section_id == "7"
    assert any(item.special_chunk_type == "table" for item in grounded)
    assert any("Alpha capacity table" in item.text for item in grounded)


def test_grounding_service_uses_layout_meta_cache_for_geometric_neighbor_assets() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: SectionRecord(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture", "Alpha"],
                order_index=1,
                page_start=2,
                page_end=2,
                byte_range_start=0,
                byte_range_end=23,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        },
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=None,
                asset_type="table",
                element_ref="asset-table-1",
                page_no=2,
                caption="Alpha layout table",
                content_hash="asset-hash",
                storage_key="asset-11.txt",
            )
        },
        layouts={
            42: LayoutMetaCacheRecord(
                source_id=9,
                doc_id=42,
                content_hash="hash-42",
                layout_json={
                    "elements": [
                        {
                            "element_id": "text-1",
                            "kind": "text",
                            "toc_path": ["Architecture", "Alpha"],
                            "page_no": 2,
                            "bbox": [0, 0, 100, 80],
                        },
                        {
                            "element_id": "asset-table-1",
                            "kind": "table",
                            "toc_path": ["Architecture", "Alpha"],
                            "page_no": 2,
                            "bbox": [0, 84, 120, 140],
                        },
                    ]
                },
            )
        },
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={"doc-42.txt": b"Alpha architecture body."}),
        token_accounting=_token_accounting(),
    )

    grounded = service.ground(
        query="What is Alpha architecture?",
        evidence=[
            EvidenceItem(
                chunk_id="summary:section_summary:7",
                doc_id="42",
                source_id="9",
                citation_anchor="Architecture / Alpha",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id="42", source_id="9", section_id="7"),
            )
        ],
    )

    assert any(item.special_chunk_type == "table" for item in grounded)
    assert any("Alpha layout table" in item.text for item in grounded)


def test_grounding_service_enforces_input_and_output_budgets() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: SectionRecord(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture"],
                order_index=1,
                byte_range_start=0,
                byte_range_end=200,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            ),
            8: SectionRecord(
                section_id=8,
                doc_id=42,
                source_id=9,
                toc_path=["Operations"],
                order_index=2,
                byte_range_start=0,
                byte_range_end=200,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash-2",
            ),
        }
    )
    object_store = _ObjectStore(
        payloads={
            "doc-42.txt": (
                "alpha one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen twenty"
            ).encode("utf-8")
        }
    )
    token_accounting = _token_accounting()
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=object_store,
        token_accounting=token_accounting,
        budgets=GroundingBudgets(
            max_targets_to_read=1,
            max_output_tokens=12,
            local_chunk_tokens=6,
            local_chunk_overlap_tokens=0,
            max_neighbor_assets=0,
        ),
    )

    grounded = service.ground(
        query="alpha architecture",
        evidence=[
            EvidenceItem(
                chunk_id="summary:section_summary:7",
                doc_id="42",
                source_id="9",
                citation_anchor="Architecture",
                text="section summary A",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id="42", section_id="7"),
            ),
            EvidenceItem(
                chunk_id="summary:section_summary:8",
                doc_id="42",
                source_id="9",
                citation_anchor="Operations",
                text="section summary B",
                score=0.89,
                grounding_target=GroundingTarget(kind="section", doc_id="42", section_id="8"),
            ),
        ],
    )

    assert len(object_store.range_calls) == 1
    assert all(item.chunk_id.startswith("grounded:") for item in grounded)
    assert sum(token_accounting.count(item.text) for item in grounded) <= 12


def test_grounding_service_passthrough_when_grounding_target_is_missing() -> None:
    service = GroundingService(
        metadata_repo=_MetadataRepo(),
        object_store=_ObjectStore(payloads={}),
        token_accounting=_token_accounting(),
    )
    evidence = [
        EvidenceItem(
            chunk_id="chunk-1",
            doc_id="42",
            citation_anchor="#a",
            text="existing chunk text",
            score=0.7,
        )
    ]

    grounded = service.ground(query="alpha", evidence=evidence)

    assert grounded == evidence


def test_grounding_service_reuses_single_executor_per_query(monkeypatch) -> None:
    created_executors: list[int] = []

    class _FakeFuture:
        def __init__(self, value: bytes) -> None:
            self._value = value

        def result(self, timeout: float | None = None) -> bytes:
            del timeout
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            created_executors.append(max_workers)

        def __enter__(self) -> _FakeExecutor:
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            del exc_type, exc, tb

        def submit(self, fn, *args):
            return _FakeFuture(fn(*args))

    monkeypatch.setattr(grounding_module, "ThreadPoolExecutor", _FakeExecutor)
    metadata_repo = _MetadataRepo(
        sections={
            7: SectionRecord(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture", "Alpha"],
                order_index=1,
                page_start=2,
                page_end=2,
                byte_range_start=0,
                byte_range_end=23,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        },
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=7,
                asset_type="table",
                page_no=2,
                caption=None,
                content_hash="asset-hash",
                storage_key="asset-11.txt",
            )
        },
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(
            payloads={
                "doc-42.txt": b"Alpha architecture body.",
                "asset-11.txt": b"Alpha table preview",
            }
        ),
        token_accounting=_token_accounting(),
    )

    grounded = service.ground(
        query="What is Alpha architecture?",
        evidence=[
            EvidenceItem(
                chunk_id="summary:section_summary:7",
                doc_id="42",
                source_id="9",
                citation_anchor="Architecture / Alpha",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id="42", source_id="9", section_id="7"),
            )
        ],
    )

    assert grounded
    assert created_executors == [service.budgets.max_parallel_reads]


def test_grounding_service_uses_rerank_binding_for_local_evidence_scoring() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: SectionRecord(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture"],
                order_index=1,
                byte_range_start=0,
                byte_range_end=64,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        }
    )
    payload = "Background filler. Alpha engine handles ingestion. Miscellaneous notes."
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={"doc-42.txt": payload.encode("utf-8")}),
        token_accounting=_TokenAccountingStub(
            chunks=["Background filler", "Alpha engine handles", "Miscellaneous notes"]
        ),
        budgets=GroundingBudgets(local_chunk_tokens=3, local_chunk_overlap_tokens=0, max_neighbor_assets=0),
        rerank_binding=_RerankBinding(ranking=[1, 0, 2]),
    )

    grounded = service.ground(
        query="What handles ingestion?",
        evidence=[
            EvidenceItem(
                chunk_id="summary:section_summary:7",
                doc_id="42",
                source_id="9",
                citation_anchor="Architecture",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id="42", section_id="7"),
            )
        ],
    )

    assert grounded[0].text == "Alpha engine handles"
