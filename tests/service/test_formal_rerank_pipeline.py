from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from pkp.rerank.cross_encoder import CrossEncoderConfig, ProviderBackedCrossEncoder
from pkp.rerank.evaluation import RerankEvaluator
from pkp.rerank.models import RerankCandidate, RerankEvaluationCase, RerankRequest
from pkp.rerank.pipeline import FormalRerankService, RerankPipelineConfig
from pkp.rerank.training import ExportFormat, TrainingSampleExporter
from pkp.types.query import QueryUnderstanding


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.calls: list[list[str]] = []
        self.backend_name = "fake-cross-encoder"
        self.model_name = "fake-model"

    def score(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        config: CrossEncoderConfig,
    ) -> list[float]:
        del query, config
        self.calls.append([candidate.chunk_id for candidate in candidates])
        return list(self._scores[: len(candidates)])


class FakeProvider:
    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        del query
        return sorted(range(len(candidates)), key=lambda index: candidates[index], reverse=True)


def _query_analysis(**updates: object) -> QueryUnderstanding:
    payload = {
        "intent": "structure_lookup",
        "query_type": "structure",
        "needs_dense": True,
        "needs_sparse": True,
        "needs_special": False,
        "needs_structure": True,
        "needs_metadata": False,
        "structure_constraints": {"preferred_section_terms": ["系统架构"], "prefer_heading_match": True},
        "metadata_filters": {},
        "special_targets": [],
        "confidence": 0.9,
    }
    payload.update(updates)
    return QueryUnderstanding.model_validate(payload)


def test_formal_rerank_pipeline_returns_structured_ranked_items() -> None:
    service = FormalRerankService(
        cross_encoder=FakeCrossEncoder([0.15, 0.92]),
        config=RerankPipelineConfig(top_k=8, top_n=3),
    )
    request = RerankRequest(
        query="系统架构分为哪几层？",
        query_analysis=_query_analysis(),
        candidate_list=[
            RerankCandidate(
                chunk_id="chunk-general",
                doc_id="doc-a",
                parent_id="parent-general",
                text="系统由多个模块组成。",
                chunk_type="child",
                section_path=["项目介绍"],
                heading_text="项目介绍",
                page_start=1,
                page_end=1,
                retrieval_channels=["vector"],
                dense_score=0.88,
                sparse_score=0.34,
                special_score=0.0,
                fusion_score=0.62,
                rrf_score=0.62,
                unified_rank=1,
                metadata={"source_type": "markdown", "order_index": "1"},
            ),
            RerankCandidate(
                chunk_id="chunk-arch",
                doc_id="doc-a",
                parent_id="parent-arch",
                text="系统架构分为接入层、检索层、生成层。",
                chunk_type="child",
                section_path=["系统架构"],
                heading_text="系统架构",
                page_start=3,
                page_end=3,
                retrieval_channels=["section", "vector"],
                dense_score=0.71,
                sparse_score=0.45,
                special_score=0.0,
                fusion_score=0.59,
                rrf_score=0.59,
                unified_rank=2,
                metadata={"source_type": "markdown", "order_index": "8"},
            ),
        ],
    )

    response = service.run(request)

    assert [item.chunk_id for item in response.items] == ["chunk-arch", "chunk-general"]
    assert response.items[0].rank_before == 2
    assert response.items[0].rank_after == 1
    assert response.items[0].rerank_score > response.items[1].rerank_score
    assert response.items[0].final_score > response.items[1].final_score
    assert response.items[0].channel_summary == ["section", "vector"]
    assert response.items[0].feature_summary["cross_encoder_score"] == 0.92
    assert float(response.items[0].feature_summary["section_path_hit"]) > 0
    assert response.items[0].metadata["source_type"] == "markdown"


def test_formal_rerank_pipeline_deduplicates_same_parent_and_preserves_special_diversity() -> None:
    service = FormalRerankService(
        cross_encoder=FakeCrossEncoder([0.94, 0.91, 0.55]),
        config=RerankPipelineConfig(top_k=8, top_n=3, max_children_per_parent=1, preserve_special_slots=1),
    )
    request = RerankRequest(
        query="表格里告警是多少？",
        query_analysis=_query_analysis(
            intent="special_lookup",
            query_type="table",
            needs_special=True,
            needs_structure=False,
            structure_constraints={},
            special_targets=["table"],
        ),
        candidate_list=[
            RerankCandidate(
                chunk_id="child-a1",
                doc_id="doc-a",
                parent_id="parent-a",
                text="月返专项工作中发现了多个异常。",
                chunk_type="child",
                section_path=["专项工作"],
                heading_text="专项工作",
                page_start=2,
                page_end=2,
                retrieval_channels=["vector"],
                dense_score=0.82,
                sparse_score=0.41,
                special_score=0.0,
                fusion_score=0.61,
                rrf_score=0.61,
                unified_rank=1,
                metadata={"source_type": "docx", "order_index": "11"},
            ),
            RerankCandidate(
                chunk_id="child-a2",
                doc_id="doc-a",
                parent_id="parent-a",
                text="月返专项工作中发现了多个异常，需继续跟进。",
                chunk_type="child",
                section_path=["专项工作"],
                heading_text="专项工作",
                page_start=2,
                page_end=2,
                retrieval_channels=["sparse"],
                dense_score=0.79,
                sparse_score=0.46,
                special_score=0.0,
                fusion_score=0.6,
                rrf_score=0.6,
                unified_rank=2,
                metadata={"source_type": "docx", "order_index": "12"},
            ),
            RerankCandidate(
                chunk_id="table-1",
                doc_id="doc-a",
                parent_id="parent-table",
                text="| 指标 | 数值 |\n| 告警 | 7 |",
                chunk_type="table",
                section_path=["专项工作", "统计表"],
                heading_text="统计表",
                page_start=3,
                page_end=3,
                retrieval_channels=["special"],
                dense_score=0.22,
                sparse_score=0.3,
                special_score=0.94,
                fusion_score=0.66,
                rrf_score=0.66,
                unified_rank=3,
                metadata={"source_type": "docx", "page_no": "3"},
            ),
        ],
    )

    response = service.run(request)

    assert {item.chunk_id for item in response.items} == {"child-a1", "table-1"}
    assert response.items[0].chunk_id == "table-1"
    assert response.dropped_items[0].chunk_id == "child-a2"
    assert response.dropped_items[0].drop_reason == "same_parent_redundant"


def test_rerank_evaluator_compares_stages() -> None:
    evaluator = RerankEvaluator()
    cases = [
        RerankEvaluationCase(
            query_id="q-1",
            query="系统架构分为哪几层？",
            query_type="structure",
            source_type="markdown",
            expected_chunk_ids=["chunk-arch"],
            expected_chunk_type="child",
        )
    ]

    summary = evaluator.evaluate(
        cases=cases,
        stage_rankings={
            "retrieval_only": {"q-1": ["chunk-general", "chunk-arch"]},
            "rerank_final": {"q-1": ["chunk-arch", "chunk-general"]},
        },
    )

    assert summary.by_stage["retrieval_only"].hit_at_1 == 0.0
    assert summary.by_stage["rerank_final"].hit_at_1 == 1.0
    assert summary.by_stage["rerank_final"].mrr == 1.0


def test_training_sample_exporter_outputs_pairwise_samples_with_hard_negatives() -> None:
    service = FormalRerankService(
        cross_encoder=FakeCrossEncoder([0.21, 0.88, 0.51]),
        config=RerankPipelineConfig(top_k=8, top_n=3),
    )
    response = service.run(
        RerankRequest(
            query="系统架构分为哪几层？",
            query_analysis=_query_analysis(),
            candidate_list=[
                RerankCandidate(
                    chunk_id="chunk-general",
                    doc_id="doc-a",
                    parent_id="parent-a",
                    text="系统由多个模块组成。",
                    chunk_type="child",
                    section_path=["项目介绍"],
                    heading_text="项目介绍",
                    page_start=1,
                    page_end=1,
                    retrieval_channels=["vector"],
                    dense_score=0.83,
                    sparse_score=0.31,
                    special_score=0.0,
                    fusion_score=0.58,
                    rrf_score=0.58,
                    unified_rank=1,
                    metadata={"source_type": "markdown"},
                ),
                RerankCandidate(
                    chunk_id="chunk-arch",
                    doc_id="doc-a",
                    parent_id="parent-arch",
                    text="系统架构分为接入层、检索层、生成层。",
                    chunk_type="child",
                    section_path=["系统架构"],
                    heading_text="系统架构",
                    page_start=3,
                    page_end=3,
                    retrieval_channels=["section", "vector"],
                    dense_score=0.76,
                    sparse_score=0.4,
                    special_score=0.0,
                    fusion_score=0.57,
                    rrf_score=0.57,
                    unified_rank=2,
                    metadata={"source_type": "markdown"},
                ),
                RerankCandidate(
                    chunk_id="chunk-cli",
                    doc_id="doc-a",
                    parent_id="parent-cli",
                    text="uv run pkp query --mode fast --query \"系统架构分为哪几层？\"",
                    chunk_type="child",
                    section_path=["查询"],
                    heading_text="查询",
                    page_start=5,
                    page_end=5,
                    retrieval_channels=["sparse"],
                    dense_score=0.33,
                    sparse_score=0.48,
                    special_score=0.0,
                    fusion_score=0.42,
                    rrf_score=0.42,
                    unified_rank=3,
                    metadata={"source_type": "markdown"},
                ),
            ],
        )
    )
    exporter = TrainingSampleExporter()

    samples = exporter.export(
        response=response,
        positive_chunk_ids={"chunk-arch"},
        export_format=ExportFormat.PAIRWISE,
    )

    assert len(samples) == 2
    assert all(sample.query == "系统架构分为哪几层？" for sample in samples)
    assert all(sample.positive_candidate.chunk_id == "chunk-arch" for sample in samples)
    assert {sample.hard_negative_candidates[0].chunk_id for sample in samples} == {"chunk-general", "chunk-cli"}


def test_provider_backed_cross_encoder_uses_provider_ranking_as_scores(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pkp.rerank.cross_encoder.ProviderBackedCrossEncoder._try_flag_embedding_backend",
        lambda self: None,
    )
    encoder = ProviderBackedCrossEncoder(provider=FakeProvider(), config=CrossEncoderConfig(batch_size=2))

    scores = encoder.score(
        "排序这些候选",
        [
            RerankCandidate(
                chunk_id="a",
                doc_id="doc",
                text="alpha",
                chunk_type="child",
                unified_rank=1,
            ),
            RerankCandidate(
                chunk_id="b",
                doc_id="doc",
                text="beta",
                chunk_type="child",
                unified_rank=2,
            ),
        ],
        config=CrossEncoderConfig(batch_size=2),
    )

    assert encoder.backend_name == "provider_rerank"
    assert scores[1] > scores[0]


def test_provider_backed_cross_encoder_prefers_snapshot_resolved_model_path(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models--BAAI--bge-reranker-v2-m3"
    snapshot = model_root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (model_root / "refs").mkdir(parents=True)
    (model_root / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeFlagReranker:
        def __init__(self, model_name_or_path: str, *, use_fp16: bool) -> None:
            captured["model_name_or_path"] = model_name_or_path
            captured["use_fp16"] = use_fp16

        def compute_score(self, pairs: list[list[str]], *, batch_size: int, max_length: int) -> list[float]:
            captured["pairs"] = pairs
            captured["batch_size"] = batch_size
            captured["max_length"] = max_length
            return [0.88 for _ in pairs]

    monkeypatch.setattr(
        "pkp.rerank.cross_encoder.importlib.import_module",
        lambda _name: SimpleNamespace(FlagReranker=FakeFlagReranker),
    )
    encoder = ProviderBackedCrossEncoder(
        config=CrossEncoderConfig(
            model_name="BAAI/bge-reranker-v2-m3",
            model_path=str(model_root),
            batch_size=2,
            max_length=256,
        )
    )

    scores = encoder.score(
        "本地 BGE rerank 是否可用",
        [
            RerankCandidate(
                chunk_id="chunk-a",
                doc_id="doc-a",
                text="本地 BGE rerank 已经走通。",
                chunk_type="child",
                unified_rank=1,
            )
        ],
    )

    assert scores == [0.88]
    assert encoder.backend_name == "bge_local"
    assert captured["model_name_or_path"] == str(snapshot)
