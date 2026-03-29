from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from pkp.algorithms.retrieval.search_backed_factory import SearchBackedRetrievalFactory
from pkp.eval.embedding_repo import LexicalEmbeddingRepo
from pkp.eval.models import (
    ChunkInspectionSample,
    OfflineEvalFixture,
    OfflineEvalQuestion,
    OfflineEvalQuestionResult,
    OfflineEvalReport,
    OfflineEvalRunResult,
    OfflineEvalSummary,
    QualityAuditSummary,
    RetrievalHit,
)
from pkp.eval.sample_pack import prepare_builtin_eval_pack
from pkp.repo.interfaces import (
    EmbeddingProviderBinding,
    ModelProviderRepo,
    OcrVisionRepo,
)
from pkp.repo.search.sqlite_vector_repo import SQLiteVectorRepo
from pkp.runtime.adapters import InstrumentedReranker
from pkp.runtime.provider_metadata import embedding_space, provider_model, provider_name
from pkp.service.artifact_service import ArtifactService
from pkp.service.evidence_service import CandidateLike, EvidenceService
from pkp.service.graph_expansion_service import GraphExpansionService
from pkp.service.ingest_service import IngestResult, IngestService
from pkp.service.rerank_service import HeuristicRerankService
from pkp.service.retrieval_service import Reranker, RetrievalService
from pkp.service.routing_service import RoutingService
from pkp.service.telemetry_service import TelemetryService
from pkp.types.access import (
    AccessPolicy,
    ExecutionLocationPreference,
    ExternalRetrievalPolicy,
)
from pkp.types.content import Chunk, ChunkRole, SourceType
from pkp.types.text import text_unit_count


class OfflineEvalService:
    def __init__(
        self,
        output_dir: Path,
        *,
        top_k: int = 5,
        min_child_words: int = 5,
        max_child_words: int = 220,
    ) -> None:
        self._output_dir = output_dir
        self._top_k = top_k
        self._min_child_words = min_child_words
        self._max_child_words = max_child_words

    def run_builtin_pack(self) -> OfflineEvalRunResult:
        samples_dir = self._output_dir / "samples"
        pack = prepare_builtin_eval_pack(samples_dir)
        return self._run_eval(
            fixtures=pack.fixtures,
            questions=pack.questions,
            ocr_repo=pack.ocr_repo,
        )

    def run_file(self, *, file_path: Path, questions_path: Path) -> OfflineEvalRunResult:
        fixture_id = f"user_file_{file_path.stem.lower().replace('-', '_')}"
        fixture = OfflineEvalFixture(
            fixture_id=fixture_id,
            filename=file_path.name,
            source_type=self._source_type_for_path(file_path),
            description="User-supplied fixture for offline retrieval evaluation.",
            path=file_path,
        )
        questions = self._load_questions(
            questions_path,
            default_fixture_id=fixture.fixture_id,
        )
        return self._run_eval(
            fixtures=[fixture],
            questions=questions,
            ocr_repo=None,
        )

    def _run_eval(
        self,
        *,
        fixtures: list[OfflineEvalFixture],
        questions: list[OfflineEvalQuestion],
        ocr_repo: OcrVisionRepo | None,
    ) -> OfflineEvalRunResult:
        runtime_dir = self._output_dir / "runtime"
        ingest_service = IngestService.create_in_memory(runtime_dir, ocr_repo=ocr_repo)
        lexical_provider = LexicalEmbeddingRepo()
        ingest_service.embedding_bindings = (
            EmbeddingProviderBinding(
                provider=lexical_provider,
                space=embedding_space(lexical_provider),
                location="runtime",
            ),
        )
        retrieval_service = self._build_retrieval_service(ingest_service)

        results = self._ingest_fixtures(fixtures, ingest_service)
        quality_audit = self._build_quality_audit(results)
        question_results = self._evaluate_questions(
            questions=questions,
            fixtures=fixtures,
            results=results,
            ingest_service=ingest_service,
            retrieval_service=retrieval_service,
        )
        return self._write_report(
            fixtures=fixtures,
            question_results=question_results,
            quality_audit=quality_audit,
            binding=ingest_service.embedding_bindings[0],
        )

    def _write_report(
        self,
        *,
        fixtures: list[OfflineEvalFixture],
        question_results: list[OfflineEvalQuestionResult],
        quality_audit: QualityAuditSummary,
        binding: EmbeddingProviderBinding,
    ) -> OfflineEvalRunResult:
        report = OfflineEvalReport(
            summary=self._build_summary(
                binding,
                fixtures=fixtures,
                question_results=question_results,
            ),
            fixtures=fixtures,
            quality_audit=quality_audit,
            question_results=question_results,
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = self._output_dir / "report.json"
        report_markdown_path = self._output_dir / "report.md"
        report_json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        report_markdown_path.write_text(
            self._render_markdown_report(report),
            encoding="utf-8",
        )

        return OfflineEvalRunResult(
            output_dir=self._output_dir,
            report_json_path=report_json_path,
            report_markdown_path=report_markdown_path,
            report=report,
        )

    def _build_retrieval_service(self, ingest_service: IngestService) -> RetrievalService:
        factory = SearchBackedRetrievalFactory(
            metadata_repo=ingest_service.metadata_repo,
            fts_repo=ingest_service.fts_repo,
            graph_repo=ingest_service.graph_repo,
        )
        vector_retriever = factory.vector_retriever_from_repo(
            cast(SQLiteVectorRepo, ingest_service.vector_repo),
            ingest_service.embedding_bindings,
            default_preference=ExecutionLocationPreference.LOCAL_ONLY,
        )
        standard_retriever = cast(
            Callable[[str, list[str]], Sequence[CandidateLike]],
            factory.full_text_retriever,
        )
        instrumented_vector_retriever = cast(
            Callable[[str, list[str]], Sequence[CandidateLike]],
            vector_retriever,
        )
        local_retriever = cast(
            Callable[[str, list[str]], Sequence[CandidateLike]],
            factory.local_retriever_from_repo(
                cast(SQLiteVectorRepo, ingest_service.vector_repo),
                ingest_service.embedding_bindings,
                default_preference=ExecutionLocationPreference.LOCAL_ONLY,
            ),
        )
        global_retriever = cast(
            Callable[[str, list[str]], Sequence[CandidateLike]],
            factory.global_retriever_from_repo(
                cast(SQLiteVectorRepo, ingest_service.vector_repo),
                ingest_service.embedding_bindings,
                default_preference=ExecutionLocationPreference.LOCAL_ONLY,
            ),
        )
        section_retriever = cast(
            Callable[[str, list[str]], Sequence[CandidateLike]],
            factory.section_retriever,
        )
        special_retriever = cast(
            Callable[[str, list[str]], Sequence[CandidateLike]],
            factory.special_retriever_from_repo(
                cast(SQLiteVectorRepo, ingest_service.vector_repo),
                ingest_service.embedding_bindings,
                default_preference=ExecutionLocationPreference.LOCAL_ONLY,
            ),
        )
        metadata_retriever = cast(
            Callable[[str, list[str]], Sequence[CandidateLike]],
            factory.metadata_retriever,
        )
        graph_expander = cast(
            Callable[[str, list[str], list[CandidateLike]], Sequence[CandidateLike]],
            factory.graph_expander,
        )
        instrumented_reranker = InstrumentedReranker(HeuristicRerankService())
        return RetrievalService(
            full_text_retriever=standard_retriever,
            vector_retriever=instrumented_vector_retriever,
            local_retriever=local_retriever,
            global_retriever=global_retriever,
            section_retriever=section_retriever,
            special_retriever=special_retriever,
            metadata_retriever=metadata_retriever,
            graph_expander=graph_expander,
            web_retriever=cast(
                Callable[[str, list[str]], Sequence[CandidateLike]],
                lambda _query, _scope: [],
            ),
            reranker=cast(Reranker, instrumented_reranker),
            routing_service=RoutingService(),
            evidence_service=EvidenceService(),
            graph_expansion_service=GraphExpansionService(),
            artifact_service=ArtifactService(),
            telemetry_service=TelemetryService.create_in_memory(),
        )

    @staticmethod
    def _ingest_fixtures(
        fixtures: list[OfflineEvalFixture],
        ingest_service: IngestService,
    ) -> dict[str, IngestResult]:
        results: dict[str, IngestResult] = {}
        for fixture in fixtures:
            results[fixture.fixture_id] = ingest_service.ingest_file(
                location=str(fixture.path),
                file_path=fixture.path,
                owner="offline-eval",
                access_policy=AccessPolicy(
                    external_retrieval=ExternalRetrievalPolicy.DENY,
                ),
            )
        return results

    def _build_quality_audit(self, results: dict[str, IngestResult]) -> QualityAuditSummary:
        duplicate_count = 0
        blank_chunks = 0
        too_short_child_chunks = 0
        too_long_child_chunks = 0
        missing_metadata_chunks = 0
        total_parent_chunks = 0
        total_child_chunks = 0
        total_special_chunks = 0
        seen_searchable: set[tuple[str, str]] = set()
        inspection_samples: list[ChunkInspectionSample] = []

        for fixture_id, result in results.items():
            processing = result.processing
            if processing is None:
                continue
            fixture_name = Path(result.source.location).name
            parent_map = {chunk.chunk_id: chunk for chunk in processing.parent_chunks}
            total_parent_chunks += len(processing.parent_chunks)
            total_child_chunks += len(processing.child_chunks)
            total_special_chunks += len(processing.special_chunks)

            for chunk in [*processing.parent_chunks, *processing.child_chunks, *processing.special_chunks]:
                normalized_text = self._normalize_text(chunk.text)
                if not normalized_text:
                    blank_chunks += 1
                if chunk.chunk_role is ChunkRole.CHILD:
                    word_count = text_unit_count(chunk.text)
                    if word_count < self._min_child_words:
                        too_short_child_chunks += 1
                    if word_count > self._max_child_words:
                        too_long_child_chunks += 1
                if chunk.chunk_role is not ChunkRole.PARENT:
                    key = (chunk.doc_id, normalized_text)
                    if normalized_text in {"", "|"}:
                        pass
                    elif key in seen_searchable:
                        duplicate_count += 1
                    else:
                        seen_searchable.add(key)
                if self._missing_metadata_fields(chunk):
                    missing_metadata_chunks += 1

            sample_candidates = [
                *processing.parent_chunks[:1],
                *processing.child_chunks[:1],
                *processing.special_chunks[:2],
            ]
            for chunk in sample_candidates:
                parent_text = None
                if chunk.parent_chunk_id is not None:
                    parent_chunk = parent_map.get(chunk.parent_chunk_id)
                    parent_text = None if parent_chunk is None else parent_chunk.text
                inspection_samples.append(
                    ChunkInspectionSample(
                        fixture_id=fixture_id,
                        filename=fixture_name,
                        chunk_id=chunk.chunk_id,
                        chunk_role=chunk.chunk_role,
                        special_chunk_type=chunk.special_chunk_type,
                        text=chunk.text,
                        citation_anchor=chunk.citation_anchor,
                        parent_text=parent_text,
                        metadata=chunk.metadata,
                    )
                )

        return QualityAuditSummary(
            total_parent_chunks=total_parent_chunks,
            total_child_chunks=total_child_chunks,
            total_special_chunks=total_special_chunks,
            duplicate_searchable_chunks=duplicate_count,
            blank_chunks=blank_chunks,
            too_short_child_chunks=too_short_child_chunks,
            too_long_child_chunks=too_long_child_chunks,
            missing_metadata_chunks=missing_metadata_chunks,
            inspection_samples=inspection_samples,
        )

    def _evaluate_questions(
        self,
        *,
        questions: list[OfflineEvalQuestion],
        fixtures: list[OfflineEvalFixture],
        results: dict[str, IngestResult],
        ingest_service: IngestService,
        retrieval_service: RetrievalService,
    ) -> list[OfflineEvalQuestionResult]:
        del fixtures
        question_results: list[OfflineEvalQuestionResult] = []
        vector_repo = cast(SQLiteVectorRepo, ingest_service.vector_repo)
        binding = ingest_service.embedding_bindings[0]
        provider = cast(ModelProviderRepo, binding.provider)
        metadata_repo = ingest_service.metadata_repo

        for question in questions:
            result = results[question.fixture_id]
            processing = result.processing
            if processing is None:
                raise RuntimeError(f"missing processing package for {question.fixture_id}")
            searchable_chunks = [*processing.child_chunks, *processing.special_chunks]
            parent_chunks = {chunk.chunk_id: chunk for chunk in processing.parent_chunks}
            all_chunks = [*processing.parent_chunks, *searchable_chunks]
            corpus_has_expected_answer = any(
                self._term_count(chunk.text, question.expected_terms)
                >= question.min_expected_terms
                for chunk in all_chunks
            )
            scope = [result.document.doc_id] if question.scope_to_fixture else []

            vector_hits: list[RetrievalHit] = []
            query_vector = provider.embed([question.question])[0]
            for index, vector_record in enumerate(
                vector_repo.search(
                    query_vector,
                    limit=question.top_k or self._top_k,
                    doc_ids=scope or None,
                    embedding_space=binding.space,
                ),
                start=1,
            ):
                chunk = metadata_repo.get_chunk(vector_record.chunk_id)
                if chunk is None:
                    continue
                vector_hits.append(
                    self._build_hit(
                        retrieval_kind="vector",
                        rank=index,
                        chunk=chunk,
                        parent_chunks=parent_chunks,
                        expected=question,
                        score=float(vector_record.score),
                    )
                )

            fts_hits: list[RetrievalHit] = []
            for index, fts_record in enumerate(
                ingest_service.fts_repo.search(
                    question.question,
                    limit=question.top_k or self._top_k,
                    doc_ids=scope or None,
                ),
                start=1,
            ):
                chunk = metadata_repo.get_chunk(fts_record.chunk_id)
                if chunk is None:
                    continue
                fts_hits.append(
                    self._build_hit(
                        retrieval_kind="fts",
                        rank=index,
                        chunk=chunk,
                        parent_chunks=parent_chunks,
                        expected=question,
                        score=float(fts_record.score),
                    )
                )

            retrieval_result = retrieval_service.retrieve(
                question.question,
                access_policy=AccessPolicy(
                    external_retrieval=ExternalRetrievalPolicy.DENY,
                ),
                source_scope=scope,
                execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
            )
            runtime_hits: list[RetrievalHit] = []
            runtime_evidence = retrieval_result.evidence.all[: question.top_k or self._top_k]
            for index, evidence in enumerate(runtime_evidence, start=1):
                chunk = metadata_repo.get_chunk(evidence.chunk_id)
                if chunk is None:
                    continue
                runtime_hits.append(
                    self._build_hit(
                        retrieval_kind="runtime",
                        rank=index,
                        chunk=chunk,
                        parent_chunks=parent_chunks,
                        expected=question,
                        score=float(evidence.score),
                    )
                )

            vector_hit = any(hit.is_expected_hit for hit in vector_hits)
            fts_hit = any(hit.is_expected_hit for hit in fts_hits)
            runtime_hit = any(hit.is_expected_hit for hit in runtime_hits)
            parent_backfill_improves = any(
                hit.parent_matched_term_count > hit.matched_term_count
                for hit in [*vector_hits, *fts_hits, *runtime_hits]
                if hit.parent_text_preview is not None
            )
            likely_issue = self._classify_issue(
                corpus_has_expected_answer=corpus_has_expected_answer,
                vector_hit=vector_hit,
                fts_hit=fts_hit,
                runtime_hit=runtime_hit,
                parent_backfill_improves=parent_backfill_improves,
            )

            question_results.append(
                OfflineEvalQuestionResult(
                    question_id=question.question_id,
                    fixture_id=question.fixture_id,
                    question=question.question,
                    category=question.category,
                    expected_terms=question.expected_terms,
                    corpus_has_expected_answer=corpus_has_expected_answer,
                    likely_issue=likely_issue,
                    vector_hit=vector_hit,
                    fts_hit=fts_hit,
                    runtime_hit=runtime_hit,
                    parent_backfill_improves=parent_backfill_improves,
                    vector_top_k=vector_hits,
                    fts_top_k=fts_hits,
                    runtime_top_k=runtime_hits,
                )
            )

        return question_results

    def _build_hit(
        self,
        *,
        retrieval_kind: str,
        rank: int,
        chunk: Chunk,
        parent_chunks: dict[str, Chunk],
        expected: OfflineEvalQuestion,
        score: float,
    ) -> RetrievalHit:
        matched_terms = self._matched_terms(chunk.text, expected.expected_terms)
        parent_chunk = (
            None
            if chunk.parent_chunk_id is None
            else parent_chunks.get(chunk.parent_chunk_id)
        )
        parent_matched_terms = (
            []
            if parent_chunk is None
            else self._matched_terms(parent_chunk.text, expected.expected_terms)
        )
        direct_ok = self._hit_matches_expectation(
            chunk=chunk,
            matched_term_count=len(matched_terms),
            expected=expected,
        )
        parent_ok = False
        if expected.expect_parent_backfill and parent_chunk is not None:
            parent_ok = len(parent_matched_terms) >= expected.min_expected_terms
        return RetrievalHit(
            retrieval_kind=retrieval_kind,
            rank=rank,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            score=score,
            chunk_role=chunk.chunk_role,
            special_chunk_type=chunk.special_chunk_type,
            citation_anchor=chunk.citation_anchor,
            matched_terms=matched_terms,
            matched_term_count=len(matched_terms),
            text_preview=self._preview(chunk.text),
            parent_chunk_id=chunk.parent_chunk_id,
            parent_matched_terms=parent_matched_terms,
            parent_matched_term_count=len(parent_matched_terms),
            parent_text_preview=None if parent_chunk is None else self._preview(parent_chunk.text),
            is_expected_hit=direct_ok or parent_ok,
        )

    @staticmethod
    def _hit_matches_expectation(
        *,
        chunk: Chunk,
        matched_term_count: int,
        expected: OfflineEvalQuestion,
    ) -> bool:
        if matched_term_count < expected.min_expected_terms:
            return False
        if (
            expected.expected_chunk_role is not None
            and chunk.chunk_role is not expected.expected_chunk_role
        ):
            return False
        if (
            expected.expected_special_chunk_type is not None
            and chunk.special_chunk_type != expected.expected_special_chunk_type
        ):
            return False
        return True

    @staticmethod
    def _build_summary(
        binding: EmbeddingProviderBinding,
        *,
        fixtures: list[OfflineEvalFixture],
        question_results: list[OfflineEvalQuestionResult],
    ) -> OfflineEvalSummary:
        total_questions = len(question_results) or 1
        return OfflineEvalSummary(
            embedding_provider=provider_name(binding.provider),
            embedding_model=provider_model(binding.provider, "embed"),
            embedding_space=binding.space,
            total_documents=len(fixtures),
            total_questions=len(question_results),
            vector_hit_rate=(
                sum(1 for item in question_results if item.vector_hit) / total_questions
            ),
            fts_hit_rate=sum(1 for item in question_results if item.fts_hit) / total_questions,
            runtime_hit_rate=(
                sum(1 for item in question_results if item.runtime_hit) / total_questions
            ),
            parent_backfill_question_count=sum(
                1 for item in question_results if item.category == "parent_backfill"
            ),
            parent_backfill_success_count=sum(
                1 for item in question_results if item.parent_backfill_improves
            ),
            table_question_success_count=sum(
                1 for item in question_results if item.category == "table" and item.runtime_hit
            ),
            image_question_success_count=sum(
                1
                for item in question_results
                if item.category in {"ocr", "image_summary"} and item.runtime_hit
            ),
        )

    @staticmethod
    def _classify_issue(
        *,
        corpus_has_expected_answer: bool,
        vector_hit: bool,
        fts_hit: bool,
        runtime_hit: bool,
        parent_backfill_improves: bool,
    ) -> str:
        if not corpus_has_expected_answer:
            return "chunking_or_parsing"
        if runtime_hit or vector_hit or fts_hit:
            return "parent_context_needed" if parent_backfill_improves else "ok"
        return "embedding_or_retrieval"

    @staticmethod
    def _missing_metadata_fields(chunk: Chunk) -> list[str]:
        missing: list[str] = []
        for key in ("location", "toc_path"):
            if not chunk.metadata.get(key):
                missing.append(key)
        if chunk.content_hash in {None, ""}:
            missing.append("content_hash")
        if chunk.chunk_role is ChunkRole.CHILD and not chunk.parent_chunk_id:
            missing.append("parent_chunk_id")
        if chunk.chunk_role is ChunkRole.SPECIAL and not chunk.special_chunk_type:
            missing.append("special_chunk_type")
        return missing

    @staticmethod
    def _matched_terms(text: str, expected_terms: list[str]) -> list[str]:
        normalized_text = OfflineEvalService._normalize_text(text)
        return [term for term in expected_terms if OfflineEvalService._normalize_text(term) in normalized_text]

    @staticmethod
    def _term_count(text: str, expected_terms: list[str]) -> int:
        return len(OfflineEvalService._matched_terms(text, expected_terms))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _preview(text: str, limit: int = 220) -> str:
        normalized = " ".join(text.split())
        return normalized if len(normalized) <= limit else f"{normalized[:limit].rstrip()}..."

    @staticmethod
    def _source_type_for_path(file_path: Path) -> SourceType:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return SourceType.PDF
        if suffix in {".md", ".markdown"}:
            return SourceType.MARKDOWN
        if suffix == ".docx":
            return SourceType.DOCX
        if suffix in {".png", ".jpg", ".jpeg"}:
            return SourceType.IMAGE
        raise ValueError(f"Unsupported file type for offline evaluation: {file_path.suffix or '<none>'}")

    @staticmethod
    def _load_questions(
        questions_path: Path,
        *,
        default_fixture_id: str,
    ) -> list[OfflineEvalQuestion]:
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        raw_questions = payload if isinstance(payload, list) else payload.get("questions", [])
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("questions file must contain a non-empty 'questions' array")

        questions: list[OfflineEvalQuestion] = []
        for index, item in enumerate(raw_questions, start=1):
            if not isinstance(item, dict):
                raise ValueError("each question entry must be an object")
            fixture_id = item.get("fixture_id", default_fixture_id)
            if fixture_id != default_fixture_id:
                raise ValueError("evaluate-file only supports questions for the provided file")
            normalized = {
                "question_id": item.get("question_id", f"question_{index}"),
                "fixture_id": fixture_id,
                **item,
            }
            questions.append(OfflineEvalQuestion.model_validate(normalized))
        return questions

    def _render_markdown_report(self, report: OfflineEvalReport) -> str:
        lines = [
            "# 离线检索评估报告",
            "",
            "## 总览",
            f"- 文档数：{report.summary.total_documents}",
            f"- 问题数：{report.summary.total_questions}",
            f"- Embedding Provider：`{report.summary.embedding_provider}`",
            f"- Embedding Model：`{report.summary.embedding_model or 'unknown'}`",
            f"- Embedding Space：`{report.summary.embedding_space}`",
            f"- Vector Hit Rate：{report.summary.vector_hit_rate:.2f}",
            f"- FTS Hit Rate：{report.summary.fts_hit_rate:.2f}",
            f"- Runtime Hit Rate：{report.summary.runtime_hit_rate:.2f}",
            "",
            "## Chunk 质检",
            f"- Parent chunks：{report.quality_audit.total_parent_chunks}",
            f"- Child chunks：{report.quality_audit.total_child_chunks}",
            f"- Special chunks：{report.quality_audit.total_special_chunks}",
            f"- 重复 searchable chunks：{report.quality_audit.duplicate_searchable_chunks}",
            f"- 空白 chunks：{report.quality_audit.blank_chunks}",
            f"- 过短 child chunks：{report.quality_audit.too_short_child_chunks}",
            f"- 过长 child chunks：{report.quality_audit.too_long_child_chunks}",
            f"- metadata 缺失 chunks：{report.quality_audit.missing_metadata_chunks}",
            "",
            "## 人工检查怎么做",
            "1. 先看每个问题的 `runtime_top_k` 前 3 名，确认是否已经出现正确答案词。",
            (
                "2. 如果命中的是 child chunk，再看它的 "
                "`parent_text_preview`，确认 parent 回填后上下文是否更完整。"
            ),
            "3. 表格题要确认命中的 special chunk 类型是 `table chunk`，而不是普通段落。",
            "4. 图片题要分别确认：数值题命中 `ocr_region`，场景题命中 `image_summary`。",
            (
                "5. 抽样区里重点看长段落 child：如果 child 单独看不完整，"
                "但 parent 能补全，说明切分和回填接口是有效的。"
            ),
            "",
            "## 抽样 Chunk",
        ]
        for sample in report.quality_audit.inspection_samples:
            lines.extend(
                [
                    f"### {sample.filename} / {sample.chunk_role.value}",
                    f"- chunk_id：`{sample.chunk_id}`",
                    f"- special_chunk_type：`{sample.special_chunk_type or 'normal'}`",
                    f"- citation_anchor：`{sample.citation_anchor}`",
                    f"- text：{sample.text}",
                    f"- parent：{sample.parent_text or 'N/A'}",
                    f"- metadata：`{json.dumps(sample.metadata, ensure_ascii=False, sort_keys=True)}`",
                    "",
                ]
            )
        lines.append("## 问题结果")
        for item in report.question_results:
            lines.extend(
                [
                    f"### {item.question_id}",
                    f"- question：{item.question}",
                    f"- category：`{item.category}`",
                    f"- expected_terms：`{', '.join(item.expected_terms)}`",
                    f"- corpus_has_expected_answer：`{item.corpus_has_expected_answer}`",
                    f"- likely_issue：`{item.likely_issue}`",
                    (
                        "- vector_hit / fts_hit / runtime_hit："
                        f"`{item.vector_hit}` / `{item.fts_hit}` / `{item.runtime_hit}`"
                    ),
                    f"- parent_backfill_improves：`{item.parent_backfill_improves}`",
                    "",
                    (
                        "| kind | rank | role | special | score | matched_terms | "
                        "parent_matched_terms | expected_hit | preview |"
                    ),
                    "| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |",
                ]
            )
            for hit in item.runtime_top_k:
                lines.append(
                    "| runtime | "
                    f"{hit.rank} | {hit.chunk_role.value} | {hit.special_chunk_type or ''} | {hit.score:.3f} | "
                    f"{', '.join(hit.matched_terms)} | {', '.join(hit.parent_matched_terms)} | {hit.is_expected_hit} | "
                    f"{hit.text_preview.replace('|', '/')} |"
                )
            for hit in item.vector_top_k:
                lines.append(
                    "| vector | "
                    f"{hit.rank} | {hit.chunk_role.value} | {hit.special_chunk_type or ''} | {hit.score:.3f} | "
                    f"{', '.join(hit.matched_terms)} | {', '.join(hit.parent_matched_terms)} | {hit.is_expected_hit} | "
                    f"{hit.text_preview.replace('|', '/')} |"
                )
            for hit in item.fts_top_k:
                lines.append(
                    "| fts | "
                    f"{hit.rank} | {hit.chunk_role.value} | {hit.special_chunk_type or ''} | {hit.score:.3f} | "
                    f"{', '.join(hit.matched_terms)} | {', '.join(hit.parent_matched_terms)} | {hit.is_expected_hit} | "
                    f"{hit.text_preview.replace('|', '/')} |"
                )
            lines.append("")
        return "\n".join(lines)


def run_builtin_offline_eval(output_dir: Path, *, top_k: int = 5) -> OfflineEvalRunResult:
    return OfflineEvalService(output_dir=output_dir, top_k=top_k).run_builtin_pack()


def run_file_offline_eval(
    *,
    file_path: Path,
    questions_path: Path,
    output_dir: Path,
    top_k: int = 5,
) -> OfflineEvalRunResult:
    return OfflineEvalService(output_dir=output_dir, top_k=top_k).run_file(
        file_path=file_path,
        questions_path=questions_path,
    )
