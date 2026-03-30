from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pkp.interfaces._bootstrap import build_runtime_container, load_settings
from pkp.interfaces._config import build_execution_policy, default_access_policy
from pkp.schema._types import ComplexityLevel, ExecutionLocationPreference, TaskType

DEFAULT_METRIC_NAMES = [
    "id_based_context_precision",
    "id_based_context_recall",
    "answer_relevancy",
    "faithfulness",
]
DEFAULT_RAGAS_MAX_TOKENS = 4096


class CorpusManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    file_name: str
    local_path: str
    source_url: str
    source_type: str
    format: str
    language: str
    notes: list[str] = Field(default_factory=list)
    auxiliary_files: list[str] = Field(default_factory=list)


class ChunkReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_doc_id: str
    runtime_doc_id: str
    location: str
    file_name: str
    source_type: str
    chunk_id: str
    text: str
    citation_anchor: str
    section_path: list[str] = Field(default_factory=list)
    chunk_role: str = "child"
    special_chunk_type: str | None = None
    order_index: int = 0


class EnrichedSyntheticSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    question: str
    question_type: str
    expected_doc_id: str | None = None
    expected_runtime_doc_id: str | None = None
    expected_section_hint: str | None = None
    reference: str | None = None
    reference_contexts: list[str] = Field(default_factory=list)
    reference_context_ids: list[str] = Field(default_factory=list)
    query_style: str | None = None
    query_length: str | None = None
    synthesizer_name: str | None = None


class RuntimeAnswerRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    question: str
    question_type: str
    expected_doc_id: str | None = None
    expected_runtime_doc_id: str | None = None
    expected_section_hint: str | None = None
    response: str
    runtime_mode: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    retrieved_context_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    groundedness_flag: bool = False
    insufficient_evidence_flag: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EvaluationArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    chunks_jsonl: Path
    chunks_csv: Path
    generation_chunks_jsonl: Path
    generation_chunks_csv: Path
    testset_jsonl: Path
    testset_csv: Path
    runtime_answers_jsonl: Path
    runtime_answers_csv: Path
    ragas_scores_jsonl: Path
    ragas_scores_csv: Path
    low_scores_jsonl: Path
    low_scores_csv: Path
    summary_json: Path


class EmbeddingCompatibilityAdapter:
    """Expose both modern Ragas and legacy LangChain embedding interfaces."""

    def __init__(self, embedding: Any):
        self._embedding = embedding

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return list(self._embedding.embed_text(text, **kwargs))

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        if getattr(self._embedding, "is_async", False):
            return list(await self._embedding.aembed_text(text, **kwargs))
        return self.embed_text(text, **kwargs)

    def embed_texts(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        return [list(vector) for vector in self._embedding.embed_texts(list(texts), **kwargs)]

    async def aembed_texts(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        if getattr(self._embedding, "is_async", False):
            vectors = await self._embedding.aembed_texts(list(texts), **kwargs)
            return [list(vector) for vector in vectors]
        return self.embed_texts(texts, **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text, **kwargs)

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        return await self.aembed_text(text, **kwargs)

    def embed_documents(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        return self.embed_texts(texts, **kwargs)

    async def aembed_documents(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        return await self.aembed_texts(texts, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embedding, name)


def infer_question_type(question: str) -> str:
    normalized = " ".join(question.lower().split())
    if any(token in normalized for token in ("summarize", "summary", "overview", "main idea")):
        return "summary"
    if any(
        token in normalized
        for token in (
            "which section",
            "what section",
            "where in the document",
            "what page",
            "under what heading",
        )
    ):
        return "section_lookup"
    if "table" in normalized:
        return "table_question"
    if any(token in normalized for token in ("figure", "diagram", "chart")):
        return "figure_question"
    if any(
        token in normalized
        for token in (
            "ocr",
            "scanned",
            "document image",
            "letter",
            "invoice",
            "form",
            "header",
            "date appears",
        )
    ):
        return "ocr_question"
    return "factual"


def enrich_generated_samples(
    rows: Sequence[Mapping[str, Any]],
    chunks: Sequence[ChunkReference],
) -> list[EnrichedSyntheticSample]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunks_by_text = _build_chunk_text_lookup(chunks)
    enriched: list[EnrichedSyntheticSample] = []
    for index, row in enumerate(rows, start=1):
        reference_contexts = [value for value in row.get("reference_contexts", []) if isinstance(value, str)]
        reference_context_ids = _resolve_reference_context_ids(
            row=row,
            chunk_by_id=chunk_by_id,
            chunks_by_text=chunks_by_text,
        )
        matched_chunks = [chunk_by_id[chunk_id] for chunk_id in reference_context_ids if chunk_id in chunk_by_id]
        expected_doc_id, expected_runtime_doc_id = _resolve_expected_doc_ids(matched_chunks)
        question = str(row.get("user_input", "")).strip()
        enriched.append(
            EnrichedSyntheticSample(
                question_id=f"synthetic_{index:03d}",
                question=question,
                question_type=infer_question_type(question),
                expected_doc_id=expected_doc_id,
                expected_runtime_doc_id=expected_runtime_doc_id,
                expected_section_hint=_build_expected_section_hint(matched_chunks),
                reference=_maybe_str(row.get("reference")),
                reference_contexts=reference_contexts,
                reference_context_ids=reference_context_ids,
                query_style=_maybe_str(row.get("query_style")),
                query_length=_maybe_str(row.get("query_length")),
                synthesizer_name=_maybe_str(row.get("synthesizer_name")),
            )
        )
    return enriched


def summarize_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"sample_count": len(rows)}
    per_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    overall_scores: list[float] = []
    for row in rows:
        row_scores: list[float] = []
        for metric_name in metric_names:
            score = _coerce_score(row.get(metric_name))
            if score is None:
                continue
            per_metric[metric_name].append(score)
            row_scores.append(score)
        if row_scores:
            overall_scores.append(round(sum(row_scores) / len(row_scores), 4))
    for metric_name in metric_names:
        values = per_metric[metric_name]
        summary[metric_name] = round(sum(values) / len(values), 4) if values else 0.0
    summary["overall_score"] = round(sum(overall_scores) / len(overall_scores), 4) if overall_scores else 0.0
    return summary


def select_low_score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    threshold: float,
) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for row in rows:
        row_payload = dict(row)
        reasons: list[str] = []
        scores: list[float] = []
        for metric_name in metric_names:
            score = _coerce_score(row_payload.get(metric_name))
            if score is None:
                continue
            scores.append(score)
            if score < threshold:
                reasons.append(metric_name)
        overall_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        row_payload["overall_score"] = overall_score
        if overall_score < threshold:
            reasons.append("overall_score")
        if reasons:
            row_payload["low_score_reasons"] = reasons
            flagged.append(row_payload)
    return flagged


def select_generation_chunks(
    chunks: Sequence[ChunkReference],
    *,
    max_child_chunks_per_doc: int = 8,
    max_special_chunks_per_doc: int = 4,
    max_ocr_region_chunks_per_doc: int = 2,
) -> list[ChunkReference]:
    grouped_chunks: dict[str, list[ChunkReference]] = {}
    doc_order: list[str] = []
    for chunk in chunks:
        if chunk.manifest_doc_id not in grouped_chunks:
            grouped_chunks[chunk.manifest_doc_id] = []
            doc_order.append(chunk.manifest_doc_id)
        grouped_chunks[chunk.manifest_doc_id].append(chunk)

    selected: list[ChunkReference] = []
    seen_chunk_ids: set[str] = set()
    for doc_id in doc_order:
        doc_chunks = sorted(grouped_chunks[doc_id], key=lambda chunk: (chunk.order_index, chunk.chunk_id))
        child_chunks = [chunk for chunk in doc_chunks if chunk.chunk_role == "child"]
        structured_special_chunks = [
            chunk for chunk in doc_chunks if _is_structured_special_chunk(chunk)
        ]
        ocr_region_chunks = [chunk for chunk in doc_chunks if chunk.special_chunk_type == "ocr_region"]

        selected_for_doc = [
            *_sample_evenly(child_chunks, max_child_chunks_per_doc),
            *_select_structured_special_chunks(structured_special_chunks, max_special_chunks_per_doc),
            *ocr_region_chunks[: max(0, max_ocr_region_chunks_per_doc)],
        ]
        for chunk in selected_for_doc:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
    return selected


def run_prechunked_ragas_eval(
    *,
    corpus_root: Path,
    manifest_path: Path,
    output_dir: Path,
    question_count: int = 24,
    mode: str = "fast",
    low_score_threshold: float = 0.6,
    skip_ingest: bool = False,
    debugging_logs: bool = False,
    with_factual_correctness: bool = False,
    response_relevancy_strictness: int = 1,
    max_child_chunks_per_doc: int = 8,
    max_special_chunks_per_doc: int = 4,
    max_ocr_region_chunks_per_doc: int = 2,
) -> dict[str, Any]:
    settings = load_settings()
    project_root = Path(__file__).resolve().parents[3]
    container = build_runtime_container(settings)
    artifacts = _prepare_artifacts(output_dir)

    manifest_entries = _load_manifest(manifest_path)
    ingested_docs = _ensure_corpus_ingested(
        container=container,
        manifest_entries=manifest_entries,
        corpus_root=corpus_root,
        project_root=project_root,
        skip_ingest=skip_ingest,
    )
    chunks = _collect_chunk_references(
        metadata_repo=container.metadata_repo,
        ingested_docs=ingested_docs,
    )
    if not chunks:
        raise RuntimeError("No chunks were found for the requested corpus.")
    _write_records(chunks, artifacts.chunks_jsonl, artifacts.chunks_csv)
    generation_chunks = select_generation_chunks(
        chunks,
        max_child_chunks_per_doc=max_child_chunks_per_doc,
        max_special_chunks_per_doc=max_special_chunks_per_doc,
        max_ocr_region_chunks_per_doc=max_ocr_region_chunks_per_doc,
    )
    if not generation_chunks:
        generation_chunks = list(chunks)
    _write_records(generation_chunks, artifacts.generation_chunks_jsonl, artifacts.generation_chunks_csv)

    generated_rows = _generate_testset_rows(
        chunks=generation_chunks,
        settings=settings,
        question_count=question_count,
        debugging_logs=debugging_logs,
    )
    enriched_samples = enrich_generated_samples(generated_rows, chunks)
    _write_records(enriched_samples, artifacts.testset_jsonl, artifacts.testset_csv)

    runtime_answers, evaluation_dataset = _run_runtime_answers(
        container=container,
        samples=enriched_samples,
        runtime_doc_ids=[doc["runtime_doc_id"] for doc in ingested_docs],
        mode=mode,
        execution_location_preference=settings.runtime.execution_location_preference,
    )
    _write_records(runtime_answers, artifacts.runtime_answers_jsonl, artifacts.runtime_answers_csv)

    score_rows = _evaluate_runtime_answers(
        evaluation_dataset=evaluation_dataset,
        samples=enriched_samples,
        runtime_answers=runtime_answers,
        settings=settings,
        with_factual_correctness=with_factual_correctness,
        response_relevancy_strictness=response_relevancy_strictness,
    )
    metric_names = list(DEFAULT_METRIC_NAMES)
    if with_factual_correctness:
        metric_names.append("factual_correctness")
    _write_rows(score_rows, artifacts.ragas_scores_jsonl, artifacts.ragas_scores_csv)

    low_score_rows = select_low_score_rows(
        score_rows,
        metric_names=metric_names,
        threshold=low_score_threshold,
    )
    _write_rows(low_score_rows, artifacts.low_scores_jsonl, artifacts.low_scores_csv)

    summary = summarize_metric_rows(score_rows, metric_names=metric_names)
    summary_payload = {
        "corpus_root": str(corpus_root),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "mode": mode,
        "question_count": len(enriched_samples),
        "document_count": len(ingested_docs),
        "chunk_count": len(chunks),
        "generation_chunk_count": len(generation_chunks),
        "low_score_threshold": low_score_threshold,
        "generation_sampling": {
            "max_child_chunks_per_doc": max_child_chunks_per_doc,
            "max_special_chunks_per_doc": max_special_chunks_per_doc,
            "max_ocr_region_chunks_per_doc": max_ocr_region_chunks_per_doc,
        },
        "metric_names": metric_names,
        "metrics": summary,
        "artifacts": artifacts.model_dump(mode="json"),
    }
    artifacts.summary_json.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and evaluate a Ragas synthetic testset from PKP chunks.")
    parser.add_argument(
        "--corpus-root",
        default="data/test_corpus",
        help="Directory containing the prepared test corpus.",
    )
    parser.add_argument(
        "--manifest",
        default="data/test_corpus/manifest.json",
        help="Path to the corpus manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval/ragas_prechunked",
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--question-count",
        type=int,
        default=24,
        help="Number of synthetic questions to generate.",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "deep"),
        default="fast",
        help="Which PKP runtime to evaluate.",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.6,
        help="Threshold below which rows are exported to the low-score file.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Assume the corpus has already been ingested into the current runtime.",
    )
    parser.add_argument(
        "--debugging-logs",
        action="store_true",
        help="Enable verbose Ragas generation logs.",
    )
    parser.add_argument(
        "--with-factual-correctness",
        action="store_true",
        help="Enable the slower factual_correctness metric in addition to the default metric set.",
    )
    parser.add_argument(
        "--response-relevancy-strictness",
        type=int,
        default=1,
        help="Ragas ResponseRelevancy strictness. Lower values are faster.",
    )
    parser.add_argument(
        "--max-child-chunks-per-doc",
        type=int,
        default=8,
        help="Maximum child chunks per document used only for synthetic question generation.",
    )
    parser.add_argument(
        "--max-special-chunks-per-doc",
        type=int,
        default=4,
        help="Maximum non-OCR special chunks per document used only for synthetic question generation.",
    )
    parser.add_argument(
        "--max-ocr-region-chunks-per-doc",
        type=int,
        default=2,
        help="Maximum OCR-region chunks per document used only for synthetic question generation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run_prechunked_ragas_eval(
        corpus_root=Path(args.corpus_root),
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        question_count=args.question_count,
        mode=args.mode,
        low_score_threshold=args.low_score_threshold,
        skip_ingest=args.skip_ingest,
        debugging_logs=args.debugging_logs,
        with_factual_correctness=args.with_factual_correctness,
        response_relevancy_strictness=args.response_relevancy_strictness,
        max_child_chunks_per_doc=args.max_child_chunks_per_doc,
        max_special_chunks_per_doc=args.max_special_chunks_per_doc,
        max_ocr_region_chunks_per_doc=args.max_ocr_region_chunks_per_doc,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def _load_manifest(manifest_path: Path) -> list[CorpusManifestEntry]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{manifest_path} must contain a JSON array.")
    return [CorpusManifestEntry.model_validate(item) for item in payload]


def _prepare_artifacts(output_dir: Path) -> EvaluationArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    return EvaluationArtifacts(
        output_dir=output_dir,
        chunks_jsonl=output_dir / "chunks.jsonl",
        chunks_csv=output_dir / "chunks.csv",
        generation_chunks_jsonl=output_dir / "generation_chunks.jsonl",
        generation_chunks_csv=output_dir / "generation_chunks.csv",
        testset_jsonl=output_dir / "synthetic_testset.jsonl",
        testset_csv=output_dir / "synthetic_testset.csv",
        runtime_answers_jsonl=output_dir / "runtime_answers.jsonl",
        runtime_answers_csv=output_dir / "runtime_answers.csv",
        ragas_scores_jsonl=output_dir / "ragas_scores.jsonl",
        ragas_scores_csv=output_dir / "ragas_scores.csv",
        low_scores_jsonl=output_dir / "low_score_samples.jsonl",
        low_scores_csv=output_dir / "low_score_samples.csv",
        summary_json=output_dir / "summary.json",
    )


def _ensure_corpus_ingested(
    *,
    container: Any,
    manifest_entries: Sequence[CorpusManifestEntry],
    corpus_root: Path,
    project_root: Path,
    skip_ingest: bool,
) -> list[dict[str, str]]:
    metadata_repo = container.metadata_repo
    if metadata_repo is None:
        raise RuntimeError("Runtime container does not expose metadata_repo.")
    ingested_docs: list[dict[str, str]] = []
    for entry in manifest_entries:
        relative_location, absolute_path = _resolve_manifest_file(entry, corpus_root=corpus_root, project_root=project_root)
        if skip_ingest:
            document = metadata_repo.get_latest_document_for_location(relative_location)
            if document is None:
                raise RuntimeError(f"Document {relative_location} was not found in the current runtime.")
            runtime_doc_id = document.doc_id
        else:
            result = container.ingest_runtime.process_file(location=relative_location, title=entry.title)
            runtime_doc_id = str(result["doc_id"])
        ingested_docs.append(
            {
                "manifest_doc_id": entry.doc_id,
                "runtime_doc_id": runtime_doc_id,
                "location": relative_location,
                "absolute_path": str(absolute_path),
                "file_name": entry.file_name,
                "source_type": entry.source_type,
            }
        )
    return ingested_docs


def _resolve_manifest_file(
    entry: CorpusManifestEntry,
    *,
    corpus_root: Path,
    project_root: Path,
) -> tuple[str, Path]:
    candidates = [
        project_root / entry.local_path,
        corpus_root.parent / entry.local_path,
        corpus_root / Path(entry.local_path).name,
    ]
    if entry.local_path.startswith("test_corpus/"):
        candidates.append(corpus_root / Path(entry.local_path).relative_to("test_corpus"))
    for candidate in candidates:
        if candidate.exists():
            try:
                relative_location = candidate.resolve().relative_to(project_root.resolve()).as_posix()
            except ValueError as exc:
                raise RuntimeError(f"{candidate} is not under project root {project_root}") from exc
            return relative_location, candidate
    raise FileNotFoundError(f"Could not resolve {entry.local_path} from manifest.")


def _collect_chunk_references(
    *,
    metadata_repo: Any,
    ingested_docs: Sequence[Mapping[str, str]],
) -> list[ChunkReference]:
    references: list[ChunkReference] = []
    for doc in ingested_docs:
        document = metadata_repo.get_document(doc["runtime_doc_id"])
        if document is None:
            continue
        segments = {segment.segment_id: segment for segment in metadata_repo.list_segments(document.doc_id)}
        for chunk in metadata_repo.list_chunks(document.doc_id):
            segment = segments.get(chunk.segment_id)
            chunk_role = getattr(chunk.chunk_role, "value", str(chunk.chunk_role))
            references.append(
                ChunkReference(
                    manifest_doc_id=doc["manifest_doc_id"],
                    runtime_doc_id=document.doc_id,
                    location=doc["location"],
                    file_name=doc["file_name"],
                    source_type=doc["source_type"],
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    citation_anchor=chunk.citation_anchor,
                    section_path=[] if segment is None else list(segment.toc_path),
                    chunk_role=chunk_role,
                    special_chunk_type=chunk.special_chunk_type,
                    order_index=chunk.order_index,
                )
            )
    return references


def _generate_testset_rows(
    *,
    chunks: Sequence[ChunkReference],
    settings: Any,
    question_count: int,
    debugging_logs: bool,
) -> list[dict[str, Any]]:
    from langchain_core.documents import Document as LCDocument
    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator

    client = OpenAI(
        api_key=settings.openai.api_key.get_secret_value(),
        base_url=settings.openai.base_url,
    )
    llm = llm_factory(
        settings.openai.model,
        client=client,
        provider="openai",
        max_tokens=DEFAULT_RAGAS_MAX_TOKENS,
    )
    embeddings = _build_ragas_embeddings(client=client, settings=settings)
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    documents = [
        LCDocument(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "manifest_doc_id": chunk.manifest_doc_id,
                "runtime_doc_id": chunk.runtime_doc_id,
                "citation_anchor": chunk.citation_anchor,
                "section_path": list(chunk.section_path),
                "location": chunk.location,
                "file_name": chunk.file_name,
                "source_type": chunk.source_type,
            },
        )
        for chunk in chunks
    ]
    testset = generator.generate_with_chunks(
        documents,
        testset_size=question_count,
        with_debugging_logs=debugging_logs,
        raise_exceptions=True,
    )
    return testset.to_list()


def _run_runtime_answers(
    *,
    container: Any,
    samples: Sequence[EnrichedSyntheticSample],
    runtime_doc_ids: Sequence[str],
    mode: str,
    execution_location_preference: ExecutionLocationPreference,
) -> tuple[list[RuntimeAnswerRecord], Any]:
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    runtime_answers: list[RuntimeAnswerRecord] = []
    evaluation_samples: list[Any] = []
    source_scope = list(runtime_doc_ids)
    for sample in samples:
        policy = build_execution_policy(
            task_type=TaskType.RESEARCH if mode == "deep" else TaskType.LOOKUP,
            complexity_level=ComplexityLevel.L4_RESEARCH if mode == "deep" else ComplexityLevel.L1_DIRECT,
            access_policy=default_access_policy(),
            source_scope=source_scope,
            execution_location_preference=execution_location_preference,
        )
        if mode == "deep":
            response = container.deep_research_runtime.run(sample.question, policy, session_id=sample.question_id)
        else:
            response = container.fast_query_runtime.run(sample.question, policy)
        runtime_record = RuntimeAnswerRecord(
            question_id=sample.question_id,
            question=sample.question,
            question_type=sample.question_type,
            expected_doc_id=sample.expected_doc_id,
            expected_runtime_doc_id=sample.expected_runtime_doc_id,
            expected_section_hint=sample.expected_section_hint,
            response=response.answer_text or response.conclusion,
            runtime_mode=response.runtime_mode.value,
            retrieved_contexts=[item.text for item in response.evidence],
            retrieved_context_ids=[item.chunk_id for item in response.evidence],
            retrieved_doc_ids=[item.doc_id for item in response.evidence],
            citations=[item.chunk_id for item in response.citations],
            groundedness_flag=response.groundedness_flag,
            insufficient_evidence_flag=response.insufficient_evidence_flag,
            diagnostics=response.diagnostics.model_dump(mode="json"),
        )
        runtime_answers.append(runtime_record)
        evaluation_samples.append(
            SingleTurnSample(
                user_input=sample.question,
                retrieved_contexts=runtime_record.retrieved_contexts,
                reference_contexts=list(sample.reference_contexts),
                retrieved_context_ids=runtime_record.retrieved_context_ids,
                reference_context_ids=list(sample.reference_context_ids),
                response=runtime_record.response,
                reference=sample.reference,
                persona_name=sample.expected_doc_id,
                query_style=sample.query_style,
                query_length=sample.query_length,
            )
        )
    return runtime_answers, EvaluationDataset(samples=evaluation_samples)


def _evaluate_runtime_answers(
    *,
    evaluation_dataset: Any,
    samples: Sequence[EnrichedSyntheticSample],
    runtime_answers: Sequence[RuntimeAnswerRecord],
    settings: Any,
    with_factual_correctness: bool,
    response_relevancy_strictness: int,
) -> list[dict[str, Any]]:
    from openai import OpenAI
    from ragas import evaluate
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics import (
        Faithfulness,
        FactualCorrectness,
        IDBasedContextPrecision,
        IDBasedContextRecall,
        ResponseRelevancy,
    )

    client = OpenAI(
        api_key=settings.openai.api_key.get_secret_value(),
        base_url=settings.openai.base_url,
    )
    llm = llm_factory(
        settings.openai.model,
        client=client,
        provider="openai",
        max_tokens=DEFAULT_RAGAS_MAX_TOKENS,
    )
    embeddings = _build_ragas_embeddings(client=client, settings=settings)
    metrics = [
        IDBasedContextPrecision(),
        IDBasedContextRecall(),
        ResponseRelevancy(strictness=response_relevancy_strictness),
        Faithfulness(),
    ]
    if with_factual_correctness:
        metrics.append(FactualCorrectness())
    result = evaluate(
        evaluation_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
    )
    score_rows = result.to_pandas().to_dict(orient="records")
    merged_rows: list[dict[str, Any]] = []
    metric_names = list(DEFAULT_METRIC_NAMES)
    if with_factual_correctness:
        metric_names.append("factual_correctness")
    for sample, runtime_answer, score_row in zip(samples, runtime_answers, score_rows, strict=True):
        row = {
            "question_id": sample.question_id,
            "question": sample.question,
            "question_type": sample.question_type,
            "expected_doc_id": sample.expected_doc_id,
            "expected_runtime_doc_id": sample.expected_runtime_doc_id,
            "expected_section_hint": sample.expected_section_hint,
            "reference": sample.reference,
            "reference_context_ids": list(sample.reference_context_ids),
            "retrieved_context_ids": list(runtime_answer.retrieved_context_ids),
            "retrieved_doc_ids": list(runtime_answer.retrieved_doc_ids),
            "response": runtime_answer.response,
            "runtime_mode": runtime_answer.runtime_mode,
            "groundedness_flag": runtime_answer.groundedness_flag,
            "insufficient_evidence_flag": runtime_answer.insufficient_evidence_flag,
            "diagnostics": runtime_answer.diagnostics,
        }
        for metric_name in metric_names:
            row[metric_name] = _coerce_score(score_row.get(metric_name))
        row["overall_score"] = _row_overall_score(row, metric_names)
        merged_rows.append(row)
    return merged_rows


def _write_records(records: Sequence[BaseModel], jsonl_path: Path, csv_path: Path) -> None:
    _write_rows([record.model_dump(mode="json") for record in records], jsonl_path, csv_path)


def _build_ragas_embeddings(*, client: Any, settings: Any) -> EmbeddingCompatibilityAdapter:
    from ragas.embeddings import embedding_factory

    embedding = embedding_factory(
        provider="openai",
        model=settings.openai.embedding_model,
        client=client,
    )
    return EmbeddingCompatibilityAdapter(embedding)


def _write_rows(rows: Sequence[Mapping[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=True) + "\n")

    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _resolve_reference_context_ids(
    *,
    row: Mapping[str, Any],
    chunk_by_id: Mapping[str, ChunkReference],
    chunks_by_text: Mapping[str, list[ChunkReference]],
) -> list[str]:
    resolved: list[str] = []
    for value in row.get("reference_context_ids", []):
        if isinstance(value, str) and value in chunk_by_id:
            resolved.append(value)
    if resolved:
        return _deduplicate(resolved)

    for context in row.get("reference_contexts", []):
        if not isinstance(context, str):
            continue
        normalized = _normalize_text(context)
        for chunk in chunks_by_text.get(normalized, []):
            resolved.append(chunk.chunk_id)
            break
        if resolved:
            continue
        best_match: ChunkReference | None = None
        best_overlap = 0.0
        for chunk in chunk_by_id.values():
            overlap = _text_overlap(normalized, _normalize_text(chunk.text))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = chunk
        if best_match is not None and best_overlap >= 0.4:
            resolved.append(best_match.chunk_id)
    return _deduplicate(resolved)


def _resolve_expected_doc_ids(
    matched_chunks: Sequence[ChunkReference],
) -> tuple[str | None, str | None]:
    if not matched_chunks:
        return None, None
    manifest_counts = Counter(chunk.manifest_doc_id for chunk in matched_chunks)
    runtime_counts = Counter(chunk.runtime_doc_id for chunk in matched_chunks)
    manifest_doc_id = manifest_counts.most_common(1)[0][0]
    runtime_doc_id = runtime_counts.most_common(1)[0][0]
    return manifest_doc_id, runtime_doc_id


def _build_expected_section_hint(matched_chunks: Sequence[ChunkReference]) -> str | None:
    if not matched_chunks:
        return None
    for chunk in matched_chunks:
        if chunk.section_path:
            return " > ".join(chunk.section_path)
    return matched_chunks[0].citation_anchor or None


def _build_chunk_text_lookup(chunks: Sequence[ChunkReference]) -> dict[str, list[ChunkReference]]:
    lookup: dict[str, list[ChunkReference]] = {}
    for chunk in chunks:
        normalized = _normalize_text(chunk.text)
        if not normalized:
            continue
        lookup.setdefault(normalized, []).append(chunk)
    return lookup


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _text_overlap(left: str, right: str) -> float:
    left_terms = set(left.split())
    right_terms = set(right.split())
    if not left_terms or not right_terms:
        return 0.0
    intersection = left_terms & right_terms
    union = left_terms | right_terms
    return len(intersection) / len(union)


def _deduplicate(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _sample_evenly(chunks: Sequence[ChunkReference], limit: int) -> list[ChunkReference]:
    if limit <= 0 or not chunks:
        return []
    if len(chunks) <= limit:
        return list(chunks)
    if limit == 1:
        return [chunks[0]]

    selected_indices = {
        round(index * (len(chunks) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [chunk for index, chunk in enumerate(chunks) if index in selected_indices]


def _is_structured_special_chunk(chunk: ChunkReference) -> bool:
    return chunk.chunk_role == "special" and chunk.special_chunk_type not in {None, "ocr_region"}


def _select_structured_special_chunks(
    chunks: Sequence[ChunkReference],
    limit: int,
) -> list[ChunkReference]:
    if limit <= 0 or not chunks:
        return []
    prioritized = sorted(
        chunks,
        key=lambda chunk: (_structured_special_priority(chunk.special_chunk_type), chunk.order_index, chunk.chunk_id),
    )
    return prioritized[:limit]


def _structured_special_priority(special_chunk_type: str | None) -> int:
    priorities = {
        "table": 0,
        "figure": 1,
        "image_summary": 2,
        "caption": 3,
    }
    return priorities.get(special_chunk_type, 10)


def _maybe_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_score(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isnan(number):
            return None
        return number
    return None


def _row_overall_score(row: Mapping[str, Any], metric_names: Sequence[str]) -> float:
    scores = [score for metric_name in metric_names if (score := _coerce_score(row.get(metric_name))) is not None]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _csv_cell(value: object) -> str | int | float | None:
    normalized = _jsonable(value)
    if normalized is None:
        return None
    if isinstance(normalized, (str, int, float)):
        return normalized
    return json.dumps(normalized, ensure_ascii=True)


__all__ = [
    "ChunkReference",
    "CorpusManifestEntry",
    "EmbeddingCompatibilityAdapter",
    "EnrichedSyntheticSample",
    "EvaluationArtifacts",
    "RuntimeAnswerRecord",
    "build_arg_parser",
    "enrich_generated_samples",
    "infer_question_type",
    "main",
    "run_prechunked_ragas_eval",
    "select_generation_chunks",
    "select_low_score_rows",
    "summarize_metric_rows",
]
