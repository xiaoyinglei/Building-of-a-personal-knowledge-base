from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import EmbeddingProviderBinding, VectorSearchResult
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.search.web_search_repo import DeterministicWebSearchRepo
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.runtime.provider_metadata import provider_model, provider_name
from pkp.service.answer_generation_service import AnswerGenerationService
from pkp.service.artifact_service import ArtifactService
from pkp.service.evidence_service import CandidateLike, EvidenceBundle, EvidenceService
from pkp.service.ingest_service import IngestService
from pkp.service.query_understanding_service import QueryUnderstandingService
from pkp.service.retrieval_service import RetrievalResult, RetrievalService
from pkp.service.telemetry_service import TelemetryService
from pkp.types import (
    AccessPolicy,
    ArtifactStatus,
    Document,
    EvidenceItem,
    ExecutionLocation,
    ExecutionLocationPreference,
    ExecutionPolicy,
    KnowledgeArtifact,
    ModelDiagnostics,
    PreservationSuggestion,
    ProviderAttempt,
    QueryDiagnostics,
    QueryResponse,
    RuntimeMode,
)
from pkp.types.content import ChunkRole
from pkp.types.generation import GroundedAnswer
from pkp.types.text import (
    focus_terms,
    keyword_overlap,
    looks_command_like,
    looks_definition_query,
    looks_definition_text,
    looks_operation_query,
    looks_operation_text,
    looks_structure_query,
    looks_structure_text,
    search_terms,
    split_sentences,
)


@dataclass(frozen=True)
class RetrievedCandidate:
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str = "internal"
    source_id: str | None = None
    section_path: tuple[str, ...] = ()
    effective_access_policy: AccessPolicy | None = None
    chunk_role: ChunkRole | None = None
    special_chunk_type: str | None = None
    parent_chunk_id: str | None = None
    parent_text: str | None = None
    metadata: dict[str, str] | None = None
    file_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_type: str | None = None
    source_type: str | None = None


class VectorSearchRepoProtocol(Protocol):
    def search(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
    ) -> Sequence[VectorSearchResult]: ...

    def count_vectors(self, *, embedding_space: str | None = None, distinct_chunks: bool = False) -> int: ...


class MultiProviderBackedVectorRetriever:
    def __init__(
        self,
        *,
        factory: SearchBackedRetrievalFactory,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingProviderBinding],
        default_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST,
    ) -> None:
        self._factory = factory
        self._vector_repo = vector_repo
        self._bindings = tuple(bindings)
        self._default_preference = default_preference
        self._prepared_locations: tuple[str, ...] = ("local", "cloud")
        self.last_provider: str | None = None
        self.last_attempts: list[ProviderAttempt] = []

    def prepare_for_policy(
        self,
        *,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference | None,
    ) -> None:
        preference = execution_location_preference or self._default_preference
        if access_policy.local_only or preference is ExecutionLocationPreference.LOCAL_ONLY:
            self._prepared_locations = ("local",)
            return
        preferred_order = (
            (ExecutionLocation.LOCAL, ExecutionLocation.CLOUD)
            if preference is ExecutionLocationPreference.LOCAL_FIRST
            else (ExecutionLocation.CLOUD, ExecutionLocation.LOCAL)
        )
        self._prepared_locations = tuple(
            location.value for location in preferred_order if access_policy.allows_location(location)
        )

    def __call__(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        self.last_provider = None
        self.last_attempts = []
        ordered_bindings = self._ordered_bindings()
        for binding in ordered_bindings:
            candidates = self._search_binding(
                binding,
                query=query,
                source_scope=source_scope,
                target_space=binding.space,
            )
            if candidates:
                self.last_provider = provider_name(binding.provider)
                return candidates
        for binding in ordered_bindings:
            candidates = self._search_binding(
                binding,
                query=query,
                source_scope=source_scope,
                target_space="default",
            )
            if candidates:
                self.last_provider = provider_name(binding.provider)
                return candidates
        return []

    def _ordered_bindings(self) -> list[EmbeddingProviderBinding]:
        ordered: list[EmbeddingProviderBinding] = []
        remaining = list(self._bindings)
        for location in self._prepared_locations:
            matched = [binding for binding in remaining if binding.location == location]
            ordered.extend(matched)
            remaining = [binding for binding in remaining if binding.location != location]
        ordered.extend(remaining)
        return ordered

    def _search_binding(
        self,
        binding: EmbeddingProviderBinding,
        *,
        query: str,
        source_scope: list[str],
        target_space: str,
    ) -> list[RetrievedCandidate]:
        attempt = ProviderAttempt(
            stage="embedding",
            capability="embed",
            provider=provider_name(binding.provider),
            location=binding.location,
            model=provider_model(binding.provider, "embed"),
            status="success",
        )
        embed = getattr(binding.provider, "embed", None)
        if not callable(embed):
            self.last_attempts.append(
                attempt.model_copy(update={"status": "failed", "error": "embedding not supported"})
            )
            return []
        try:
            query_vectors = cast(list[list[float]], embed([query]))
        except RuntimeError as exc:
            self.last_attempts.append(attempt.model_copy(update={"status": "failed", "error": str(exc)}))
            return []
        if not query_vectors:
            self.last_attempts.append(
                attempt.model_copy(update={"status": "failed", "error": "provider returned no vectors"})
            )
            return []
        if self._vector_repo.count_vectors(embedding_space=target_space) == 0:
            self.last_attempts.append(
                attempt.model_copy(update={"status": "failed", "error": f"no indexed vectors for {target_space}"})
            )
            return []

        results = self._vector_repo.search(
            query_vectors[0],
            limit=12,
            doc_ids=source_scope or None,
            embedding_space=target_space,
        )
        if not results:
            self.last_attempts.append(
                attempt.model_copy(update={"status": "failed", "error": f"no vector hits in {target_space}"})
            )
            return []

        self.last_attempts.append(attempt)
        ordered_chunk_ids = [result.chunk_id for result in results]
        score_map = {result.chunk_id: float(result.score) for result in results}
        candidates = self._factory._build_candidates_from_chunk_ids(
            ordered_chunk_ids,
            source_kind="internal",
            scope=source_scope,
        )
        return [
            candidate
            if candidate.chunk_id not in score_map
            else RetrievedCandidate(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                source_id=candidate.source_id,
                text=candidate.text,
                citation_anchor=candidate.citation_anchor,
                score=score_map[candidate.chunk_id],
                rank=index,
                source_kind=candidate.source_kind,
                section_path=candidate.section_path,
                effective_access_policy=candidate.effective_access_policy,
                chunk_role=candidate.chunk_role,
                special_chunk_type=candidate.special_chunk_type,
                parent_chunk_id=candidate.parent_chunk_id,
                parent_text=candidate.parent_text,
                metadata=candidate.metadata,
                file_name=candidate.file_name,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                chunk_type=candidate.chunk_type,
                source_type=candidate.source_type,
            )
            for index, candidate in enumerate(candidates, start=1)
        ]


class InstrumentedReranker:
    def __init__(self, rerank_service: object) -> None:
        self._rerank_service = rerank_service
        self.provider_name = getattr(rerank_service, "provider_name", "heuristic")
        self.rerank_model_name = getattr(rerank_service, "rerank_model_name", "heuristic")
        self.is_rerank_configured = True
        self.last_provider: str | None = self.provider_name
        self.last_attempts: list[ProviderAttempt] = []

    def __call__(self, query: str, candidates: list[CandidateLike]) -> Sequence[CandidateLike]:
        attempt = ProviderAttempt(
            stage="rerank",
            capability="rerank",
            provider=self.provider_name,
            location="runtime",
            model=self.rerank_model_name,
            status="success",
        )
        rerank = getattr(self._rerank_service, "rerank", None)
        if not callable(rerank):
            self.last_attempts = [attempt.model_copy(update={"status": "failed", "error": "rerank not supported"})]
            return candidates
        try:
            result = rerank(query, candidates)
        except RuntimeError as exc:
            self.last_attempts = [attempt.model_copy(update={"status": "failed", "error": str(exc)})]
            raise
        self.provider_name = getattr(self._rerank_service, "provider_name", self.provider_name)
        self.rerank_model_name = getattr(self._rerank_service, "rerank_model_name", self.rerank_model_name)
        self.last_provider = self.provider_name
        attempt = attempt.model_copy(update={"provider": self.provider_name, "model": self.rerank_model_name})
        self.last_attempts = [attempt]
        return cast(Sequence[CandidateLike], result)

    def rerank(self, query: str, candidates: Sequence[CandidateLike]) -> list[CandidateLike]:
        return list(self(query, list(candidates)))


class SearchBackedRetrievalFactory:
    def __init__(
        self,
        *,
        metadata_repo: SQLiteMetadataRepo,
        fts_repo: SQLiteFTSRepo,
        graph_repo: SQLiteGraphRepo,
    ) -> None:
        self._metadata_repo = metadata_repo
        self._fts_repo = fts_repo
        self._graph_repo = graph_repo
        self._query_understanding = QueryUnderstandingService()

    def full_text_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        return self._build_candidates_from_chunk_ids(
            [result.chunk_id for result in self._fts_repo.search(query, limit=12, doc_ids=source_scope or None)],
            source_kind="internal",
        )

    def vector_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        vector_repo = cast(object, self._metadata_repo)
        del vector_repo
        return []

    def section_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        query_terms = search_terms(query)
        candidates: list[RetrievedCandidate] = []
        for document in self._iter_documents(source_scope):
            source = self._metadata_repo.get_source(document.source_id)
            for chunk in self._metadata_repo.list_chunks(document.doc_id):
                segment = self._metadata_repo.get_segment(chunk.segment_id)
                if segment is None:
                    continue
                section_text = " ".join(segment.toc_path)
                overlap = keyword_overlap(query_terms, section_text)
                if overlap == 0:
                    continue
                candidates.append(
                    RetrievedCandidate(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        source_id=document.source_id,
                        text=chunk.text,
                        citation_anchor=chunk.citation_anchor,
                        score=float(overlap),
                        rank=1,
                        section_path=tuple(segment.toc_path),
                        effective_access_policy=chunk.effective_access_policy,
                        chunk_role=chunk.chunk_role,
                        special_chunk_type=chunk.special_chunk_type,
                        parent_chunk_id=chunk.parent_chunk_id,
                        metadata=dict(chunk.metadata),
                        file_name=self._resolve_file_name(document.title, None if source is None else source.location),
                        page_start=None if segment.page_range is None else segment.page_range[0],
                        page_end=None if segment.page_range is None else segment.page_range[1],
                        chunk_type=chunk.special_chunk_type or chunk.chunk_role.value,
                        source_type=None if source is None else source.source_type.value,
                    )
                )
        candidates.sort(key=lambda item: (-item.score, item.chunk_id))
        return candidates[:12]

    def special_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        query_terms = search_terms(query)
        lowered = query.lower()
        candidates: list[RetrievedCandidate] = []
        for document in self._iter_documents(source_scope):
            for chunk in self._metadata_repo.list_chunks(document.doc_id):
                if chunk.chunk_role is not ChunkRole.SPECIAL:
                    continue
                score = keyword_overlap(query_terms, chunk.text)
                special_type = chunk.special_chunk_type or ""
                if special_type and special_type in lowered:
                    score += 2
                if special_type == "table" and any(term in lowered for term in ("表格", "表", "指标", "数值")):
                    score += 2
                if special_type == "ocr_region" and any(term in lowered for term in ("ocr", "识别", "图片文字")):
                    score += 2
                if special_type == "image_summary" and any(term in lowered for term in ("图片", "图像", "画面")):
                    score += 1
                if score <= 0:
                    continue
                candidates.extend(
                    self._build_candidates_from_chunk_ids(
                        [chunk.chunk_id],
                        source_kind="internal",
                        scope=source_scope,
                    )
                )
                if candidates:
                    latest = candidates[-1]
                    candidates[-1] = RetrievedCandidate(
                        chunk_id=latest.chunk_id,
                        doc_id=latest.doc_id,
                        source_id=latest.source_id,
                        text=latest.text,
                        citation_anchor=latest.citation_anchor,
                        score=float(score),
                        rank=1,
                        source_kind=latest.source_kind,
                        section_path=latest.section_path,
                        effective_access_policy=latest.effective_access_policy,
                        chunk_role=latest.chunk_role,
                        special_chunk_type=latest.special_chunk_type,
                        parent_chunk_id=latest.parent_chunk_id,
                        parent_text=latest.parent_text,
                        metadata=latest.metadata,
                        file_name=latest.file_name,
                        page_start=latest.page_start,
                        page_end=latest.page_end,
                        chunk_type=latest.chunk_type,
                        source_type=latest.source_type,
                    )
        candidates.sort(key=lambda item: (-item.score, item.chunk_id))
        return candidates[:12]

    def metadata_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        understanding = self._query_understanding.analyze(query)
        page_numbers = set(self._as_str_list(understanding.metadata_filters.get("page_numbers")))
        source_types = set(self._as_str_list(understanding.metadata_filters.get("source_types")))
        preferred_sections = set(
            self._as_str_list(understanding.structure_constraints.get("preferred_section_terms"))
        )
        if not (page_numbers or source_types or preferred_sections or understanding.special_targets):
            return []

        candidates: list[RetrievedCandidate] = []
        for document in self._iter_documents(source_scope):
            source = self._metadata_repo.get_source(document.source_id)
            source_type = "" if source is None else source.source_type.value
            for chunk in self._metadata_repo.list_chunks(document.doc_id):
                if chunk.chunk_role is ChunkRole.PARENT:
                    continue
                segment = self._metadata_repo.get_segment(chunk.segment_id)
                section_text = "" if segment is None else " ".join(segment.toc_path)
                score = 0.0
                if source_types and source_type in source_types:
                    score += 3.0
                if preferred_sections and any(term in section_text for term in preferred_sections):
                    score += 3.5
                if page_numbers:
                    page_no = chunk.metadata.get("page_no", "")
                    if page_no in page_numbers:
                        score += 4.0
                    elif segment is not None and segment.page_range is not None:
                        start, end = segment.page_range
                        if any(start <= int(page) <= end for page in page_numbers):
                            score += 2.5
                if understanding.special_targets and chunk.special_chunk_type in understanding.special_targets:
                    score += 2.0
                if score <= 0:
                    continue
                candidates.extend(
                    self._build_candidates_from_chunk_ids(
                        [chunk.chunk_id],
                        source_kind="internal",
                        scope=source_scope,
                    )
                )
                if candidates:
                    latest = candidates[-1]
                    candidates[-1] = RetrievedCandidate(
                        chunk_id=latest.chunk_id,
                        doc_id=latest.doc_id,
                        source_id=latest.source_id,
                        text=latest.text,
                        citation_anchor=latest.citation_anchor,
                        score=score,
                        rank=1,
                        source_kind=latest.source_kind,
                        section_path=latest.section_path,
                        effective_access_policy=latest.effective_access_policy,
                        chunk_role=latest.chunk_role,
                        special_chunk_type=latest.special_chunk_type,
                        parent_chunk_id=latest.parent_chunk_id,
                        parent_text=latest.parent_text,
                        metadata=latest.metadata,
                        file_name=latest.file_name,
                        page_start=latest.page_start,
                        page_end=latest.page_end,
                        chunk_type=latest.chunk_type,
                        source_type=latest.source_type,
                    )
        candidates.sort(key=lambda item: (-item.score, item.chunk_id))
        return candidates[:12]

    def graph_expander(
        self,
        query: str,
        source_scope: list[str],
        non_graph_evidence: list[CandidateLike],
    ) -> list[RetrievedCandidate]:
        del query
        chunk_ids = {
            chunk_id for edge in self._graph_repo.list_candidate_edges() for chunk_id in edge.evidence_chunk_ids
        }
        seed_chunk_ids = [candidate.chunk_id for candidate in non_graph_evidence]
        if seed_chunk_ids:
            chunk_ids.update(seed_chunk_ids)
        return self._build_candidates_from_chunk_ids(list(chunk_ids), source_kind="graph", scope=source_scope)

    def web_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        del source_scope
        web_search = DeterministicWebSearchRepo()
        results = web_search.search(query)
        return [
            RetrievedCandidate(
                chunk_id=f"web-{index}",
                doc_id=f"web-doc-{index}",
                source_id=result.url,
                text=result.snippet,
                citation_anchor=result.title,
                score=result.score or 0.5,
                rank=index,
                source_kind="external",
                file_name=result.title,
                chunk_type="external",
                source_type="web",
            )
            for index, result in enumerate(results, start=1)
        ]

    def vector_retriever_from_repo(
        self,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingProviderBinding],
        *,
        default_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST,
    ) -> MultiProviderBackedVectorRetriever:
        return MultiProviderBackedVectorRetriever(
            factory=self,
            vector_repo=vector_repo,
            bindings=bindings,
            default_preference=default_preference,
        )

    def _iter_documents(self, source_scope: list[str]) -> list[Document]:
        documents = self._metadata_repo.list_documents(active_only=True)
        if not source_scope:
            return documents
        allowed = set(source_scope)
        return [document for document in documents if {document.doc_id, document.source_id} & allowed]

    @staticmethod
    def _as_str_list(value: list[str] | str | bool | None) -> list[str]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        if isinstance(value, str):
            return [value]
        return []

    @staticmethod
    def _resolve_file_name(title: str, location: str | None) -> str | None:
        if title.strip():
            return title.strip()
        if not location:
            return None
        if "://" in location:
            return location
        return Path(location).name or location

    def _build_candidates_from_chunk_ids(
        self,
        chunk_ids: list[str],
        *,
        source_kind: str,
        scope: list[str] | None = None,
    ) -> list[RetrievedCandidate]:
        candidates: list[RetrievedCandidate] = []
        allowed = set(scope or [])
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            chunk = self._metadata_repo.get_chunk(chunk_id)
            if chunk is None:
                continue
            document = self._metadata_repo.get_document(chunk.doc_id)
            if document is None:
                continue
            source = self._metadata_repo.get_source(document.source_id)
            if allowed and not ({document.doc_id, document.source_id} & allowed):
                continue
            segment = self._metadata_repo.get_segment(chunk.segment_id)
            candidates.append(
                RetrievedCandidate(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_id=document.source_id,
                    text=chunk.text,
                    citation_anchor=chunk.citation_anchor,
                    score=max(0.0, 1.0 - (rank - 1) * 0.05),
                    rank=rank,
                    source_kind=source_kind,
                    section_path=tuple(segment.toc_path) if segment is not None else (),
                    effective_access_policy=chunk.effective_access_policy,
                    chunk_role=chunk.chunk_role,
                    special_chunk_type=chunk.special_chunk_type,
                    parent_chunk_id=chunk.parent_chunk_id,
                    parent_text=(
                        None
                        if chunk.parent_chunk_id is None
                        else (
                            None
                            if (parent_chunk := self._metadata_repo.get_chunk(chunk.parent_chunk_id)) is None
                            else parent_chunk.text
                        )
                    ),
                    metadata=dict(chunk.metadata),
                    file_name=self._resolve_file_name(document.title, None if source is None else source.location),
                    page_start=(
                        None
                        if segment is None or segment.page_range is None
                        else segment.page_range[0]
                    ),
                    page_end=(
                        None
                        if segment is None or segment.page_range is None
                        else segment.page_range[1]
                    ),
                    chunk_type=chunk.special_chunk_type or chunk.chunk_role.value,
                    source_type=None if source is None else source.source_type.value,
                )
            )
        return candidates


class RetrievalRuntimeAdapter:
    def __init__(self, retrieval_service: RetrievalService) -> None:
        self._retrieval_service = retrieval_service
        self.last_result: RetrievalResult | None = None

    def retrieve(
        self,
        query: str,
        policy: ExecutionPolicy,
        mode: RuntimeMode,
        round_index: int,
    ) -> list[EvidenceItem]:
        del mode, round_index
        result = self._retrieval_service.retrieve(
            query,
            access_policy=policy.effective_access_policy,
            source_scope=policy.source_scope,
            execution_location_preference=policy.execution_location_preference,
        )
        self.last_result = result
        return result.evidence.all

    def rerank(
        self,
        query: str,
        hits: list[EvidenceItem],
        policy: ExecutionPolicy,
    ) -> list[EvidenceItem]:
        del query, policy
        return hits


class RuntimeEvidenceAdapter:
    def __init__(
        self,
        *,
        evidence_service: EvidenceService,
        artifact_service: ArtifactService,
        metadata_repo: SQLiteMetadataRepo,
        ingest_service: IngestService,
        telemetry_service: TelemetryService | None = None,
        cloud_providers: Sequence[object] = (),
        local_providers: Sequence[object] = (),
        answer_generation_service: AnswerGenerationService | None = None,
    ) -> None:
        self._evidence_service = evidence_service
        self._artifact_service = artifact_service
        self._metadata_repo = metadata_repo
        self._ingest_service = ingest_service
        self._telemetry_service = telemetry_service
        self._cloud_providers = tuple(cloud_providers)
        self._local_providers = tuple(local_providers)
        self._answer_generation_service = answer_generation_service or AnswerGenerationService()
        self._last_hits: list[EvidenceItem] = []
        self._last_policy: ExecutionPolicy | None = None
        self._last_model_diagnostics = ModelDiagnostics()
        self._last_synthesis_provider: object | None = None

    def fast_path_sufficient(self, hits: list[EvidenceItem], policy: ExecutionPolicy) -> bool:
        self._last_hits = list(hits)
        self._last_policy = policy
        bundle = self._bundle(hits)
        return self._evidence_service.evaluate_self_check(
            bundle=bundle,
            task_type=policy.task_type,
            complexity_level=policy.complexity_level,
        ).evidence_sufficient

    def detect_conflicts(self, hits: list[EvidenceItem]) -> list[str]:
        if len({item.doc_id for item in hits}) < 2:
            return []
        normalized = {" ".join(item.text.lower().split()) for item in hits}
        if len(normalized) <= 1:
            return []
        doc_ids = sorted({item.doc_id for item in hits})
        return [f"Sources {', '.join(doc_ids)} disagree on the queried claim."]

    def build_fast_response(self, query: str, hits: list[EvidenceItem]) -> QueryResponse:
        conflicts = self.detect_conflicts(hits)
        conclusion = self._extractive_conclusion(query, hits)
        answer = self._answer_generation_service.generate(
            query=query,
            evidence_pack=hits,
            runtime_mode=RuntimeMode.FAST,
            grounded_candidate=conclusion,
            trust_evidence_pack=True,
        )
        suggestion = self._persist_suggestion(query, RuntimeMode.FAST, hits, conflicts)
        return self._query_response_from_grounded_answer(
            answer=answer,
            evidence=hits,
            differences_or_conflicts=conflicts,
            uncertainty="low" if len(hits) >= 2 else "medium",
            preservation_suggestion=suggestion,
            runtime_mode=RuntimeMode.FAST,
        )

    def claim_citation_aligned(self, response: QueryResponse) -> bool:
        if response.insufficient_evidence_flag:
            return True
        if response.answer_text is not None or response.answer_sections or response.citations:
            return response.groundedness_flag
        if not response.evidence:
            return response.conclusion.lower().startswith("insufficient evidence")

        conclusion = response.conclusion.strip()
        if not conclusion:
            return False

        candidates = [conclusion]
        if ":" in conclusion:
            _, suffix = conclusion.split(":", 1)
            normalized_suffix = suffix.strip()
            if normalized_suffix:
                candidates.append(normalized_suffix)

        return any(self._text_supported_by_evidence(candidate, response.evidence) for candidate in candidates)

    def build_evidence_matrix(self, hits: list[EvidenceItem]) -> list[dict[str, object]]:
        self._last_hits = list(hits)
        matrix: list[dict[str, object]] = []
        for item in hits:
            matrix.append(
                {
                    "claim": item.text,
                    "sources": [item.doc_id],
                    "chunk_ids": [item.chunk_id],
                }
            )
        return matrix

    @staticmethod
    def _row_sources(row: dict[str, object]) -> list[str]:
        value = row.get("sources")
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _row_claim(row: dict[str, object]) -> str | None:
        value = row.get("claim")
        return value if isinstance(value, str) else None

    def evidence_sufficient(self, evidence_matrix: list[dict[str, object]], round_index: int) -> bool:
        del round_index
        doc_ids = {sources[0] for row in evidence_matrix if (sources := self._row_sources(row))}
        return len(evidence_matrix) >= 2 and len(doc_ids) >= 1

    def build_deep_response(
        self,
        query: str,
        evidence_matrix: list[dict[str, object]],
        *,
        location: str,
    ) -> QueryResponse:
        hits = self._last_hits
        conflicts = self.detect_conflicts(hits)
        summary = self._build_synthesis_summary(query, evidence_matrix)
        grounded_candidate = self._extractive_conclusion(query, hits)
        conclusion = self._synthesize_conclusion(query, summary, location, grounded_candidate)
        answer = self._answer_generation_service.answer_from_model_output(
            query=query,
            evidence_pack=hits,
            grounded_candidate=grounded_candidate,
            model_output=conclusion,
            enforce_grounding=False,
            trust_evidence_pack=True,
        )
        response = self._query_response_from_grounded_answer(
            answer=answer,
            evidence=hits,
            differences_or_conflicts=conflicts,
            uncertainty="medium" if conflicts else "low",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=RuntimeMode.DEEP,
            diagnostics=QueryDiagnostics(model=self._last_model_diagnostics),
        )
        if not self.claim_citation_aligned(response):
            if self._telemetry_service is not None:
                self._telemetry_service.record_claim_citation_failure(
                    response_mode=RuntimeMode.DEEP.value,
                    evidence_count=len(hits),
                )
            repaired_conclusion = self._repair_unaligned_conclusion(
                query=query,
                summary=summary,
                grounded_candidate=grounded_candidate,
                rejected_conclusion=answer.answer_text,
            )
            if repaired_conclusion is not None:
                repaired_answer = self._answer_generation_service.answer_from_model_output(
                    query=query,
                    evidence_pack=hits,
                    grounded_candidate=grounded_candidate,
                    model_output=repaired_conclusion,
                    enforce_grounding=False,
                    trust_evidence_pack=True,
                )
                repaired_response = self._query_response_from_grounded_answer(
                    answer=repaired_answer,
                    evidence=hits,
                    differences_or_conflicts=conflicts,
                    uncertainty="medium" if conflicts else "low",
                    preservation_suggestion=PreservationSuggestion(suggested=False),
                    runtime_mode=RuntimeMode.DEEP,
                    diagnostics=QueryDiagnostics(model=self._last_model_diagnostics),
                )
                if self.claim_citation_aligned(repaired_response):
                    suggestion = self._persist_suggestion(query, RuntimeMode.DEEP, hits, conflicts)
                    return repaired_response.model_copy(update={"preservation_suggestion": suggestion})

            self._last_model_diagnostics = self._last_model_diagnostics.model_copy(
                update={
                    "fallback_reason": "citation_alignment_failed",
                    "failed_stage": "citation_alignment",
                    "degraded_to_retrieval_only": False,
                }
            )
            fallback_answer = self._answer_generation_service.generate(
                query=query,
                evidence_pack=hits,
                runtime_mode=RuntimeMode.DEEP,
                grounded_candidate=grounded_candidate,
                trust_evidence_pack=True,
            )
            fallback_response = self._query_response_from_grounded_answer(
                answer=fallback_answer,
                evidence=hits,
                differences_or_conflicts=conflicts,
                uncertainty="medium" if conflicts else "low",
                preservation_suggestion=PreservationSuggestion(suggested=False),
                runtime_mode=RuntimeMode.DEEP,
                diagnostics=QueryDiagnostics(model=self._last_model_diagnostics),
            )
            suggestion = self._persist_suggestion(query, RuntimeMode.DEEP, hits, conflicts)
            return fallback_response.model_copy(update={"preservation_suggestion": suggestion})

        suggestion = self._persist_suggestion(query, RuntimeMode.DEEP, hits, conflicts)
        return response.model_copy(update={"preservation_suggestion": suggestion})

    @property
    def last_model_diagnostics(self) -> ModelDiagnostics:
        return self._last_model_diagnostics

    def build_retrieval_only_response(self, query: str, hits: list[EvidenceItem]) -> QueryResponse:
        diagnostics = self._fallback_model_diagnostics(hits)
        answer = self._answer_generation_service.generate(
            query=query,
            evidence_pack=hits,
            runtime_mode=RuntimeMode.DEEP,
            grounded_candidate=hits[0].text if hits else None,
            trust_evidence_pack=True,
        )
        return self._query_response_from_grounded_answer(
            answer=answer,
            evidence=hits,
            differences_or_conflicts=self.detect_conflicts(hits),
            uncertainty="high",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=RuntimeMode.DEEP,
            diagnostics=QueryDiagnostics(model=diagnostics),
        )

    @staticmethod
    def _query_response_from_grounded_answer(
        *,
        answer: object,
        evidence: list[EvidenceItem],
        differences_or_conflicts: list[str],
        uncertainty: str,
        preservation_suggestion: PreservationSuggestion,
        runtime_mode: RuntimeMode,
        diagnostics: QueryDiagnostics | None = None,
    ) -> QueryResponse:
        grounded = cast(GroundedAnswer, answer)
        return QueryResponse(
            conclusion=grounded.answer_text,
            evidence=evidence,
            differences_or_conflicts=differences_or_conflicts,
            uncertainty=uncertainty,
            preservation_suggestion=preservation_suggestion,
            runtime_mode=runtime_mode,
            diagnostics=diagnostics or QueryDiagnostics(),
            answer_text=grounded.answer_text,
            answer_sections=grounded.answer_sections,
            citations=grounded.citations,
            evidence_links=grounded.evidence_links,
            groundedness_flag=grounded.groundedness_flag,
            insufficient_evidence_flag=grounded.insufficient_evidence_flag,
        )

    def _bundle(self, hits: list[EvidenceItem]) -> EvidenceBundle:
        internal = [item for item in hits if item.evidence_kind == "internal"]
        external = [item for item in hits if item.evidence_kind == "external"]
        graph = [item for item in hits if item.evidence_kind == "graph"]
        return EvidenceBundle(internal=internal, external=external, graph=graph)

    def _persist_suggestion(
        self,
        query: str,
        runtime_mode: RuntimeMode,
        hits: list[EvidenceItem],
        conflicts: list[str],
    ) -> PreservationSuggestion:
        suggestion = self._artifact_service.suggest_preservation(
            query=query,
            runtime_mode=runtime_mode,
            evidence=hits,
            differences_or_conflicts=conflicts,
        )
        if not suggestion.suggested:
            return suggestion

        artifact = self._artifact_service.build_artifact(
            query=query,
            suggestion=suggestion,
            evidence=hits,
            differences_or_conflicts=conflicts,
        )
        list_artifacts = getattr(self._metadata_repo, "list_artifacts", None)
        existing_artifacts = list_artifacts() if callable(list_artifacts) else []
        lifecycle_artifacts = self._artifact_service.apply_lifecycle(
            proposed=artifact,
            existing_artifacts=[
                item
                for item in existing_artifacts
                if item.artifact_type is artifact.artifact_type and item.title == artifact.title
            ],
        )
        for item in lifecycle_artifacts:
            self._metadata_repo.save_artifact(item)
        return suggestion.model_copy(update={"artifact_id": artifact.artifact_id})

    def _synthesize_conclusion(
        self,
        query: str,
        summary: str,
        location: str,
        grounded_candidate: str,
    ) -> str:
        providers = self._provider_order(location)
        self._last_synthesis_provider = None
        if not providers:
            self._last_model_diagnostics = ModelDiagnostics(
                synthesis_provider=None,
                degraded_to_retrieval_only=True,
                failed_stage="synthesis",
            )
            return grounded_candidate

        prompt = self._build_synthesis_prompt(query, summary, grounded_candidate)
        last_error: RuntimeError | None = None
        failed_cloud_provider_count = 0
        attempts: list[ProviderAttempt] = []
        for provider in providers:
            chat = getattr(provider, "chat", None)
            if not callable(chat):
                continue
            attempt = ProviderAttempt(
                stage="synthesis",
                capability="chat",
                provider=provider_name(provider),
                location=self._provider_location(provider, location),
                model=provider_model(provider, "chat"),
                status="success",
            )
            try:
                response = chat(prompt)
            except RuntimeError as exc:
                last_error = exc
                attempts.append(attempt.model_copy(update={"status": "failed", "error": str(exc)}))
                if location == "cloud" and provider in self._cloud_providers:
                    failed_cloud_provider_count += 1
                continue
            if isinstance(response, str) and response.strip():
                attempts.append(attempt)
                fallback_reason = (
                    "cloud_provider_failed"
                    if location == "cloud" and provider in self._local_providers and failed_cloud_provider_count > 0
                    else None
                )
                self._last_synthesis_provider = provider
                self._last_model_diagnostics = ModelDiagnostics(
                    synthesis_provider=provider_name(provider),
                    attempts=attempts,
                    fallback_reason=fallback_reason,
                )
                if (
                    self._telemetry_service is not None
                    and location == "cloud"
                    and failed_cloud_provider_count > 0
                    and provider in self._local_providers
                ):
                    self._telemetry_service.record_local_fallback(
                        from_location="cloud",
                        to_location="local",
                        failed_provider_count=failed_cloud_provider_count,
                    )
                return response

        self._last_model_diagnostics = ModelDiagnostics(
            attempts=attempts,
            degraded_to_retrieval_only=True,
            failed_stage="synthesis",
            fallback_reason=(
                "cloud_provider_failed"
                if location == "cloud" and failed_cloud_provider_count > 0
                else None
            ),
        )
        if last_error is not None:
            raise last_error
        raise RuntimeError("No synthesis provider available")

    def _repair_unaligned_conclusion(
        self,
        *,
        query: str,
        summary: str,
        grounded_candidate: str,
        rejected_conclusion: str,
    ) -> str | None:
        provider = self._last_synthesis_provider
        if provider is None:
            return None
        chat = getattr(provider, "chat", None)
        if not callable(chat):
            return None

        attempt = ProviderAttempt(
            stage="citation_repair",
            capability="chat",
            provider=provider_name(provider),
            location=self._provider_location(provider, "local"),
            model=provider_model(provider, "chat"),
            status="success",
        )
        prompt = self._build_alignment_repair_prompt(
            query=query,
            grounded_candidate=grounded_candidate,
            summary=summary,
            rejected_conclusion=rejected_conclusion,
        )
        try:
            repaired = chat(prompt)
        except RuntimeError as exc:
            self._append_model_attempt(
                attempt.model_copy(update={"status": "failed", "error": str(exc)}),
            )
            return None
        normalized = str(repaired).strip()
        if not normalized:
            self._append_model_attempt(
                attempt.model_copy(update={"status": "failed", "error": "provider returned empty repair"}),
            )
            return None
        self._append_model_attempt(attempt)
        self._last_model_diagnostics = self._last_model_diagnostics.model_copy(
            update={
                "fallback_reason": "citation_alignment_repaired",
                "failed_stage": None,
                "degraded_to_retrieval_only": False,
            }
        )
        return normalized

    def _provider_order(self, location: str) -> tuple[object, ...]:
        if location == "cloud":
            return (*self._cloud_providers, *self._local_providers)
        if location == "local":
            return tuple(self._local_providers)
        return ()

    def _provider_location(self, provider: object, requested_location: str) -> str:
        if provider in self._cloud_providers:
            return "cloud"
        if provider in self._local_providers:
            return "local"
        return requested_location

    def _fallback_model_diagnostics(self, hits: list[EvidenceItem]) -> ModelDiagnostics:
        if self._last_model_diagnostics.attempts:
            return self._last_model_diagnostics.model_copy(
                update={
                    "degraded_to_retrieval_only": True,
                    "failed_stage": self._last_model_diagnostics.failed_stage or "synthesis",
                }
            )
        if hits:
            return ModelDiagnostics(
                degraded_to_retrieval_only=True,
                failed_stage="synthesis",
            )
        return ModelDiagnostics(
            degraded_to_retrieval_only=True,
            failed_stage="retrieval",
        )

    @staticmethod
    def _build_synthesis_prompt(query: str, summary: str, grounded_candidate: str) -> str:
        return "\n".join(
            [
                grounded_candidate,
                "",
                f"Question: {query}",
                "Answer using only the supported evidence below.",
                "Rules:",
                "- Use the same language as the question.",
                "- Return 1 or 2 plain sentences.",
                "- Prefer exact phrases from the evidence.",
                "- Do not translate module names or invent new labels.",
                "- Do not use bullets, markdown emphasis, or examples.",
                f"Grounded seed: {grounded_candidate}",
                "Evidence:",
                summary,
            ]
        )

    @staticmethod
    def _build_alignment_repair_prompt(
        *,
        query: str,
        grounded_candidate: str,
        summary: str,
        rejected_conclusion: str,
    ) -> str:
        return "\n".join(
            [
                grounded_candidate,
                "",
                f"Question: {query}",
                "The previous answer introduced wording that is not directly supported.",
                "Rewrite it so every phrase is supported by the evidence.",
                "Rules:",
                "- Use the same language as the question.",
                "- Return 1 or 2 plain sentences.",
                "- Reuse exact phrases from the evidence whenever possible.",
                "- Do not use bullets, markdown emphasis, or translated labels.",
                f"Grounded seed: {grounded_candidate}",
                f"Previous answer: {rejected_conclusion}",
                "Evidence:",
                summary,
            ]
        )

    def _append_model_attempt(self, attempt: ProviderAttempt) -> None:
        self._last_model_diagnostics = self._last_model_diagnostics.model_copy(
            update={"attempts": [*self._last_model_diagnostics.attempts, attempt]}
        )

    @staticmethod
    def _build_synthesis_summary(query: str, evidence_matrix: list[dict[str, object]]) -> str:
        allow_command_like = looks_command_like(query) or looks_operation_query(query)
        claims: list[str] = []
        seen: set[str] = set()
        for row in evidence_matrix:
            claim = RuntimeEvidenceAdapter._row_claim(row)
            if claim is None:
                continue
            normalized = " ".join(claim.split())
            if not normalized or normalized in seen:
                continue
            if not allow_command_like and looks_command_like(normalized):
                continue
            seen.add(normalized)
            claims.append(normalized)
            if len(claims) >= 4:
                break
        if not claims:
            for row in evidence_matrix[:4]:
                claim = RuntimeEvidenceAdapter._row_claim(row)
                if claim is None:
                    continue
                normalized = " ".join(claim.split())
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    claims.append(normalized)
        return "\n".join(f"- {claim}" for claim in claims) if claims else "Insufficient evidence."

    @staticmethod
    def _normalize_supported_text(text: str) -> str:
        normalized = text.replace("**", " ").replace("__", " ").replace("`", " ")
        normalized = re.sub(r"^\s*[-*]\s*", "", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"^\s*\d+\.\s*", "", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _text_supported_by_evidence(conclusion: str, evidence: list[EvidenceItem]) -> bool:
        normalized_conclusion = RuntimeEvidenceAdapter._normalize_supported_text(conclusion)
        if not normalized_conclusion:
            return False
        conclusion_sentences = split_sentences(normalized_conclusion) or [normalized_conclusion]
        normalized_evidence = [RuntimeEvidenceAdapter._normalize_supported_text(item.text) for item in evidence]

        def _supported(sentence: str) -> bool:
            sentence_terms = search_terms(sentence)
            required_overlap = max(2, (len(sentence_terms) + 1) // 2) if sentence_terms else 0
            if any(
                sentence in evidence_text
                or (
                    required_overlap > 0
                    and keyword_overlap(sentence_terms, evidence_text) >= required_overlap
                )
                for evidence_text in normalized_evidence
            ):
                return True
            if required_overlap == 0:
                return False
            combined_evidence = " ".join(normalized_evidence)
            return keyword_overlap(sentence_terms, combined_evidence) >= required_overlap

        return all(_supported(sentence) for sentence in conclusion_sentences if sentence)

    @staticmethod
    def _extractive_conclusion(query: str, hits: list[EvidenceItem]) -> str:
        if not hits:
            return "Insufficient evidence in indexed sources."

        operation_conclusion = RuntimeEvidenceAdapter._operation_aware_conclusion(query, hits)
        if operation_conclusion is not None:
            return operation_conclusion

        structure_conclusion = RuntimeEvidenceAdapter._structure_aware_conclusion(query, hits)
        if structure_conclusion is not None:
            return structure_conclusion

        query_terms = search_terms(query)
        query_focus_terms = focus_terms(query)
        query_is_command_like = looks_command_like(query)
        query_is_definition_like = looks_definition_query(query)
        query_is_structure_like = looks_structure_query(query)
        normalized_query = query.strip().lower()
        sentences = [sentence for item in hits[:5] for sentence in split_sentences(item.text)]
        if not sentences:
            return hits[0].text

        def _score(sentence: str) -> float:
            score = float(keyword_overlap(query_terms, sentence))
            score += keyword_overlap(query_focus_terms, sentence) * 0.7
            if normalized_query and normalized_query in sentence.lower():
                score += 2.0
            if not query_is_command_like and looks_command_like(sentence):
                score -= 5.0
            if (
                query_is_definition_like
                and not query_is_structure_like
                and not looks_command_like(sentence)
                and looks_definition_text(sentence)
            ):
                score += 4.0
            if query_is_structure_like and looks_structure_text(sentence):
                score += 5.0
            return score

        non_command_sentences = [sentence for sentence in sentences if not looks_command_like(sentence)]
        candidate_pool = (
            non_command_sentences
            if (query_is_definition_like or not query_is_command_like) and non_command_sentences
            else sentences
        )
        return max(candidate_pool, key=_score)

    @staticmethod
    def _operation_aware_conclusion(query: str, hits: list[EvidenceItem]) -> str | None:
        if not looks_operation_query(query):
            return None

        query_focus_terms = focus_terms(query)
        segments: list[str] = []
        seen: set[str] = set()
        for hit in hits[:12]:
            if RuntimeEvidenceAdapter._operation_fragment_score(hit.text, query_focus_terms) < 2:
                continue
            normalized = RuntimeEvidenceAdapter._normalize_operation_fragment(hit.text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            segments.append(normalized)
        if not segments:
            return None
        return "；".join(segments[:4])

    @staticmethod
    def _operation_fragment_score(text: str, query_focus_terms: Sequence[str]) -> int:
        lowered = text.lower()
        score = 0
        if looks_operation_text(text):
            score += 1
        if any(
            marker in lowered
            for marker in (
                "ollama",
                "openai",
                "local_only",
                "cloud_first",
                ".env",
                "uv sync",
                "安装依赖",
                "本地模式",
                "云端模式",
            )
        ):
            score += 2
        if looks_definition_text(text):
            score -= 2
        if looks_structure_text(text):
            score -= 3
        score += min(2, keyword_overlap(query_focus_terms, text))
        return score

    @staticmethod
    def _structure_aware_conclusion(query: str, hits: list[EvidenceItem]) -> str | None:
        if not looks_structure_query(query):
            return None

        query_focus_terms = focus_terms(query)
        lead = None
        scored_hits = sorted(
            hits[:5],
            key=lambda item: (
                keyword_overlap(query_focus_terms, item.citation_anchor) * 3.0
                + keyword_overlap(query_focus_terms, item.text) * 1.2
                + (
                    4.0
                    if looks_structure_text(item.citation_anchor)
                    and keyword_overlap(query_focus_terms, item.citation_anchor) > 0
                    else 0.0
                )
                + (
                    2.0
                    if looks_structure_text(item.text) and keyword_overlap(query_focus_terms, item.text) > 0
                    else 0.0
                )
                + float(item.score)
            ),
            reverse=True,
        )

        if RuntimeEvidenceAdapter._prefers_layer_signature_query(query):
            for hit in scored_hits:
                signature = RuntimeEvidenceAdapter._extract_layer_signature(hit.text)
                if signature is not None:
                    return signature

        for hit in scored_hits:
            lead = RuntimeEvidenceAdapter._pick_structure_lead(hit.text, query_focus_terms)
            if lead:
                break

        for hit in scored_hits:
            if keyword_overlap(query_focus_terms, hit.text) == 0 and keyword_overlap(
                query_focus_terms, hit.citation_anchor
            ) == 0:
                continue
            bullets = RuntimeEvidenceAdapter._extract_structure_points(hit.text)
            hit_lead = RuntimeEvidenceAdapter._pick_structure_lead(hit.text, query_focus_terms)
            if not hit_lead and not bullets and not lead:
                continue
            if bullets:
                segments = [lead] if lead else []
                segments.extend(bullets[:6])
                return "；".join(segment for segment in segments if segment)
            if hit_lead:
                return hit_lead
        if lead:
            return lead
        return None

    @staticmethod
    def _pick_structure_lead(text: str, query_focus_terms: Sequence[str]) -> str | None:
        signature = RuntimeEvidenceAdapter._extract_layer_signature(text)
        if signature is not None:
            return signature
        sentences = split_sentences(text)
        for sentence in sentences:
            if (
                keyword_overlap(query_focus_terms, sentence) > 0
                and looks_structure_text(sentence)
                and not looks_command_like(sentence)
            ):
                return RuntimeEvidenceAdapter._normalize_answer_fragment(sentence)
        for sentence in sentences:
            if looks_structure_text(sentence) and not looks_command_like(sentence):
                return RuntimeEvidenceAdapter._normalize_answer_fragment(sentence)
        return None

    @staticmethod
    def _extract_structure_points(text: str) -> list[str]:
        segments: list[str] = []
        parts = [part.strip() for part in re.split(r"(?=\s*-\s+)", text) if part.strip()]
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("-"):
                cleaned = cleaned[1:].strip()
            if not cleaned or looks_command_like(cleaned):
                continue
            if "：" not in cleaned and ":" not in cleaned:
                continue
            segments.append(RuntimeEvidenceAdapter._normalize_answer_fragment(cleaned))
        return segments

    @staticmethod
    def _normalize_answer_fragment(text: str) -> str:
        cleaned = " ".join(text.replace("`", "").split())
        if cleaned.endswith(("。", "！", "？", ".", "!", "?")):
            return cleaned[:-1].strip()
        return cleaned.strip()

    @staticmethod
    def _normalize_operation_fragment(text: str) -> str:
        normalized = RuntimeEvidenceAdapter._normalize_answer_fragment(text)
        lowered = normalized.lower()
        if "pkp_openai__" in lowered and "OpenAI" not in normalized:
            normalized = f"OpenAI：{normalized}"
        if "pkp_ollama__" in lowered and "Ollama" not in normalized:
            normalized = f"Ollama：{normalized}"
        return normalized

    @staticmethod
    def _prefers_layer_signature_query(query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in ("架构", "architecture", "分层", "layer", "layers"))

    @staticmethod
    def _extract_layer_signature(text: str) -> str | None:
        normalized = text.replace("`", " ")
        match = re.search(
            r"([A-Za-z][A-Za-z0-9 ]+(?:\s*->\s*[A-Za-z][A-Za-z0-9 ]+){2,})",
            normalized,
        )
        if match is None:
            return None
        return RuntimeEvidenceAdapter._normalize_answer_fragment(match.group(1))


class ResearchPlannerAdapter:
    def decompose(self, query: str, memory_hints: Sequence[str] = ()) -> list[str]:
        lowered = query.lower()
        if (
            "compare" in lowered
            or "difference" in lowered
            or "conflict" in lowered
            or "比较" in query
            or "差异" in query
            or "冲突" in query
        ):
            return [query, f"{query} evidence"]
        return [query]

    def expand(
        self,
        query: str,
        evidence_matrix: list[dict[str, object]],
        round_index: int,
        memory_hints: Sequence[str] = (),
    ) -> list[str]:
        if evidence_matrix or round_index >= 2:
            return []
        expansions = [f"{query} details"]
        expansions.extend(f"{query} {hint}" for hint in memory_hints[:2] if hint)
        return expansions


class ArtifactApprovalAdapter:
    def __init__(self, metadata_repo: SQLiteMetadataRepo) -> None:
        self._metadata_repo = metadata_repo

    def list_artifacts(self) -> list[KnowledgeArtifact]:
        return self._metadata_repo.list_artifacts()

    def get_artifact(self, artifact_id: str) -> KnowledgeArtifact | None:
        return self._metadata_repo.get_artifact(artifact_id)

    def approve(self, artifact_id: str) -> KnowledgeArtifact:
        artifact = self._metadata_repo.get_artifact(artifact_id)
        if artifact is None:
            raise ValueError(f"Unknown artifact: {artifact_id}")
        approved = artifact.model_copy(update={"status": ArtifactStatus.APPROVED})
        self._metadata_repo.save_artifact(approved)
        return approved


class ArtifactIndexerAdapter:
    def __init__(self, ingest_service: IngestService) -> None:
        self._ingest_service = ingest_service

    def index_artifact(self, artifact: KnowledgeArtifact) -> None:
        self._ingest_service.ingest_plain_text(
            location=f"artifact://{artifact.artifact_id}.md",
            text=artifact.body_markdown,
            owner="system",
            title=artifact.title,
        )
