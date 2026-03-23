from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import VectorSearchResult
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.search.web_search_repo import DeterministicWebSearchRepo
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.service.artifact_service import ArtifactService
from pkp.service.evidence_service import CandidateLike, EvidenceBundle, EvidenceService
from pkp.service.ingest_service import IngestService
from pkp.service.retrieval_service import RetrievalResult, RetrievalService
from pkp.service.telemetry_service import TelemetryService
from pkp.types import (
    ArtifactStatus,
    Document,
    EvidenceItem,
    ExecutionPolicy,
    KnowledgeArtifact,
    PreservationSuggestion,
    QueryResponse,
    RuntimeMode,
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


class VectorSearchRepoProtocol(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
    ) -> Sequence[VectorSearchResult]: ...


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
        keywords = {token.lower() for token in query.split() if len(token) > 3}
        candidates: list[RetrievedCandidate] = []
        for document in self._iter_documents(source_scope):
            for chunk in self._metadata_repo.list_chunks(document.doc_id):
                segment = self._metadata_repo.get_segment(chunk.segment_id)
                if segment is None:
                    continue
                section_text = " ".join(segment.toc_path).lower()
                overlap = sum(1 for token in keywords if token in section_text)
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
                    )
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
            )
            for index, result in enumerate(results, start=1)
        ]

    def vector_retriever_from_repo(
        self,
        vector_repo: VectorSearchRepoProtocol,
    ) -> Callable[[str, list[str]], Sequence[RetrievedCandidate]]:
        def _retrieve(query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
            results = vector_repo.search(query, limit=12, doc_ids=source_scope or None)
            ordered_chunk_ids = [result.chunk_id for result in results]
            score_map = {result.chunk_id: float(result.score) for result in results}
            candidates = self._build_candidates_from_chunk_ids(
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
                )
                for index, candidate in enumerate(candidates, start=1)
            ]

        return _retrieve

    def _iter_documents(self, source_scope: list[str]) -> list[Document]:
        documents = self._metadata_repo.list_documents(active_only=True)
        if not source_scope:
            return documents
        allowed = set(source_scope)
        return [document for document in documents if {document.doc_id, document.source_id} & allowed]

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
    ) -> None:
        self._evidence_service = evidence_service
        self._artifact_service = artifact_service
        self._metadata_repo = metadata_repo
        self._ingest_service = ingest_service
        self._telemetry_service = telemetry_service
        self._cloud_providers = tuple(cloud_providers)
        self._local_providers = tuple(local_providers)
        self._last_hits: list[EvidenceItem] = []
        self._last_policy: ExecutionPolicy | None = None

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
        conclusion = hits[0].text if hits else "Insufficient evidence."
        suggestion = self._persist_suggestion(query, RuntimeMode.FAST, hits, conflicts)
        return QueryResponse(
            conclusion=conclusion,
            evidence=hits,
            differences_or_conflicts=conflicts,
            uncertainty="low" if len(hits) >= 2 else "medium",
            preservation_suggestion=suggestion,
            runtime_mode=RuntimeMode.FAST,
        )

    def claim_citation_aligned(self, response: QueryResponse) -> bool:
        return bool(response.evidence)

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
        summary_parts: list[str] = []
        for row in evidence_matrix:
            sources = self._row_sources(row)
            claim = self._row_claim(row)
            if sources and claim:
                summary_parts.append(f"{sources[0]}: {claim}")
        summary = " | ".join(summary_parts) if summary_parts else "Insufficient evidence."
        conclusion = self._synthesize_conclusion(query, summary, location)
        suggestion = self._persist_suggestion(query, RuntimeMode.DEEP, hits, conflicts)
        return QueryResponse(
            conclusion=conclusion,
            evidence=hits,
            differences_or_conflicts=conflicts,
            uncertainty="medium" if conflicts else "low",
            preservation_suggestion=suggestion,
            runtime_mode=RuntimeMode.DEEP,
        )

    def build_retrieval_only_response(self, query: str, hits: list[EvidenceItem]) -> QueryResponse:
        del query
        return QueryResponse(
            conclusion=hits[0].text if hits else "Insufficient evidence.",
            evidence=hits,
            differences_or_conflicts=self.detect_conflicts(hits),
            uncertainty="high",
            preservation_suggestion=PreservationSuggestion(suggested=False),
            runtime_mode=RuntimeMode.DEEP,
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
        self._metadata_repo.save_artifact(artifact)
        return suggestion.model_copy(update={"artifact_id": artifact.artifact_id})

    def _synthesize_conclusion(self, query: str, summary: str, location: str) -> str:
        providers = self._provider_order(location)
        if not providers:
            return summary

        prompt = self._build_synthesis_prompt(query, summary)
        last_error: RuntimeError | None = None
        failed_cloud_provider_count = 0
        for provider in providers:
            chat = getattr(provider, "chat", None)
            if not callable(chat):
                continue
            try:
                response = chat(prompt)
            except RuntimeError as exc:
                last_error = exc
                if location == "cloud" and provider in self._cloud_providers:
                    failed_cloud_provider_count += 1
                continue
            if isinstance(response, str) and response.strip():
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

        if last_error is not None:
            raise last_error
        raise RuntimeError("No synthesis provider available")

    def _provider_order(self, location: str) -> tuple[object, ...]:
        if location == "cloud":
            return (*self._cloud_providers, *self._local_providers)
        if location == "local":
            return tuple(self._local_providers)
        return ()

    @staticmethod
    def _build_synthesis_prompt(query: str, summary: str) -> str:
        return "\n".join(
            [
                query,
                "",
                "Synthesize the evidence into a concise answer.",
                summary,
            ]
        )


class ResearchPlannerAdapter:
    def decompose(self, query: str) -> list[str]:
        lowered = query.lower()
        if "compare" in lowered or "difference" in lowered or "conflict" in lowered:
            return [query, f"{query} evidence"]
        return [query]

    def expand(self, query: str, evidence_matrix: list[dict[str, object]], round_index: int) -> list[str]:
        if evidence_matrix or round_index >= 2:
            return []
        return [f"{query} details"]


class ArtifactApprovalAdapter:
    def __init__(self, metadata_repo: SQLiteMetadataRepo) -> None:
        self._metadata_repo = metadata_repo

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
