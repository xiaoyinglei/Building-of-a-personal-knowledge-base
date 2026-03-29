from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pkp.repo.graph.sqlite_graph_repo import SQLiteGraphRepo
from pkp.repo.interfaces import EmbeddingProviderBinding, VectorSearchResult
from pkp.repo.search.sqlite_fts_repo import SQLiteFTSRepo
from pkp.repo.search.web_search_repo import DeterministicWebSearchRepo
from pkp.repo.storage.sqlite_metadata_repo import SQLiteMetadataRepo
from pkp.service.evidence_service import CandidateLike
from pkp.service.query_understanding_service import QueryUnderstandingService
from pkp.types import AccessPolicy, Document, ExecutionLocationPreference
from pkp.types.content import ChunkRole
from pkp.types.text import keyword_overlap, search_terms

_GRAPH_RELATION_WEIGHTS = {
    "depends_on": 1.18,
    "supports": 1.12,
    "requires": 1.1,
    "uses": 1.02,
    "tabulated_in": 1.08,
    "illustrated_by": 1.06,
    "captioned_by": 0.98,
    "summarized_by": 0.96,
    "observed_in": 0.94,
    "expressed_by_formula": 1.04,
    "contains_special": 0.92,
    "enables": 1.0,
    "contains": 0.96,
    "part_of": 0.94,
    "integrates_with": 0.92,
    "compares": 0.78,
    "related_to": 0.68,
}

_GRAPH_QUERY_HINTS = {
    "depends_on": ("depend", "dependency", "依赖", "upstream"),
    "supports": ("support", "supports", "支撑", "supporting"),
    "uses": ("use", "uses", "used", "使用"),
    "tabulated_in": ("table", "表", "表格", "metric", "metrics", "数值", "指标"),
    "illustrated_by": ("figure", "diagram", "image", "图片", "图", "图像"),
    "captioned_by": ("caption", "图注", "标题"),
    "summarized_by": ("summary", "visual", "image summary", "摘要", "画面"),
    "observed_in": ("ocr", "region", "识别", "截图文字"),
    "expressed_by_formula": ("formula", "equation", "latex", "公式", "表达式"),
    "contains_special": ("table", "figure", "image", "caption", "ocr", "表格", "图片"),
    "contains": ("contain", "contains", "include", "includes", "包含"),
    "part_of": ("part", "belongs", "归属"),
    "integrates_with": ("integrate", "connection", "connect", "link", "连接"),
}

_MULTIMODAL_NODE_TYPES = {"table", "figure", "caption", "ocr_region", "image_summary", "formula"}


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
        item_kind: str = "chunk",
    ) -> Sequence[VectorSearchResult]: ...

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_chunks: bool = False,
    ) -> int: ...


class MultiProviderBackedVectorRetriever:
    def __init__(
        self,
        *,
        factory: SearchBackedRetrievalFactory,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingProviderBinding],
        item_kind: str = "chunk",
        candidate_builder: Callable[[str, list[VectorSearchResult], list[str]], list[RetrievedCandidate]] | None = None,
        default_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST,
    ) -> None:
        self._factory = factory
        self._vector_repo = vector_repo
        self._bindings = tuple(bindings)
        self._item_kind = item_kind
        self._candidate_builder = candidate_builder or factory.build_chunk_candidates_from_vector_results
        self._default_preference = default_preference
        self._prepared_locations: tuple[str, ...] = ("local", "cloud")
        self.last_provider: str | None = None
        self.last_attempts: list[object] = []

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
        self._prepared_locations = (
            ("local", "cloud")
            if preference is ExecutionLocationPreference.LOCAL_FIRST
            else ("cloud", "local")
        )

    def __call__(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        self.last_provider = None
        self.last_attempts = []
        ordered_bindings = self._ordered_bindings()
        for binding in ordered_bindings:
            candidates = self._search_binding(binding, query=query, source_scope=source_scope, target_space=binding.space)
            if candidates:
                self.last_provider = self._provider_name(binding.provider)
                return candidates
        for binding in ordered_bindings:
            candidates = self._search_binding(binding, query=query, source_scope=source_scope, target_space="default")
            if candidates:
                self.last_provider = self._provider_name(binding.provider)
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
        embed = getattr(binding.provider, "embed", None)
        if not callable(embed):
            return []
        try:
            query_vectors = cast(list[list[float]], embed([query]))
        except RuntimeError:
            return []
        if not query_vectors:
            return []
        if self._vector_repo.count_vectors(embedding_space=target_space, item_kind=self._item_kind) == 0:
            return []
        results = self._vector_repo.search(
            query_vectors[0],
            limit=12,
            doc_ids=source_scope or None,
            embedding_space=target_space,
            item_kind=self._item_kind,
        )
        if not results:
            return []
        return self._candidate_builder(query, list(results), source_scope)

    @staticmethod
    def _provider_name(provider: object) -> str:
        explicit_name = getattr(provider, "provider_name", None)
        if isinstance(explicit_name, str) and explicit_name:
            return explicit_name
        fallback_name = getattr(provider, "name", None)
        if isinstance(fallback_name, str) and fallback_name:
            return fallback_name
        normalized = provider.__class__.__name__.removesuffix("ProviderRepo").removesuffix("Repo")
        return normalized.replace("_", "-").lower() or "unknown"


class HybridSpecialRetriever:
    def __init__(
        self,
        *,
        lexical_retriever: Callable[[str, list[str]], list[RetrievedCandidate]],
        vector_retriever: MultiProviderBackedVectorRetriever,
    ) -> None:
        self._lexical_retriever = lexical_retriever
        self._vector_retriever = vector_retriever
        self.last_provider: str | None = None
        self.last_attempts: list[object] = []

    def prepare_for_policy(
        self,
        *,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference | None,
    ) -> None:
        self._vector_retriever.prepare_for_policy(
            access_policy=access_policy,
            execution_location_preference=execution_location_preference,
        )

    def __call__(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        lexical_candidates = self._lexical_retriever(query, source_scope)
        vector_candidates = self._vector_retriever(query, source_scope)
        self.last_provider = self._vector_retriever.last_provider
        self.last_attempts = list(self._vector_retriever.last_attempts)
        if not vector_candidates:
            return lexical_candidates
        merged: dict[str, RetrievedCandidate] = {}
        for candidate in [*lexical_candidates, *vector_candidates]:
            existing = merged.get(candidate.chunk_id)
            if existing is None:
                merged[candidate.chunk_id] = candidate
                continue
            merged_score = max(float(existing.score), float(candidate.score))
            if existing.source_kind != candidate.source_kind:
                merged_score += 0.04
            merged[candidate.chunk_id] = existing.__class__(
                **{
                    **existing.__dict__,
                    "score": merged_score,
                    "rank": min(existing.rank, candidate.rank),
                    "special_chunk_type": existing.special_chunk_type or candidate.special_chunk_type,
                    "metadata": dict((existing.metadata or {}) | (candidate.metadata or {})),
                }
            )
        ordered = sorted(merged.values(), key=lambda item: (-item.score, item.chunk_id))
        return [
            candidate.__class__(**{**candidate.__dict__, "rank": index})
            for index, candidate in enumerate(ordered, start=1)
        ]


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

    def section_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        query_terms = search_terms(query)
        candidates: list[RetrievedCandidate] = []
        for document in self._iter_documents(source_scope):
            source = self._metadata_repo.get_source(document.source_id)
            for chunk in self._metadata_repo.list_chunks(document.doc_id):
                segment = self._metadata_repo.get_segment(chunk.segment_id)
                if segment is None:
                    continue
                overlap = keyword_overlap(query_terms, " ".join(segment.toc_path))
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
                if special_type == "formula" and any(term in lowered for term in ("公式", "equation", "formula", "latex")):
                    score += 2
                if score <= 0:
                    continue
                candidates.extend(self._build_candidates_from_chunk_ids([chunk.chunk_id], source_kind="internal", scope=source_scope))
                if candidates:
                    latest = candidates[-1]
                    candidates[-1] = latest.__class__(**{**latest.__dict__, "score": float(score), "rank": 1})
        candidates.sort(key=lambda item: (-item.score, item.chunk_id))
        return candidates[:12]

    def metadata_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        understanding = self._query_understanding.analyze(query)
        page_numbers = set(self._as_str_list(understanding.metadata_filters.get("page_numbers")))
        source_types = set(self._as_str_list(understanding.metadata_filters.get("source_types")))
        preferred_sections = set(self._as_str_list(understanding.structure_constraints.get("preferred_section_terms")))
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
                candidates.extend(self._build_candidates_from_chunk_ids([chunk.chunk_id], source_kind="internal", scope=source_scope))
                if candidates:
                    latest = candidates[-1]
                    candidates[-1] = latest.__class__(**{**latest.__dict__, "score": score, "rank": 1})
        candidates.sort(key=lambda item: (-item.score, item.chunk_id))
        return candidates[:12]

    def graph_expander(self, query: str, source_scope: list[str], non_graph_evidence: list[CandidateLike]) -> list[RetrievedCandidate]:
        query_terms = set(search_terms(query))
        seed_chunk_ids = [candidate.chunk_id for candidate in non_graph_evidence if candidate.source_kind == "internal"]
        seed_node_scores: dict[str, float] = {}
        for candidate in non_graph_evidence:
            candidate_score = max(float(candidate.score), 0.1)
            for node in self._graph_repo.list_nodes_for_chunk(candidate.chunk_id):
                seed_node_scores[node.node_id] = max(seed_node_scores.get(node.node_id, 0.0), candidate_score)
            for edge in self._graph_repo.list_edges_for_chunk(candidate.chunk_id, include_candidates=True):
                for node_id in (edge.from_node_id, edge.to_node_id):
                    edge_score = candidate_score * max(edge.confidence, 0.45)
                    seed_node_scores[node_id] = max(seed_node_scores.get(node_id, 0.0), edge_score)
        chunk_scores, support_counts = self._score_graph_walk(
            seed_node_scores=seed_node_scores,
            query_terms=query_terms,
            max_depth=2,
        )
        for chunk_id in seed_chunk_ids:
            chunk_scores.pop(chunk_id, None)
            support_counts.pop(chunk_id, None)
        self._boost_related_special_chunks(chunk_scores, support_counts)
        return self._build_scored_candidates(
            chunk_scores,
            support_counts,
            source_scope=source_scope,
            source_kind="graph",
        )

    def web_retriever(self, query: str, source_scope: list[str]) -> list[RetrievedCandidate]:
        del source_scope
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
            for index, result in enumerate(DeterministicWebSearchRepo().search(query), start=1)
        ]

    def vector_retriever_from_repo(
        self,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingProviderBinding],
        *,
        default_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST,
    ) -> MultiProviderBackedVectorRetriever:
        return MultiProviderBackedVectorRetriever(factory=self, vector_repo=vector_repo, bindings=bindings, default_preference=default_preference)

    def local_retriever_from_repo(
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
            item_kind="entity",
            candidate_builder=self.build_local_candidates_from_vector_results,
            default_preference=default_preference,
        )

    def global_retriever_from_repo(
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
            item_kind="relation",
            candidate_builder=self.build_global_candidates_from_vector_results,
            default_preference=default_preference,
        )

    def special_retriever_from_repo(
        self,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingProviderBinding],
        *,
        default_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST,
    ) -> HybridSpecialRetriever:
        return HybridSpecialRetriever(
            lexical_retriever=self.special_retriever,
            vector_retriever=MultiProviderBackedVectorRetriever(
                factory=self,
                vector_repo=vector_repo,
                bindings=bindings,
                item_kind="multimodal",
                candidate_builder=self.build_multimodal_candidates_from_vector_results,
                default_preference=default_preference,
            ),
        )

    def build_chunk_candidates_from_vector_results(self, query: str, results: list[VectorSearchResult], source_scope: list[str]) -> list[RetrievedCandidate]:
        del query
        chunk_scores = {result.item_id: float(result.score) for result in results if float(result.score) > 0.0}
        support_counts = {result.item_id: 1 for result in results if float(result.score) > 0.0}
        self._boost_related_special_chunks(chunk_scores, support_counts)
        return self._build_scored_candidates(chunk_scores, support_counts, source_scope=source_scope)

    def build_local_candidates_from_vector_results(self, query: str, results: list[VectorSearchResult], source_scope: list[str]) -> list[RetrievedCandidate]:
        query_terms = set(search_terms(query))
        seed_node_scores: dict[str, float] = {}
        for result in results:
            base_score = max(float(result.score), 0.0)
            if base_score <= 0.0:
                continue
            seed_node_scores[result.item_id] = max(seed_node_scores.get(result.item_id, 0.0), base_score)
        chunk_scores, support_counts = self._score_graph_walk(
            seed_node_scores=seed_node_scores,
            query_terms=query_terms,
            max_depth=2,
        )
        self._boost_related_special_chunks(chunk_scores, support_counts)
        return self._build_scored_candidates(chunk_scores, support_counts, source_scope=source_scope)

    def build_global_candidates_from_vector_results(self, query: str, results: list[VectorSearchResult], source_scope: list[str]) -> list[RetrievedCandidate]:
        query_terms = set(search_terms(query))
        chunk_scores: dict[str, float] = defaultdict(float)
        support_counts: dict[str, int] = defaultdict(int)
        seed_node_scores: dict[str, float] = {}
        for result in results:
            edge = self._graph_repo.get_edge(result.item_id, include_candidates=True)
            if edge is None:
                continue
            base_score = max(float(result.score), 0.0)
            if base_score <= 0.0:
                continue
            edge_weight = self._edge_traversal_weight(edge.relation_type, query_terms=query_terms, confidence=edge.confidence, depth=0)
            self._add_scored_chunk_ids(chunk_scores, support_counts, edge.evidence_chunk_ids, score=base_score + edge_weight * 0.45)
            for node_id in (edge.from_node_id, edge.to_node_id):
                seed_node_scores[node_id] = max(seed_node_scores.get(node_id, 0.0), base_score * edge_weight * 0.92)
        walked_scores, walked_support = self._score_graph_walk(
            seed_node_scores=seed_node_scores,
            query_terms=query_terms,
            max_depth=1,
        )
        for chunk_id, score in walked_scores.items():
            chunk_scores[chunk_id] += score
            support_counts[chunk_id] += walked_support.get(chunk_id, 0)
        self._boost_related_special_chunks(chunk_scores, support_counts)
        return self._build_scored_candidates(chunk_scores, support_counts, source_scope=source_scope)

    def build_multimodal_candidates_from_vector_results(
        self,
        query: str,
        results: list[VectorSearchResult],
        source_scope: list[str],
    ) -> list[RetrievedCandidate]:
        query_terms = set(search_terms(query))
        chunk_scores: dict[str, float] = defaultdict(float)
        support_counts: dict[str, int] = defaultdict(int)
        for result in results:
            base_score = max(float(result.score), 0.0)
            if base_score <= 0.0:
                continue
            node = self._graph_repo.get_node(result.item_id)
            if node is None or node.node_type not in _MULTIMODAL_NODE_TYPES:
                continue
            type_bonus = 1.0
            for hint in _GRAPH_QUERY_HINTS.get(self._multimodal_relation_for_node_type(node.node_type), ()):
                if hint in query_terms or any(hint in term for term in query_terms):
                    type_bonus = 1.18
                    break
            self._add_scored_chunk_ids(
                chunk_scores,
                support_counts,
                self._graph_repo.list_node_evidence_chunk_ids(node.node_id),
                score=base_score * 1.24 * type_bonus,
            )
        return self._build_scored_candidates(chunk_scores, support_counts, source_scope=source_scope)

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

    @staticmethod
    def _add_scored_chunk_ids(chunk_scores: dict[str, float], support_counts: dict[str, int], chunk_ids: Sequence[str], *, score: float) -> None:
        if score <= 0.0:
            return
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score / (1.0 + (rank - 1) * 0.15)
            support_counts[chunk_id] = support_counts.get(chunk_id, 0) + 1

    def _boost_related_special_chunks(
        self,
        chunk_scores: dict[str, float],
        support_counts: dict[str, int],
    ) -> None:
        if not chunk_scores:
            return
        doc_chunk_cache: dict[str, list[object]] = {}
        scored_items = sorted(chunk_scores.items(), key=lambda item: (-item[1], item[0]))
        for chunk_id, base_score in scored_items:
            if base_score <= 0.0:
                continue
            seed_chunk = self._metadata_repo.get_chunk(chunk_id)
            if seed_chunk is None:
                continue
            doc_chunks = doc_chunk_cache.setdefault(seed_chunk.doc_id, self._metadata_repo.list_chunks(seed_chunk.doc_id))
            for candidate in doc_chunks:
                if candidate.chunk_role is not ChunkRole.SPECIAL or candidate.chunk_id == seed_chunk.chunk_id:
                    continue
                if not self._is_related_special_chunk(seed_chunk_id=seed_chunk.chunk_id, seed_chunk=seed_chunk, candidate_chunk=candidate):
                    continue
                boost = base_score * 0.28
                if candidate.special_chunk_type in {"table", "figure", "caption", "image_summary", "ocr_region", "formula"}:
                    boost *= 1.10
                chunk_scores[candidate.chunk_id] = max(chunk_scores.get(candidate.chunk_id, 0.0), boost)
                support_counts[candidate.chunk_id] = max(support_counts.get(candidate.chunk_id, 0), 1)

    def _score_graph_walk(
        self,
        *,
        seed_node_scores: dict[str, float],
        query_terms: set[str],
        max_depth: int,
    ) -> tuple[dict[str, float], dict[str, int]]:
        chunk_scores: dict[str, float] = defaultdict(float)
        support_counts: dict[str, int] = defaultdict(int)
        if not seed_node_scores:
            return chunk_scores, support_counts

        frontier: list[tuple[str, float, int]] = [
            (node_id, score, 0)
            for node_id, score in seed_node_scores.items()
            if score > 0.0
        ]
        best_node_scores = dict(seed_node_scores)

        while frontier:
            node_id, node_score, depth = frontier.pop(0)
            if node_score <= 0.0:
                continue
            current_node = self._graph_repo.get_node(node_id)
            current_node_type = None if current_node is None else current_node.node_type

            node_evidence_score = node_score * (1.92 if depth == 0 else 0.82 / (depth + 0.3))
            self._add_scored_chunk_ids(
                chunk_scores,
                support_counts,
                self._graph_repo.list_node_evidence_chunk_ids(node_id),
                score=node_evidence_score,
            )
            if depth >= max_depth:
                continue

            for edge in self._graph_repo.list_edges_for_node(node_id, include_candidates=True):
                if current_node_type in {"table", "figure", "caption", "ocr_region", "image_summary", "formula"} and edge.relation_type == "contains_special":
                    continue
                traversal_weight = self._edge_traversal_weight(
                    edge.relation_type,
                    query_terms=query_terms,
                    confidence=edge.confidence,
                    depth=depth,
                )
                if traversal_weight <= 0.0:
                    continue
                edge_score = node_score * traversal_weight
                self._add_scored_chunk_ids(
                    chunk_scores,
                    support_counts,
                    edge.evidence_chunk_ids,
                    score=edge_score * 0.72,
                )
                adjacent = edge.to_node_id if edge.from_node_id == node_id else edge.from_node_id
                next_score = edge_score * 0.76
                if next_score <= best_node_scores.get(adjacent, 0.0):
                    continue
                best_node_scores[adjacent] = next_score
                frontier.append((adjacent, next_score, depth + 1))
        return chunk_scores, support_counts

    @staticmethod
    def _edge_traversal_weight(
        relation_type: str,
        *,
        query_terms: set[str],
        confidence: float,
        depth: int,
    ) -> float:
        normalized_relation = relation_type.lower()
        relation_weight = _GRAPH_RELATION_WEIGHTS.get(normalized_relation, 0.9)
        hints = _GRAPH_QUERY_HINTS.get(normalized_relation, ())
        if hints and any(hint in query_terms or any(hint in term for term in query_terms) for hint in hints):
            relation_weight += 0.12
        depth_decay = 1.0 / (1.0 + depth * 0.55)
        confidence_factor = max(confidence, 0.45)
        return relation_weight * confidence_factor * depth_decay

    @staticmethod
    def _multimodal_relation_for_node_type(node_type: str) -> str:
        mapping = {
            "table": "tabulated_in",
            "figure": "illustrated_by",
            "caption": "captioned_by",
            "image_summary": "summarized_by",
            "ocr_region": "observed_in",
            "formula": "expressed_by_formula",
        }
        return mapping.get(node_type, "contains_special")

    @staticmethod
    def _is_related_special_chunk(*, seed_chunk_id: str, seed_chunk: object, candidate_chunk: object) -> bool:
        same_segment = getattr(seed_chunk, "segment_id", None) == getattr(candidate_chunk, "segment_id", None)
        same_parent = (
            getattr(seed_chunk, "parent_chunk_id", None) is not None
            and getattr(seed_chunk, "parent_chunk_id", None) == getattr(candidate_chunk, "parent_chunk_id", None)
        )
        linked_by_parent = getattr(candidate_chunk, "parent_chunk_id", None) == seed_chunk_id
        linked_by_chain = (
            getattr(candidate_chunk, "prev_chunk_id", None) == seed_chunk_id
            or getattr(candidate_chunk, "next_chunk_id", None) == seed_chunk_id
            or getattr(seed_chunk, "prev_chunk_id", None) == getattr(candidate_chunk, "chunk_id", None)
            or getattr(seed_chunk, "next_chunk_id", None) == getattr(candidate_chunk, "chunk_id", None)
        )
        return same_segment or same_parent or linked_by_parent or linked_by_chain

    def _build_scored_candidates(
        self,
        chunk_scores: dict[str, float],
        support_counts: dict[str, int],
        *,
        source_scope: list[str],
        source_kind: str = "internal",
        limit: int = 12,
    ) -> list[RetrievedCandidate]:
        ordered = sorted(chunk_scores.items(), key=lambda item: (-item[1], -support_counts.get(item[0], 0), item[0]))[:limit]
        if not ordered:
            return []
        candidates = self._build_candidates_from_chunk_ids(
            [chunk_id for chunk_id, _score in ordered],
            source_kind=source_kind,
            scope=source_scope,
        )
        return self._override_candidate_scores(candidates, {chunk_id: score for chunk_id, score in ordered})

    @staticmethod
    def _override_candidate_scores(candidates: list[RetrievedCandidate], score_map: dict[str, float]) -> list[RetrievedCandidate]:
        return [
            candidate if candidate.chunk_id not in score_map else candidate.__class__(**{**candidate.__dict__, "score": score_map[candidate.chunk_id], "rank": index})
            for index, candidate in enumerate(candidates, start=1)
        ]

    def _build_candidates_from_chunk_ids(self, chunk_ids: list[str], *, source_kind: str, scope: list[str] | None = None) -> list[RetrievedCandidate]:
        allowed = set(scope or [])
        candidates: list[RetrievedCandidate] = []
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
            parent_chunk = None if chunk.parent_chunk_id is None else self._metadata_repo.get_chunk(chunk.parent_chunk_id)
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
                    parent_text=None if parent_chunk is None else parent_chunk.text,
                    metadata=dict(chunk.metadata),
                    file_name=self._resolve_file_name(document.title, None if source is None else source.location),
                    page_start=None if segment is None or segment.page_range is None else segment.page_range[0],
                    page_end=None if segment is None or segment.page_range is None else segment.page_range[1],
                    chunk_type=chunk.special_chunk_type or chunk.chunk_role.value,
                    source_type=None if source is None else source.source_type.value,
                )
            )
        return candidates
