from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.retrieval.analysis import section_family_aliases
from rag.schema.core import ChunkRole
from rag.schema.query import (
    ArtifactStatus,
    ArtifactType,
    ComplexityLevel,
    EvidenceItem,
    KnowledgeArtifact,
    PreservationSuggestion,
    QueryUnderstanding,
    TaskType,
)
from rag.schema.runtime import AccessPolicy, RuntimeMode
from rag.utils.text import search_terms

if TYPE_CHECKING:
    from rag.retrieval.models import RetrievalResult


class CandidateLike(Protocol):
    chunk_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str
    source_id: str | None
    section_path: Sequence[str]
    chunk_role: ChunkRole | None
    special_chunk_type: str | None
    parent_chunk_id: str | None


def classify_retrieval_family(
    *,
    evidence_kind: str,
    special_chunk_type: str | None,
    retrieval_channels: Sequence[str] = (),
) -> str:
    channels = {channel for channel in retrieval_channels if channel}
    if evidence_kind == "external":
        return "external"
    if special_chunk_type is not None or channels & {"special", "section", "metadata"}:
        return "multimodal"
    if evidence_kind == "graph" or channels & {"local", "global"}:
        return "kg"
    if channels & {"vector", "full_text"}:
        return "vector"
    return "kg" if evidence_kind == "graph" else "vector"


class EvidenceThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    fast_min_evidence_chunks: int = 2
    fast_min_sections: int = 1
    deep_min_evidence_chunks: int = 4
    deep_min_supporting_units: int = 2


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    internal: list[EvidenceItem] = Field(default_factory=list)
    external: list[EvidenceItem] = Field(default_factory=list)
    graph: list[EvidenceItem] = Field(default_factory=list)

    @property
    def all(self) -> list[EvidenceItem]:
        return [*self.internal, *self.external, *self.graph]


class SelfCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieve_more: bool
    evidence_sufficient: bool
    claim_supported: bool


class EvidenceService:
    def __init__(self, thresholds: EvidenceThresholds | None = None) -> None:
        self._thresholds = thresholds or EvidenceThresholds()

    @staticmethod
    def _candidate_source_scope(candidate: CandidateLike) -> set[str]:
        scope = {candidate.doc_id}
        if candidate.source_id:
            scope.add(candidate.source_id)
        return scope

    @staticmethod
    def _candidate_access_policy(candidate: object) -> AccessPolicy | None:
        policy = getattr(candidate, "effective_access_policy", None)
        if isinstance(policy, AccessPolicy):
            return policy
        return None

    @staticmethod
    def _is_candidate_external(candidate: object) -> bool:
        return getattr(candidate, "source_kind", "internal") == "external"

    @staticmethod
    def _is_candidate_graph(candidate: object) -> bool:
        return getattr(candidate, "source_kind", "internal") == "graph"

    def filter_candidates(
        self,
        candidates: Sequence[CandidateLike],
        *,
        source_scope: Sequence[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        query_understanding: QueryUnderstanding | None = None,
    ) -> list[CandidateLike]:
        allowed_scope = set(source_scope)
        filtered: list[CandidateLike] = []
        for candidate in candidates:
            if allowed_scope and not self._candidate_source_scope(candidate) & allowed_scope:
                continue
            if self._is_candidate_external(candidate) and access_policy.external_retrieval.value != "allow":
                continue
            candidate_policy = self._candidate_access_policy(candidate)
            if candidate_policy is not None:
                if runtime_mode not in candidate_policy.allowed_runtimes:
                    continue
                if not (candidate_policy.allowed_locations & access_policy.allowed_locations):
                    continue
            if query_understanding is not None and not self._matches_explicit_constraints(
                candidate, query_understanding
            ):
                continue
            filtered.append(candidate)
        return filtered

    @staticmethod
    def _matches_explicit_constraints(candidate: CandidateLike, query_understanding: QueryUnderstanding) -> bool:
        if getattr(candidate, "source_kind", "internal") == "external":
            return True
        if not query_understanding.has_explicit_constraints():
            return True
        if not EvidenceService._matches_metadata_constraints(candidate, query_understanding):
            return False
        if not EvidenceService._matches_structure_constraints(candidate, query_understanding):
            return False
        return True

    @staticmethod
    def _matches_metadata_constraints(candidate: CandidateLike, query_understanding: QueryUnderstanding) -> bool:
        metadata_filters = query_understanding.metadata_filters
        candidate_metadata = getattr(candidate, "metadata", {}) or {}
        candidate_source_type = getattr(candidate, "source_type", None) or candidate_metadata.get("source_type")
        if metadata_filters.source_types and candidate_source_type is not None:
            if candidate_source_type not in metadata_filters.source_types:
                return False

        candidate_file_name = getattr(candidate, "file_name", None)
        if metadata_filters.file_names and candidate_file_name is not None:
            if candidate_file_name not in metadata_filters.file_names:
                return False
        if metadata_filters.document_titles and candidate_file_name is not None:
            if candidate_file_name not in metadata_filters.document_titles:
                return False

        if not metadata_filters.page_numbers and not metadata_filters.page_ranges:
            return True
        candidate_pages = EvidenceService._candidate_pages(candidate)
        if not candidate_pages:
            return True
        if metadata_filters.page_numbers and not candidate_pages & set(metadata_filters.page_numbers):
            if not metadata_filters.page_ranges:
                return False
        if metadata_filters.page_ranges and not any(
            any(page_range.start <= page <= page_range.end for page in candidate_pages)
            for page_range in metadata_filters.page_ranges
        ):
            if not metadata_filters.page_numbers:
                return False
            if not candidate_pages & set(metadata_filters.page_numbers):
                return False
        return True

    @staticmethod
    def _matches_structure_constraints(candidate: CandidateLike, query_understanding: QueryUnderstanding) -> bool:
        constraints = query_understanding.structure_constraints
        if not constraints.requires_structure_match:
            return True
        section_path = tuple(getattr(candidate, "section_path", ()) or ())
        if not section_path:
            return True
        section_text = " ".join(section_path).lower()
        preferred_terms = {term.lower() for term in query_understanding.preferred_section_terms}
        heading_hints = {hint.lower() for hint in constraints.heading_hints}
        title_hints = {hint.lower() for hint in constraints.title_hints}
        semantic_aliases = {
            alias.lower()
            for family in constraints.semantic_section_families
            for alias in section_family_aliases(family)
        }
        match_terms = preferred_terms | heading_hints | title_hints | semantic_aliases
        if not match_terms:
            return True
        matched = any(term in section_text for term in match_terms)
        if constraints.prefer_heading_match:
            return matched
        return matched

    @staticmethod
    def _candidate_pages(candidate: CandidateLike) -> set[int]:
        metadata = getattr(candidate, "metadata", {}) or {}
        pages: set[int] = set()
        page_no = metadata.get("page_no")
        if isinstance(page_no, str) and page_no.isdigit():
            pages.add(int(page_no))
        page_start = getattr(candidate, "page_start", None)
        page_end = getattr(candidate, "page_end", None)
        if isinstance(page_start, int) and isinstance(page_end, int):
            pages.update(range(page_start, page_end + 1))
        elif isinstance(page_start, int):
            pages.add(page_start)
        elif isinstance(page_end, int):
            pages.add(page_end)
        return pages

    @staticmethod
    def _to_evidence_item(candidate: CandidateLike) -> EvidenceItem:
        evidence_kind = getattr(candidate, "source_kind", "internal")
        if evidence_kind not in {"internal", "external", "graph"}:
            evidence_kind = "internal"
        retrieval_channels = list(getattr(candidate, "retrieval_channels", []) or [])
        retrieval_family = classify_retrieval_family(
            evidence_kind=evidence_kind,
            special_chunk_type=getattr(candidate, "special_chunk_type", None),
            retrieval_channels=retrieval_channels,
        )
        return EvidenceItem(
            chunk_id=candidate.chunk_id,
            doc_id=candidate.doc_id,
            source_id=getattr(candidate, "source_id", None),
            citation_anchor=candidate.citation_anchor,
            text=candidate.text,
            score=float(candidate.score),
            evidence_kind=evidence_kind,
            chunk_role=getattr(candidate, "chunk_role", None),
            special_chunk_type=getattr(candidate, "special_chunk_type", None),
            parent_chunk_id=getattr(candidate, "parent_chunk_id", None),
            file_name=getattr(candidate, "file_name", None),
            section_path=list(getattr(candidate, "section_path", ()) or ()),
            page_start=getattr(candidate, "page_start", None),
            page_end=getattr(candidate, "page_end", None),
            chunk_type=getattr(candidate, "chunk_type", None),
            source_type=getattr(candidate, "source_type", None),
            retrieval_channels=retrieval_channels,
            retrieval_family=retrieval_family,
        )

    def assemble_bundle(self, candidates: Sequence[CandidateLike]) -> EvidenceBundle:
        internal: list[EvidenceItem] = []
        external: list[EvidenceItem] = []
        graph: list[EvidenceItem] = []
        for candidate in candidates:
            item = self._to_evidence_item(candidate)
            if item.evidence_kind == "external":
                external.append(item)
            elif item.evidence_kind == "graph":
                graph.append(item)
            else:
                internal.append(item)
        return EvidenceBundle(internal=internal, external=external, graph=graph)

    def evaluate_self_check(
        self,
        *,
        bundle: EvidenceBundle,
        task_type: TaskType,
        complexity_level: ComplexityLevel,
    ) -> SelfCheckResult:
        internal = bundle.internal
        section_keys = {item.citation_anchor if item.citation_anchor else item.chunk_id for item in internal}
        doc_ids = {item.doc_id for item in internal}

        if task_type in {TaskType.LOOKUP, TaskType.SINGLE_DOC_QA} or complexity_level in {
            ComplexityLevel.L1_DIRECT,
            ComplexityLevel.L2_SCOPED,
        }:
            evidence_sufficient = (
                len(internal) >= self._thresholds.fast_min_evidence_chunks
                and len(section_keys) >= self._thresholds.fast_min_sections
            )
        else:
            evidence_sufficient = len(internal) >= self._thresholds.deep_min_evidence_chunks and (
                len(doc_ids) >= self._thresholds.deep_min_supporting_units
                or len(section_keys) >= self._thresholds.deep_min_supporting_units
            )

        claim_supported = evidence_sufficient and bool(internal)
        retrieve_more = not evidence_sufficient
        return SelfCheckResult(
            retrieve_more=retrieve_more,
            evidence_sufficient=evidence_sufficient,
            claim_supported=claim_supported,
        )

    @staticmethod
    def evidence_counts(bundle: EvidenceBundle) -> Counter[str]:
        return Counter(item.evidence_kind for item in bundle.all)


class ContextEvidenceMerger:
    def merge(self, retrieval: RetrievalResult) -> list[EvidenceItem]:
        internal_by_id = {item.chunk_id: item for item in retrieval.evidence.internal}
        ordered_internal = [
            internal_by_id[chunk_id] for chunk_id in retrieval.reranked_chunk_ids if chunk_id in internal_by_id
        ]
        seen_internal = {item.chunk_id for item in ordered_internal}
        ordered_internal.extend(item for item in retrieval.evidence.internal if item.chunk_id not in seen_internal)

        merged: list[EvidenceItem] = []
        merged_by_chunk_id: dict[str, EvidenceItem] = {}
        ordered_chunk_ids: list[str] = []

        for item in [*ordered_internal, *retrieval.evidence.graph]:
            existing = merged_by_chunk_id.get(item.chunk_id)
            if existing is None:
                merged_by_chunk_id[item.chunk_id] = item
                ordered_chunk_ids.append(item.chunk_id)
                continue
            merged_by_chunk_id[item.chunk_id] = self._merge_duplicate_item(existing, item)

        merged.extend(merged_by_chunk_id[chunk_id] for chunk_id in ordered_chunk_ids)

        seen_external: set[str] = set()
        for item in retrieval.evidence.external:
            if item.chunk_id in seen_external:
                continue
            seen_external.add(item.chunk_id)
            merged.append(item)
        return merged

    @staticmethod
    def _merge_duplicate_item(existing: EvidenceItem, incoming: EvidenceItem) -> EvidenceItem:
        preferred = existing
        secondary = incoming
        if existing.evidence_kind != "internal" and incoming.evidence_kind == "internal":
            preferred = incoming
            secondary = existing

        merged_kind = (
            "internal" if "internal" in {existing.evidence_kind, incoming.evidence_kind} else preferred.evidence_kind
        )
        merged_text = preferred.text if len(preferred.text) >= len(secondary.text) else secondary.text
        merged_channels = list(dict.fromkeys([*existing.retrieval_channels, *incoming.retrieval_channels]))
        merged_family = classify_retrieval_family(
            evidence_kind=merged_kind,
            special_chunk_type=preferred.special_chunk_type or secondary.special_chunk_type,
            retrieval_channels=merged_channels,
        )

        return preferred.model_copy(
            update={
                "evidence_kind": merged_kind,
                "score": max(float(existing.score), float(incoming.score)),
                "text": merged_text,
                "section_path": preferred.section_path or secondary.section_path,
                "file_name": preferred.file_name or secondary.file_name,
                "source_id": preferred.source_id or secondary.source_id,
                "chunk_type": preferred.chunk_type or secondary.chunk_type,
                "source_type": preferred.source_type or secondary.source_type,
                "special_chunk_type": preferred.special_chunk_type or secondary.special_chunk_type,
                "parent_chunk_id": preferred.parent_chunk_id or secondary.parent_chunk_id,
                "page_start": preferred.page_start if preferred.page_start is not None else secondary.page_start,
                "page_end": preferred.page_end if preferred.page_end is not None else secondary.page_end,
                "retrieval_channels": merged_channels,
                "retrieval_family": merged_family,
            }
        )


class ArtifactService:
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "compare",
        "comparison",
        "the",
        "this",
        "that",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "with",
        "from",
        "into",
        "over",
        "path",
        "page",
        "topic",
    }

    @staticmethod
    def _unique_docs(evidence: Sequence[EvidenceItem]) -> set[str]:
        return {item.doc_id for item in evidence}

    @staticmethod
    def _artifact_id(query: str, evidence: Sequence[EvidenceItem]) -> str:
        artifact_seed = query + "|" + "|".join(item.chunk_id for item in evidence)
        return f"artifact-{sha256(artifact_seed.encode()).hexdigest()[:12]}"

    @staticmethod
    def _definition_heading(artifact_type: ArtifactType) -> str:
        return {
            ArtifactType.COMPARISON_PAGE: "Comparison Definition",
            ArtifactType.TIMELINE: "Timeline Definition",
            ArtifactType.DOCUMENT_SUMMARY: "Document Definition",
            ArtifactType.SECTION_SUMMARY: "Section Definition",
            ArtifactType.OPEN_QUESTION_PAGE: "Open Question Definition",
        }.get(artifact_type, "Topic Definition")

    def _related_concepts(self, title: str, query: str) -> list[str]:
        concepts: list[str] = []
        seen: set[str] = set()
        for token in search_terms(f"{title} {query}"):
            normalized = token.lower()
            if normalized in self._STOPWORDS or len(normalized) < 4:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            concepts.append(token)
        return concepts

    @staticmethod
    def _key_conclusions(
        evidence: Sequence[EvidenceItem],
        differences_or_conflicts: Sequence[str],
    ) -> list[str]:
        conclusions: list[str] = []
        seen: set[str] = set()
        for value in [*differences_or_conflicts, *(item.text for item in evidence)]:
            normalized = " ".join(value.split()).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            conclusions.append(value)
            if len(conclusions) == 3:
                break
        return conclusions

    @staticmethod
    def _bullet_lines(values: Sequence[str], empty_message: str) -> list[str]:
        return [f"- {value}" for value in values] or [f"- {empty_message}"]

    @staticmethod
    def _evidence_lines(evidence: Sequence[EvidenceItem]) -> list[str]:
        return [f"- {item.doc_id} | {item.citation_anchor}: {item.text}" for item in evidence] or [
            "- No supporting evidence captured."
        ]

    @staticmethod
    def _normalized_query(query: str) -> str:
        return " ".join(query.strip().lower().split())

    def _render_topic_or_comparison_body(
        self,
        *,
        query: str,
        title: str,
        artifact_type: ArtifactType,
        evidence: Sequence[EvidenceItem],
        differences_or_conflicts: Sequence[str],
        reviewed_at: datetime,
    ) -> str:
        doc_ids = sorted(self._unique_docs(evidence))
        related_concepts = self._related_concepts(title, query)
        key_conclusions = self._key_conclusions(evidence, differences_or_conflicts)
        conclusion_lines = self._bullet_lines(key_conclusions, "No conclusions extracted.")
        evidence_lines = self._evidence_lines(evidence)
        boundaries = [f"- Coverage is limited to the cited evidence from {len(doc_ids)} document(s)."]
        if differences_or_conflicts:
            boundaries.append("- Conflicting evidence means downstream answers should preserve ambiguity.")
        else:
            boundaries.append("- Failure cases are not exhaustively enumerated in the current evidence set.")
        disagreements = self._bullet_lines(
            differences_or_conflicts,
            "No material disagreements identified in current evidence.",
        )
        related_documents = self._bullet_lines(doc_ids, "No related documents captured.")
        related_entities = self._bullet_lines(related_concepts, "Not extracted from the current evidence.")
        sections = [
            f"# {title}",
            "",
            f"## {self._definition_heading(artifact_type)}",
            f"- Query focus: {query}",
            "",
            "## Key Conclusions",
            *conclusion_lines,
            "",
            "## Key Evidence",
            *evidence_lines,
            "",
            "## Boundaries and Failure Cases",
            *boundaries,
            "",
            "## Disagreements",
            *disagreements,
            "",
            "## Related Documents",
            *related_documents,
            "",
            "## Related Concepts and Entities",
            *related_entities,
            "",
            "## Open Questions",
            "- What additional evidence would strengthen, refine, or falsify this page?",
            "",
            "## Last Reviewed",
            f"- {reviewed_at.isoformat()}",
            "",
            "## Evidence Coverage",
            f"- Coverage: {len(evidence)} evidence item(s) across {len(doc_ids)} document(s).",
        ]
        return "\n".join(sections)

    def _render_document_summary_body(
        self,
        *,
        query: str,
        title: str,
        evidence: Sequence[EvidenceItem],
        reviewed_at: datetime,
    ) -> str:
        doc_ids = sorted(self._unique_docs(evidence))
        summary_lines = self._bullet_lines(
            self._key_conclusions(evidence, []),
            "No summary extracted.",
        )
        sections = [
            f"# {title}",
            "",
            f"## {self._definition_heading(ArtifactType.DOCUMENT_SUMMARY)}",
            f"- Query focus: {query}",
            f"- Document scope: {', '.join(doc_ids) if doc_ids else 'No related documents captured.'}",
            "",
            "## Summary",
            *summary_lines,
            "",
            "## Key Evidence",
            *self._evidence_lines(evidence),
            "",
            "## Open Questions",
            "- What parts of this document still need validation or follow-up evidence?",
            "",
            "## Last Reviewed",
            f"- {reviewed_at.isoformat()}",
            "",
            "## Evidence Coverage",
            f"- Coverage: {len(evidence)} evidence item(s) across {len(doc_ids)} document(s).",
        ]
        return "\n".join(sections)

    def _render_section_summary_body(
        self,
        *,
        query: str,
        title: str,
        evidence: Sequence[EvidenceItem],
        reviewed_at: datetime,
    ) -> str:
        doc_ids = sorted(self._unique_docs(evidence))
        summary_lines = self._bullet_lines(
            self._key_conclusions(evidence, []),
            "No section summary extracted.",
        )
        sections = [
            f"# {title}",
            "",
            f"## {self._definition_heading(ArtifactType.SECTION_SUMMARY)}",
            f"- Query focus: {query}",
            f"- Section scope: {', '.join(doc_ids) if doc_ids else 'No related documents captured.'}",
            "",
            "## Section Summary",
            *summary_lines,
            "",
            "## Key Evidence",
            *self._evidence_lines(evidence),
            "",
            "## Open Questions",
            "- What section-level detail still needs corroboration?",
            "",
            "## Last Reviewed",
            f"- {reviewed_at.isoformat()}",
            "",
            "## Evidence Coverage",
            f"- Coverage: {len(evidence)} evidence item(s) across {len(doc_ids)} document(s).",
        ]
        return "\n".join(sections)

    def _render_timeline_body(
        self,
        *,
        query: str,
        title: str,
        evidence: Sequence[EvidenceItem],
        reviewed_at: datetime,
    ) -> str:
        doc_ids = sorted(self._unique_docs(evidence))
        event_lines = [
            f"- {index + 1}. {item.doc_id} | {item.citation_anchor}: {item.text}" for index, item in enumerate(evidence)
        ] or ["- No temporal events captured."]
        sections = [
            f"# {title}",
            "",
            f"## {self._definition_heading(ArtifactType.TIMELINE)}",
            f"- Query focus: {query}",
            f"- Timeline scope: {', '.join(doc_ids) if doc_ids else 'No related documents captured.'}",
            "",
            "## Timeline",
            *event_lines,
            "",
            "## Key Evidence",
            *self._evidence_lines(evidence),
            "",
            "## Open Questions",
            "- Which temporal gaps remain unresolved?",
            "",
            "## Last Reviewed",
            f"- {reviewed_at.isoformat()}",
            "",
            "## Evidence Coverage",
            f"- Coverage: {len(evidence)} evidence item(s) across {len(doc_ids)} document(s).",
        ]
        return "\n".join(sections)

    def _render_open_question_body(
        self,
        *,
        query: str,
        title: str,
        evidence: Sequence[EvidenceItem],
        reviewed_at: datetime,
    ) -> str:
        doc_ids = sorted(self._unique_docs(evidence))
        question_lines = self._bullet_lines(
            [query.strip().rstrip("?")] if query.strip() else [],
            "No open questions captured.",
        )
        candidate_answers = self._bullet_lines(
            self._key_conclusions(evidence, []),
            "No candidate answers extracted.",
        )
        sections = [
            f"# {title}",
            "",
            f"## {self._definition_heading(ArtifactType.OPEN_QUESTION_PAGE)}",
            f"- Query focus: {query}",
            f"- Evidence scope: {', '.join(doc_ids) if doc_ids else 'No related documents captured.'}",
            "",
            "## Questions",
            *question_lines,
            "",
            "## Candidate Answers",
            *candidate_answers,
            "",
            "## Key Evidence",
            *self._evidence_lines(evidence),
            "",
            "## Open Questions",
            "- Which unresolved point should be investigated next?",
            "",
            "## Last Reviewed",
            f"- {reviewed_at.isoformat()}",
            "",
            "## Evidence Coverage",
            f"- Coverage: {len(evidence)} evidence item(s) across {len(doc_ids)} document(s).",
        ]
        return "\n".join(sections)

    def _render_body(
        self,
        *,
        query: str,
        title: str,
        artifact_type: ArtifactType,
        evidence: Sequence[EvidenceItem],
        differences_or_conflicts: Sequence[str],
        reviewed_at: datetime,
    ) -> str:
        if artifact_type is ArtifactType.DOCUMENT_SUMMARY:
            return self._render_document_summary_body(
                query=query,
                title=title,
                evidence=evidence,
                reviewed_at=reviewed_at,
            )
        if artifact_type is ArtifactType.SECTION_SUMMARY:
            return self._render_section_summary_body(
                query=query,
                title=title,
                evidence=evidence,
                reviewed_at=reviewed_at,
            )
        if artifact_type is ArtifactType.TIMELINE:
            return self._render_timeline_body(
                query=query,
                title=title,
                evidence=evidence,
                reviewed_at=reviewed_at,
            )
        if artifact_type is ArtifactType.OPEN_QUESTION_PAGE:
            return self._render_open_question_body(
                query=query,
                title=title,
                evidence=evidence,
                reviewed_at=reviewed_at,
            )
        return self._render_topic_or_comparison_body(
            query=query,
            title=title,
            artifact_type=artifact_type,
            evidence=evidence,
            differences_or_conflicts=differences_or_conflicts,
            reviewed_at=reviewed_at,
        )

    def suggest_preservation(
        self,
        *,
        query: str,
        runtime_mode: RuntimeMode,
        evidence: Sequence[EvidenceItem],
        differences_or_conflicts: Sequence[str] | None = None,
    ) -> PreservationSuggestion:
        doc_count = len(self._unique_docs(evidence))
        conflict_count = len(differences_or_conflicts or ())
        normalized = self._normalized_query(query)
        is_timeline_query = any(token in normalized for token in ("timeline", "trend", "chronology", "over time"))
        is_open_question_query = any(
            token in normalized for token in ("open question", "open questions", "unknown", "unresolved", "gap")
        )
        is_section_summary_query = "section" in normalized and any(
            token in normalized for token in ("summarize", "summary", "recap", "overview")
        )
        is_document_summary_query = "document" in normalized and any(
            token in normalized for token in ("summarize", "summary", "recap", "overview")
        )
        reusable = runtime_mode is RuntimeMode.DEEP and (
            doc_count >= 2
            or conflict_count > 0
            or len(evidence) >= 4
            or is_timeline_query
            or is_open_question_query
            or is_section_summary_query
            or is_document_summary_query
        )
        if not reusable:
            return PreservationSuggestion(suggested=False)

        if "compare" in normalized or conflict_count > 0:
            artifact_type = ArtifactType.COMPARISON_PAGE.value
        elif is_timeline_query:
            artifact_type = ArtifactType.TIMELINE.value
        elif is_open_question_query:
            artifact_type = ArtifactType.OPEN_QUESTION_PAGE.value
        elif is_section_summary_query:
            artifact_type = ArtifactType.SECTION_SUMMARY.value
        elif is_document_summary_query:
            artifact_type = ArtifactType.DOCUMENT_SUMMARY.value
        else:
            artifact_type = ArtifactType.TOPIC_PAGE.value

        title = query.strip().rstrip("?") or "Reusable knowledge"
        rationale = "Evidence spans multiple documents and is likely reusable."
        if conflict_count > 0:
            rationale = "Evidence captures a stable comparison or conflict map."
        elif artifact_type == ArtifactType.TIMELINE.value:
            rationale = "Evidence is organized around a temporal sequence."
        elif artifact_type == ArtifactType.OPEN_QUESTION_PAGE.value:
            rationale = "Evidence captures unresolved questions worth preserving."
        elif artifact_type in {
            ArtifactType.DOCUMENT_SUMMARY.value,
            ArtifactType.SECTION_SUMMARY.value,
        }:
            rationale = "Evidence is concentrated enough to preserve a reusable summary."

        return PreservationSuggestion(
            suggested=True,
            artifact_type=artifact_type,
            title=title,
            rationale=rationale,
        )

    def apply_lifecycle(
        self,
        *,
        proposed: KnowledgeArtifact,
        existing_artifacts: Sequence[KnowledgeArtifact],
    ) -> list[KnowledgeArtifact]:
        updated_existing: list[KnowledgeArtifact] = []
        proposed_chunks = set(proposed.supported_chunk_ids)
        for artifact in existing_artifacts:
            status = artifact.status
            if artifact.artifact_type is proposed.artifact_type and artifact.title == proposed.title:
                existing_chunks = set(artifact.supported_chunk_ids)
                overlap = bool(existing_chunks & proposed_chunks)
                if overlap and artifact.body_markdown != proposed.body_markdown:
                    status = ArtifactStatus.CONFLICTED
                elif proposed_chunks > existing_chunks:
                    status = ArtifactStatus.STALE
            updated_existing.append(artifact.model_copy(update={"status": status}))

        return [*updated_existing, proposed]

    @staticmethod
    def build_timestamp() -> datetime:
        return datetime.now(UTC)

    def build_artifact(
        self,
        *,
        query: str,
        suggestion: PreservationSuggestion,
        evidence: Sequence[EvidenceItem],
        differences_or_conflicts: Sequence[str],
        reviewed_at: datetime | None = None,
    ) -> KnowledgeArtifact:
        title = suggestion.title or query.strip().rstrip("?") or "Reusable knowledge"
        artifact_type = ArtifactType(suggestion.artifact_type or ArtifactType.TOPIC_PAGE.value)
        last_reviewed_at = reviewed_at or self.build_timestamp()
        return KnowledgeArtifact(
            artifact_id=self._artifact_id(query, evidence),
            artifact_type=artifact_type,
            title=title,
            supported_chunk_ids=[item.chunk_id for item in evidence],
            confidence=None,
            status=ArtifactStatus.SUGGESTED,
            last_reviewed_at=last_reviewed_at,
            body_markdown=self._render_body(
                query=query,
                title=title,
                artifact_type=artifact_type,
                evidence=evidence,
                differences_or_conflicts=differences_or_conflicts,
                reviewed_at=last_reviewed_at,
            ),
            source_scope=sorted({item.doc_id for item in evidence}),
            metadata={
                "coverage_documents": str(len(self._unique_docs(evidence))),
                "coverage_evidence_items": str(len(evidence)),
                "reviewed_at": last_reviewed_at.isoformat(),
            },
        )

__all__ = [
    "ArtifactService",
    "CandidateLike",
    "ContextEvidenceMerger",
    "EvidenceBundle",
    "EvidenceService",
    "EvidenceThresholds",
    "SelfCheckResult",
    "classify_retrieval_family",
]
