from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256

from pkp.types.access import RuntimeMode
from pkp.types.artifact import ArtifactStatus, ArtifactType, KnowledgeArtifact
from pkp.types.envelope import EvidenceItem, PreservationSuggestion


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
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", f"{title} {query}"):
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
        return re.sub(r"\s+", " ", query.strip().lower())

    def _render_topic_or_comparison_body(
        self,
        *,
        query: str,
        title: str,
        artifact_type: ArtifactType,
        evidence: Sequence[EvidenceItem],
        differences_or_conflicts: Sequence[str],
        reviewed_at: datetime,
        confidence: float,
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
            "## Confidence and Coverage",
            f"- Confidence: {confidence:.2f}",
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
        confidence: float,
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
            "## Confidence and Coverage",
            f"- Confidence: {confidence:.2f}",
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
        confidence: float,
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
            "## Confidence and Coverage",
            f"- Confidence: {confidence:.2f}",
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
        confidence: float,
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
            "- What date, order, or milestone is still missing from this timeline?",
            "",
            "## Last Reviewed",
            f"- {reviewed_at.isoformat()}",
            "",
            "## Confidence and Coverage",
            f"- Confidence: {confidence:.2f}",
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
        confidence: float,
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
            "## Confidence and Coverage",
            f"- Confidence: {confidence:.2f}",
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
        confidence: float,
    ) -> str:
        if artifact_type is ArtifactType.DOCUMENT_SUMMARY:
            return self._render_document_summary_body(
                query=query,
                title=title,
                evidence=evidence,
                reviewed_at=reviewed_at,
                confidence=confidence,
            )
        if artifact_type is ArtifactType.SECTION_SUMMARY:
            return self._render_section_summary_body(
                query=query,
                title=title,
                evidence=evidence,
                reviewed_at=reviewed_at,
                confidence=confidence,
            )
        if artifact_type is ArtifactType.TIMELINE:
            return self._render_timeline_body(
                query=query,
                title=title,
                evidence=evidence,
                reviewed_at=reviewed_at,
                confidence=confidence,
            )
        if artifact_type is ArtifactType.OPEN_QUESTION_PAGE:
            return self._render_open_question_body(
                query=query,
                title=title,
                evidence=evidence,
                reviewed_at=reviewed_at,
                confidence=confidence,
            )
        return self._render_topic_or_comparison_body(
            query=query,
            title=title,
            artifact_type=artifact_type,
            evidence=evidence,
            differences_or_conflicts=differences_or_conflicts,
            reviewed_at=reviewed_at,
            confidence=confidence,
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
        confidence = 0.8 if differences_or_conflicts else 0.9
        return KnowledgeArtifact(
            artifact_id=self._artifact_id(query, evidence),
            artifact_type=artifact_type,
            title=title,
            supported_chunk_ids=[item.chunk_id for item in evidence],
            confidence=confidence,
            status=ArtifactStatus.SUGGESTED,
            last_reviewed_at=last_reviewed_at,
            body_markdown=self._render_body(
                query=query,
                title=title,
                artifact_type=artifact_type,
                evidence=evidence,
                differences_or_conflicts=differences_or_conflicts,
                reviewed_at=last_reviewed_at,
                confidence=confidence,
            ),
            source_scope=sorted({item.doc_id for item in evidence}),
            metadata={
                "coverage_documents": str(len(self._unique_docs(evidence))),
                "coverage_evidence_items": str(len(evidence)),
                "reviewed_at": last_reviewed_at.isoformat(),
            },
        )
