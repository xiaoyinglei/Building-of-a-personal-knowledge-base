from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rag.llm.assembly import ChatCapabilityBinding
from rag.llm.generation import AnswerGenerationService
from rag.schema._types.access import AccessPolicy, ExecutionLocationPreference, RuntimeMode
from rag.schema._types.diagnostics import ProviderAttempt
from rag.schema._types.envelope import EvidenceItem
from rag.schema._types.generation import AnswerSection, GroundedAnswer
from rag.schema._types.query import QueryUnderstanding
from rag.schema._types.text import keyword_overlap, looks_command_like, search_terms, split_sentences

_GENERIC_TERMS = {
    "什么",
    "哪些",
    "哪里",
    "一下",
    "这个",
    "那个",
    "这里",
    "那里",
    "请问",
    "请",
    "how",
    "what",
    "which",
    "where",
    "why",
    "when",
    "this",
    "that",
    "these",
    "those",
    "document",
    "doc",
    "source",
}


@dataclass(frozen=True, slots=True)
class AnswerGenerationResult:
    answer: GroundedAnswer
    provider: str | None
    model: str | None
    attempts: list[ProviderAttempt]


@dataclass(slots=True)
class AnswerGenerator:
    answer_generation_service: AnswerGenerationService = field(default_factory=AnswerGenerationService)
    chat_bindings: tuple[ChatCapabilityBinding, ...] = ()

    def grounded_candidate(
        self,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        *,
        query_understanding: QueryUnderstanding | None = None,
    ) -> str:
        hits = [item for item in evidence_pack if item.text.strip()]
        if not hits:
            return "Insufficient evidence in indexed sources."

        if query_understanding is not None and query_understanding.needs_special:
            special_candidate = self._special_aware_conclusion(hits, query_understanding)
            if special_candidate is not None:
                return special_candidate

        if query_understanding is not None and query_understanding.needs_structure:
            structure_candidate = self._structure_aware_conclusion(hits, query_understanding)
            if structure_candidate is not None:
                return structure_candidate

        return self._best_overlap_sentence(query, hits, query_understanding)

    def generate(
        self,
        *,
        query: str,
        prompt: str,
        evidence_pack: Sequence[EvidenceItem],
        grounded_candidate: str,
        runtime_mode: RuntimeMode,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference,
    ) -> AnswerGenerationResult:
        attempts: list[ProviderAttempt] = []
        for binding in self._ordered_bindings(access_policy, execution_location_preference):
            attempt = ProviderAttempt(
                stage="generation",
                capability="chat",
                provider=binding.provider_name,
                location=binding.location,
                model=binding.model_name,
                status="success",
            )
            try:
                output = binding.chat(prompt)
                answer = self.answer_generation_service.answer_from_model_output(
                    query=query,
                    evidence_pack=evidence_pack,
                    grounded_candidate=grounded_candidate,
                    model_output=output,
                    enforce_grounding=True,
                    trust_evidence_pack=True,
                )
                attempts.append(attempt)
                return AnswerGenerationResult(
                    answer=answer,
                    provider=attempt.provider,
                    model=attempt.model,
                    attempts=attempts,
                )
            except Exception as exc:
                attempts.append(attempt.model_copy(update={"status": "failed", "error": str(exc)}))

        fallback = self.answer_generation_service.generate(
            query=query,
            evidence_pack=evidence_pack,
            runtime_mode=runtime_mode,
            grounded_candidate=grounded_candidate,
            trust_evidence_pack=True,
        )
        return AnswerGenerationResult(
            answer=fallback,
            provider=None,
            model=None,
            attempts=attempts,
        )

    def generate_direct(
        self,
        *,
        query: str,
        prompt: str,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference,
    ) -> AnswerGenerationResult:
        attempts: list[ProviderAttempt] = []
        for binding in self._ordered_bindings(access_policy, execution_location_preference):
            attempt = ProviderAttempt(
                stage="generation",
                capability="chat",
                provider=binding.provider_name,
                location=binding.location,
                model=binding.model_name,
                status="success",
            )
            try:
                output = binding.chat(prompt).strip()
            except Exception as exc:
                attempts.append(attempt.model_copy(update={"status": "failed", "error": str(exc)}))
                continue
            answer_text = output or "模型没有返回内容。"
            answer = GroundedAnswer(
                answer_text=answer_text,
                answer_sections=[
                    AnswerSection(
                        section_id="direct-response",
                        title="Direct Response",
                        text=answer_text,
                        citation_ids=[],
                        evidence_chunk_ids=[],
                    )
                ],
                citations=[],
                evidence_links=[],
                groundedness_flag=False,
                insufficient_evidence_flag=False,
            )
            attempts.append(attempt)
            return AnswerGenerationResult(
                answer=answer,
                provider=attempt.provider,
                model=attempt.model,
                attempts=attempts,
            )

        fallback_answer = GroundedAnswer(
            answer_text="No chat-capable provider available for bypass mode.",
            answer_sections=[
                AnswerSection(
                    section_id="direct-response",
                    title="Direct Response",
                    text="No chat-capable provider available for bypass mode.",
                    citation_ids=[],
                    evidence_chunk_ids=[],
                )
            ],
            citations=[],
            evidence_links=[],
            groundedness_flag=False,
            insufficient_evidence_flag=True,
        )
        return AnswerGenerationResult(
            answer=fallback_answer,
            provider=None,
            model=None,
            attempts=attempts,
        )

    def _ordered_bindings(
        self,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference,
    ) -> list[ChatCapabilityBinding]:
        if not self.chat_bindings:
            return []
        preferred_locations: tuple[str, ...]
        if access_policy.local_only or execution_location_preference is ExecutionLocationPreference.LOCAL_ONLY:
            preferred_locations = ("local",)
        elif execution_location_preference is ExecutionLocationPreference.LOCAL_FIRST:
            preferred_locations = ("local", "cloud")
        else:
            preferred_locations = ("cloud", "local")

        ordered: list[ChatCapabilityBinding] = []
        remaining = list(self.chat_bindings)
        for location in preferred_locations:
            matched = [binding for binding in remaining if binding.location == location]
            ordered.extend(self._dedupe_providers(matched))
            remaining = [binding for binding in remaining if binding.location != location]
        ordered.extend(self._dedupe_providers(remaining))
        return ordered

    @staticmethod
    def _dedupe_providers(
        bindings: Sequence[ChatCapabilityBinding],
    ) -> list[ChatCapabilityBinding]:
        seen: set[int] = set()
        ordered: list[ChatCapabilityBinding] = []
        for binding in bindings:
            identity = id(binding.backend)
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(binding)
        return ordered

    @staticmethod
    def _special_aware_conclusion(
        hits: Sequence[EvidenceItem],
        understanding: QueryUnderstanding,
    ) -> str | None:
        preferred_targets = set(understanding.special_targets)
        ranked_hits = sorted(
            hits[:8],
            key=lambda item: (
                int((item.chunk_type or item.special_chunk_type or "") in preferred_targets),
                float(item.score),
            ),
            reverse=True,
        )
        for item in ranked_hits:
            chunk_type = item.chunk_type or item.special_chunk_type or ""
            if preferred_targets and chunk_type not in preferred_targets:
                continue
            if item.text.strip():
                return item.text.strip()
        return None

    @staticmethod
    def _structure_aware_conclusion(
        hits: Sequence[EvidenceItem],
        understanding: QueryUnderstanding,
    ) -> str | None:
        query_focus_terms = understanding.preferred_section_terms or understanding.quoted_terms
        ranked_hits = sorted(
            hits[:8],
            key=lambda item: (
                int(keyword_overlap(query_focus_terms, item.citation_anchor) > 0),
                keyword_overlap(query_focus_terms, item.citation_anchor),
                keyword_overlap(query_focus_terms, item.text),
                float(item.score),
            ),
            reverse=True,
        )
        for hit in ranked_hits:
            lead = AnswerGenerator._pick_structure_lead(hit.text, query_focus_terms)
            if lead:
                return lead
        return None

    @staticmethod
    def _pick_structure_lead(text: str, query_focus_terms: Sequence[str]) -> str | None:
        sentences = split_sentences(text)
        for sentence in sentences:
            if keyword_overlap(query_focus_terms, sentence) > 0 and not looks_command_like(sentence):
                return AnswerGenerator._normalize_answer_fragment(sentence)
        for sentence in sentences:
            if not looks_command_like(sentence):
                return AnswerGenerator._normalize_answer_fragment(sentence)
        return None

    @staticmethod
    def _normalize_answer_fragment(text: str) -> str:
        cleaned = " ".join(text.replace("`", "").split())
        if cleaned.endswith(("。", "！", "？", ".", "!", "?")):
            return cleaned[:-1].strip()
        return cleaned.strip()

    @staticmethod
    def _best_overlap_sentence(
        query: str,
        hits: Sequence[EvidenceItem],
        understanding: QueryUnderstanding | None,
    ) -> str:
        query_terms = search_terms(query)
        query_focus_terms = (
            list(understanding.preferred_section_terms)
            if understanding is not None and understanding.preferred_section_terms
            else _focus_terms(query)
        )
        normalized_query = query.strip().lower()
        sentences = [sentence for item in hits[:6] for sentence in split_sentences(item.text)]
        if not sentences:
            return hits[0].text

        def _score(sentence: str) -> tuple[int, int, int, int, float]:
            lowered = sentence.lower()
            exact_match = int(bool(normalized_query) and normalized_query in lowered)
            focus_overlap = keyword_overlap(query_focus_terms, sentence)
            term_overlap = keyword_overlap(query_terms, sentence)
            command_penalty = 0 if looks_command_like(sentence) else 1
            structure_priority = (
                keyword_overlap(query_focus_terms, sentence)
                if understanding is not None and understanding.needs_structure
                else 0
            )
            return (
                exact_match,
                structure_priority,
                focus_overlap,
                command_penalty,
                float(term_overlap),
            )

        candidate_pool = [sentence for sentence in sentences if not looks_command_like(sentence)] or sentences
        return max(candidate_pool, key=_score)


def _focus_terms(text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for term in search_terms(text):
        normalized = term.strip().lower()
        if not normalized or normalized in _GENERIC_TERMS or len(normalized) < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered or list(search_terms(text))
