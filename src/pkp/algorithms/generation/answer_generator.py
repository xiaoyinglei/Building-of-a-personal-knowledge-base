from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from pkp.llm.generation import AnswerGenerationService
from pkp.repo.interfaces import EmbeddingProviderBinding
from pkp.types.access import AccessPolicy, ExecutionLocation, ExecutionLocationPreference, RuntimeMode
from pkp.types.diagnostics import ProviderAttempt
from pkp.types.envelope import EvidenceItem
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


@dataclass(frozen=True, slots=True)
class AnswerGenerationResult:
    answer: GroundedAnswer
    provider: str | None
    model: str | None
    attempts: list[ProviderAttempt]


@dataclass(slots=True)
class AnswerGenerator:
    answer_generation_service: AnswerGenerationService = field(default_factory=AnswerGenerationService)
    provider_bindings: tuple[EmbeddingProviderBinding, ...] = ()

    def grounded_candidate(self, query: str, evidence_pack: Sequence[EvidenceItem]) -> str:
        hits = [item for item in evidence_pack if item.text.strip()]
        if not hits:
            return "Insufficient evidence in indexed sources."

        operation_conclusion = self._operation_aware_conclusion(query, hits)
        if operation_conclusion is not None:
            return operation_conclusion

        structure_conclusion = self._structure_aware_conclusion(query, hits)
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
            provider = binding.provider
            chat = getattr(provider, "chat", None)
            if not callable(chat):
                continue
            attempt = ProviderAttempt(
                stage="generation",
                capability="chat",
                provider=self._provider_name(provider),
                location=binding.location,
                model=self._provider_model(provider, "chat"),
                status="success",
            )
            try:
                output = str(chat(prompt))
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

    def _ordered_bindings(
        self,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference,
    ) -> list[EmbeddingProviderBinding]:
        if not self.provider_bindings:
            return []
        if access_policy.local_only or execution_location_preference is ExecutionLocationPreference.LOCAL_ONLY:
            preferred_locations = ("local",)
        elif execution_location_preference is ExecutionLocationPreference.LOCAL_FIRST:
            preferred_locations = ("local", "cloud")
        else:
            preferred_locations = ("cloud", "local")

        ordered: list[EmbeddingProviderBinding] = []
        remaining = list(self.provider_bindings)
        for location in preferred_locations:
            matched = [binding for binding in remaining if binding.location == location]
            ordered.extend(self._dedupe_providers(matched))
            remaining = [binding for binding in remaining if binding.location != location]
        ordered.extend(self._dedupe_providers(remaining))
        return ordered

    @staticmethod
    def _dedupe_providers(
        bindings: Sequence[EmbeddingProviderBinding],
    ) -> list[EmbeddingProviderBinding]:
        seen: set[int] = set()
        ordered: list[EmbeddingProviderBinding] = []
        for binding in bindings:
            identity = id(binding.provider)
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(binding)
        return ordered

    @staticmethod
    def _provider_name(provider: object) -> str:
        explicit_name = getattr(provider, "provider_name", None)
        if isinstance(explicit_name, str) and explicit_name:
            return explicit_name
        fallback_name = getattr(provider, "name", None)
        if isinstance(fallback_name, str) and fallback_name:
            return fallback_name
        class_name = provider.__class__.__name__
        normalized = class_name.removesuffix("ProviderRepo").removesuffix("Repo")
        return normalized.replace("_", "-").lower() or "unknown"

    @staticmethod
    def _provider_model(provider: object, capability: str) -> str | None:
        attribute_names = {
            "chat": ("chat_model_name", "_chat_model", "_model"),
            "embed": ("embedding_model_name", "_embedding_model"),
        }.get(capability, ())
        for attribute_name in attribute_names:
            value = getattr(provider, attribute_name, None)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _operation_aware_conclusion(query: str, hits: Sequence[EvidenceItem]) -> str | None:
        if not looks_operation_query(query):
            return None

        query_focus_terms = focus_terms(query)
        segments: list[str] = []
        seen: set[str] = set()
        for hit in hits[:12]:
            if AnswerGenerator._operation_fragment_score(hit.text, query_focus_terms) < 2:
                continue
            normalized = AnswerGenerator._normalize_operation_fragment(hit.text)
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
    def _structure_aware_conclusion(query: str, hits: Sequence[EvidenceItem]) -> str | None:
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

        if AnswerGenerator._prefers_layer_signature_query(query):
            for hit in scored_hits:
                signature = AnswerGenerator._extract_layer_signature(hit.text)
                if signature is not None:
                    return signature

        for hit in scored_hits:
            lead = AnswerGenerator._pick_structure_lead(hit.text, query_focus_terms)
            if lead:
                break

        for hit in scored_hits:
            if keyword_overlap(query_focus_terms, hit.text) == 0 and keyword_overlap(
                query_focus_terms, hit.citation_anchor
            ) == 0:
                continue
            bullets = AnswerGenerator._extract_structure_points(hit.text)
            hit_lead = AnswerGenerator._pick_structure_lead(hit.text, query_focus_terms)
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
        signature = AnswerGenerator._extract_layer_signature(text)
        if signature is not None:
            return signature
        sentences = split_sentences(text)
        for sentence in sentences:
            if (
                keyword_overlap(query_focus_terms, sentence) > 0
                and looks_structure_text(sentence)
                and not looks_command_like(sentence)
            ):
                return AnswerGenerator._normalize_answer_fragment(sentence)
        for sentence in sentences:
            if looks_structure_text(sentence) and not looks_command_like(sentence):
                return AnswerGenerator._normalize_answer_fragment(sentence)
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
            segments.append(AnswerGenerator._normalize_answer_fragment(cleaned))
        return segments

    @staticmethod
    def _normalize_answer_fragment(text: str) -> str:
        cleaned = " ".join(text.replace("`", "").split())
        if cleaned.endswith(("。", "！", "？", ".", "!", "?")):
            return cleaned[:-1].strip()
        return cleaned.strip()

    @staticmethod
    def _normalize_operation_fragment(text: str) -> str:
        normalized = AnswerGenerator._normalize_answer_fragment(text)
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
            r"([A-Za-z][A-Za-z0-9/ ]+(?:\s*->\s*[A-Za-z][A-Za-z0-9/ ]+){2,})",
            normalized,
        )
        if match is None:
            return None
        return AnswerGenerator._normalize_answer_fragment(match.group(1))
