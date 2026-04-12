from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from rag.assembly import ChatCapabilityBinding
from rag.schema import EvidenceItem, RuntimeMode
from rag.schema.query import AnswerCitation, AnswerEvidenceLink, AnswerSection, GroundedAnswer, QueryUnderstanding
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference, ProviderAttempt
from rag.utils.text import keyword_overlap, looks_command_like, search_terms, split_sentences

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_GENERIC_QUERY_TERMS = {
    "这个",
    "那个",
    "这里",
    "那里",
    "什么",
    "哪些",
    "一下",
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
    "the",
    "a",
    "an",
}


class AnswerGenerationService:
    def __init__(self, *, min_overlap: int = 2) -> None:
        self._min_overlap = min_overlap

    def build_direct_prompt(
        self,
        *,
        query: str,
        response_type: str = "Multiple Paragraphs",
        user_prompt: str | None = None,
        conversation_history: Sequence[tuple[str, str]] = (),
    ) -> str:
        lines = [
            "你是知识库问答助手。",
            f"输出格式偏好：{response_type}",
            "如果你不确定，请直接说明不确定，不要伪造引用。",
        ]
        if user_prompt:
            lines.extend(["附加要求：", user_prompt.strip()])
        if conversation_history:
            lines.append("对话历史：")
            for role, content in conversation_history:
                normalized_role = role.strip() or "user"
                normalized_content = content.strip()
                if normalized_content:
                    lines.append(f"{normalized_role}: {normalized_content}")
        lines.extend(["当前问题：", query.strip()])
        return "\n".join(lines)

    def build_prompt(
        self,
        *,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        grounded_candidate: str,
        runtime_mode: RuntimeMode,
        response_type: str = "Multiple Paragraphs",
        user_prompt: str | None = None,
        conversation_history: Sequence[tuple[str, str]] = (),
        prompt_style: Literal["full", "compact", "minimal"] = "full",
    ) -> str:
        del runtime_mode
        if prompt_style == "minimal":
            lines = [
                f"Q:{query}",
                "JSON: answer_text, answer_sections, insufficient_evidence_flag; cite E ids.",
            ]
        elif prompt_style == "compact":
            lines = [
                grounded_candidate,
                f"问题：{query}",
                f"格式：{response_type}",
                "只基于证据回答；证据不足时将 insufficient_evidence_flag 设为 true。",
            ]
        else:
            lines = [
                grounded_candidate,
                "",
                "你是知识库回答生成器，只基于证据回答，不允许编造。",
                f"问题：{query}",
                f"输出格式偏好：{response_type}",
            ]
        if user_prompt:
            lines.extend(["附加要求：", user_prompt.strip()])
        if conversation_history:
            lines.append("对话历史：")
            history = conversation_history[-2:] if prompt_style != "full" else conversation_history
            for role, content in history:
                normalized_role = role.strip() or "user"
                normalized_content = content.strip()
                if normalized_content:
                    lines.append(f"{normalized_role}: {normalized_content}")
        if prompt_style == "minimal":
            lines.append("Evidence:")
        elif prompt_style == "compact":
            lines.extend(
                [
                    "输出要求：",
                    "返回一个 JSON 对象，包含 answer_text、answer_sections、insufficient_evidence_flag。",
                    "answer_sections 中的 evidence_ids 必须引用下面的 E 编号。",
                    "证据：",
                ]
            )
        else:
            lines.extend(
                [
                    "输出要求：",
                    "- 只输出一个 JSON 对象。",
                    '- 顶层字段必须包含 "answer_text"、"answer_sections"、"insufficient_evidence_flag"。',
                    '- answer_sections 是数组，每个元素包含 "title"、"text"、"evidence_ids"。',
                    "- evidence_ids 必须引用下面证据编号，例如 E1、E2。",
                    "- 证据不足时，把 insufficient_evidence_flag 设为 true，并明确说明无法从证据中确认。",
                    "- 严格使用和问题相同的语言。",
                    "- 不要输出 Markdown、代码块、解释文字。",
                    "证据：",
                ]
            )
        for index, item in enumerate(evidence_pack, start=1):
            evidence_id = self._evidence_id(index)
            section = " > ".join(item.section_path) if item.section_path else item.citation_anchor
            file_name = item.file_name or item.source_id or item.doc_id
            page_hint = (
                ""
                if item.page_start is None
                else (
                    f" | page={item.page_start}"
                    if item.page_end in {None, item.page_start}
                    else f" | pages={item.page_start}-{item.page_end}"
                )
            )
            chunk_type = (
                item.chunk_type
                or item.special_chunk_type
                or (item.chunk_role.value if item.chunk_role is not None else "child")
            )
            lines.extend(
                [
                    (
                        f"{evidence_id} {item.text}"
                        if prompt_style == "minimal"
                        else (
                            f"{evidence_id} | kind={item.evidence_kind} | file={file_name} "
                            f"| section={section}{page_hint} | chunk_type={chunk_type}"
                        )
                    ),
                ]
            )
            if prompt_style != "minimal":
                lines.append(item.text)
        return "\n".join(lines)

    def generate(
        self,
        *,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        runtime_mode: RuntimeMode,
        model_generate: Callable[[str], str] | None = None,
        grounded_candidate: str | None = None,
        enforce_grounding: bool = True,
        trust_evidence_pack: bool = False,
    ) -> GroundedAnswer:
        evidence = [item for item in evidence_pack if item.text.strip()]
        if self._evidence_is_insufficient(query, evidence, trust_evidence_pack=trust_evidence_pack):
            return self._insufficient_answer(query)

        seed = grounded_candidate or self._fallback_answer_text(evidence)
        if model_generate is None:
            return self._grounded_fallback(seed, evidence)

        prompt = self.build_prompt(
            query=query,
            evidence_pack=evidence,
            grounded_candidate=seed,
            runtime_mode=runtime_mode,
        )
        raw_output = str(model_generate(prompt))
        return self.answer_from_model_output(
            query=query,
            evidence_pack=evidence,
            grounded_candidate=seed,
            model_output=raw_output,
            enforce_grounding=enforce_grounding,
            trust_evidence_pack=trust_evidence_pack,
        )

    def answer_from_model_output(
        self,
        *,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        grounded_candidate: str,
        model_output: str,
        enforce_grounding: bool = True,
        trust_evidence_pack: bool = False,
    ) -> GroundedAnswer:
        evidence = [item for item in evidence_pack if item.text.strip()]
        if self._evidence_is_insufficient(query, evidence, trust_evidence_pack=trust_evidence_pack):
            return self._insufficient_answer(query)

        payload = self._parse_payload(model_output)
        answer_text = self._payload_answer_text(payload, model_output, grounded_candidate)
        sections = self._payload_sections(payload, answer_text)
        answer = self._materialize_answer(answer_text=answer_text, sections=sections, evidence_pack=evidence)
        insufficient_flag = bool(payload.get("insufficient_evidence_flag")) if payload is not None else False
        if insufficient_flag:
            return self._insufficient_answer(query)
        if answer.groundedness_flag or not enforce_grounding:
            return answer
        return self._grounded_fallback(grounded_candidate, evidence)

    @staticmethod
    def _evidence_id(index: int) -> str:
        return f"E{index}"

    def _evidence_is_insufficient(
        self,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        *,
        trust_evidence_pack: bool,
    ) -> bool:
        if not evidence_pack:
            return True
        if trust_evidence_pack:
            return False
        query_terms = _focus_terms(query)
        if not query_terms:
            return False
        combined = " ".join(self._evidence_search_text(item) for item in evidence_pack)
        if keyword_overlap(query_terms, combined) >= self._min_overlap:
            return False
        return max(keyword_overlap(query_terms, self._evidence_search_text(item)) for item in evidence_pack) < 1

    @staticmethod
    def _evidence_search_text(item: EvidenceItem) -> str:
        section = " ".join(item.section_path)
        return " ".join(part for part in (item.text, item.citation_anchor, section) if part)

    @staticmethod
    def _fallback_answer_text(evidence_pack: Sequence[EvidenceItem]) -> str:
        for item in evidence_pack:
            sentences = split_sentences(item.text)
            if sentences:
                return sentences[0]
        return evidence_pack[0].text if evidence_pack else "证据不足，无法回答。"

    @staticmethod
    def _parse_payload(model_output: str) -> dict[str, object] | None:
        stripped = model_output.strip()
        if not stripped:
            return None
        for candidate in (stripped, *[match.group(0) for match in _JSON_BLOCK_RE.finditer(stripped)]):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _payload_answer_text(
        payload: dict[str, object] | None,
        model_output: str,
        grounded_candidate: str,
    ) -> str:
        if payload is not None:
            raw_answer = payload.get("answer_text")
            if isinstance(raw_answer, str) and raw_answer.strip():
                return raw_answer.strip()
        stripped = model_output.strip()
        return stripped if stripped else grounded_candidate

    @staticmethod
    def _payload_sections(
        payload: dict[str, object] | None,
        answer_text: str,
    ) -> list[dict[str, object]]:
        if payload is None:
            return [{"title": "直接回答", "text": answer_text, "evidence_ids": []}]
        raw_sections = payload.get("answer_sections")
        if not isinstance(raw_sections, list):
            return [{"title": "直接回答", "text": answer_text, "evidence_ids": []}]
        sections: list[dict[str, object]] = []
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            text = item.get("text")
            evidence_ids = item.get("evidence_ids")
            if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
                continue
            resolved_ids = (
                [value for value in evidence_ids if isinstance(value, str)] if isinstance(evidence_ids, list) else []
            )
            sections.append({"title": title.strip(), "text": text.strip(), "evidence_ids": resolved_ids})
        return sections or [{"title": "直接回答", "text": answer_text, "evidence_ids": []}]

    def _materialize_answer(
        self,
        *,
        answer_text: str,
        sections: Sequence[dict[str, object]],
        evidence_pack: Sequence[EvidenceItem],
    ) -> GroundedAnswer:
        evidence_map = {self._evidence_id(index): item for index, item in enumerate(evidence_pack, start=1)}
        citations: list[AnswerCitation] = []
        evidence_links: list[AnswerEvidenceLink] = []
        answer_sections: list[AnswerSection] = []
        citation_by_chunk_id: dict[str, AnswerCitation] = {}
        grounded = True

        for section_index, raw_section in enumerate(sections, start=1):
            section_id = f"sec-{section_index}"
            section_text = str(raw_section.get("text", "")).strip()
            raw_evidence_ids = raw_section.get("evidence_ids", [])
            evidence_ids = (
                [value for value in raw_evidence_ids if isinstance(value, str)]
                if isinstance(raw_evidence_ids, list)
                else []
            )
            section_evidence = [
                evidence_map[evidence_id] for evidence_id in evidence_ids if evidence_id in evidence_map
            ]
            if not section_evidence:
                section_evidence = self._select_supporting_evidence(section_text, evidence_pack)

            citation_ids: list[str] = []
            evidence_chunk_ids: list[str] = []
            for item in section_evidence:
                citation = citation_by_chunk_id.get(item.chunk_id)
                if citation is None:
                    citation = AnswerCitation(
                        citation_id=f"cit-{len(citations) + 1}",
                        file_name=item.file_name,
                        section_path=list(item.section_path),
                        page_start=item.page_start,
                        page_end=item.page_end,
                        chunk_id=item.chunk_id,
                        chunk_type=item.chunk_type
                        or item.special_chunk_type
                        or (item.chunk_role.value if item.chunk_role is not None else "child"),
                        citation_anchor=item.citation_anchor,
                        doc_id=item.doc_id,
                        benchmark_doc_id=item.benchmark_doc_id,
                        source_id=item.source_id,
                        source_type=item.source_type,
                    )
                    citations.append(citation)
                    citation_by_chunk_id[item.chunk_id] = citation
                citation_ids.append(citation.citation_id)
                evidence_chunk_ids.append(item.chunk_id)
                evidence_links.append(
                    AnswerEvidenceLink(
                        link_id=f"link-{len(evidence_links) + 1}",
                        answer_section_id=section_id,
                        answer_excerpt=section_text,
                        evidence_chunk_id=item.chunk_id,
                        citation_id=citation.citation_id,
                        support_score=self._support_score(section_text, item.text),
                    )
                )

            grounded = grounded and self._section_grounded(section_text, section_evidence or evidence_pack)
            answer_sections.append(
                AnswerSection(
                    section_id=section_id,
                    title=str(raw_section.get("title", "直接回答")).strip() or "直接回答",
                    text=section_text,
                    citation_ids=citation_ids,
                    evidence_chunk_ids=evidence_chunk_ids,
                )
            )

        overall_grounded = grounded and (self._section_grounded(answer_text, evidence_pack) or len(answer_sections) > 1)
        return GroundedAnswer(
            answer_text=answer_text,
            answer_sections=answer_sections,
            citations=citations,
            evidence_links=evidence_links,
            groundedness_flag=overall_grounded,
            insufficient_evidence_flag=False,
        )

    def _grounded_fallback(self, answer_text: str, evidence_pack: Sequence[EvidenceItem]) -> GroundedAnswer:
        supporting = self._select_supporting_evidence(answer_text, evidence_pack)
        supporting_ids = [
            self._evidence_id(index) for index, item in enumerate(evidence_pack, start=1) if item in supporting
        ]
        return self._materialize_answer(
            answer_text=answer_text,
            sections=[{"title": "直接回答", "text": answer_text, "evidence_ids": supporting_ids}]
            if evidence_pack
            else [{"title": "直接回答", "text": answer_text, "evidence_ids": []}],
            evidence_pack=evidence_pack,
        )

    @staticmethod
    def _insufficient_answer(query: str) -> GroundedAnswer:
        del query
        return GroundedAnswer(
            answer_text="当前证据不足，无法给出可靠回答。",
            answer_sections=[
                AnswerSection(
                    section_id="sec-1",
                    title="证据不足",
                    text="当前证据不足，无法给出可靠回答。",
                    citation_ids=[],
                    evidence_chunk_ids=[],
                )
            ],
            citations=[],
            evidence_links=[],
            groundedness_flag=True,
            insufficient_evidence_flag=True,
        )

    def _select_supporting_evidence(
        self,
        section_text: str,
        evidence_pack: Sequence[EvidenceItem],
    ) -> list[EvidenceItem]:
        if not evidence_pack:
            return []
        ranked = sorted(
            evidence_pack,
            key=lambda item: (
                keyword_overlap(_focus_terms(section_text), self._evidence_search_text(item)),
                keyword_overlap(search_terms(section_text), item.text),
                float(item.score),
            ),
            reverse=True,
        )
        if not ranked:
            return []
        query_terms = _focus_terms(section_text)
        top = ranked[0]
        if keyword_overlap(query_terms, self._evidence_search_text(top)) == 0:
            return [top]
        return [item for item in ranked[:2] if keyword_overlap(query_terms, self._evidence_search_text(item)) > 0]

    @staticmethod
    def _support_score(answer_excerpt: str, evidence_text: str) -> float:
        terms = _focus_terms(answer_excerpt)
        if not terms:
            return 1.0 if answer_excerpt.strip() else 0.0
        overlap = keyword_overlap(terms, evidence_text)
        return min(1.0, max(0.0, overlap / max(1, len(terms))))

    def _section_grounded(self, text: str, evidence_pack: Sequence[EvidenceItem]) -> bool:
        variants = self._answer_variants(text)
        return any(self._variant_grounded(variant, evidence_pack) for variant in variants if variant)

    def _variant_grounded(self, text: str, evidence_pack: Sequence[EvidenceItem]) -> bool:
        if self._text_supported_by_evidence(text, evidence_pack):
            return True
        terms = _focus_terms(text)
        if not terms:
            return False
        for item in evidence_pack:
            chunk_type = item.chunk_type or item.special_chunk_type or ""
            if chunk_type not in {"table", "image_summary", "ocr_region", "figure", "caption"}:
                continue
            if keyword_overlap(terms, self._evidence_search_text(item)) >= 1:
                return True
        return False

    @staticmethod
    def _answer_variants(text: str) -> tuple[str, ...]:
        normalized = text.strip()
        if not normalized:
            return ()
        variants = [normalized]
        if ":" in normalized:
            suffix = normalized.split(":", 1)[1].strip()
            if suffix:
                variants.append(suffix)
        if "：" in normalized:
            suffix = normalized.split("：", 1)[1].strip()
            if suffix:
                variants.append(suffix)
        return tuple(dict.fromkeys(variants))

    @staticmethod
    def _normalize_supported_text(text: str) -> str:
        normalized = text.replace("**", " ").replace("__", " ").replace("`", " ")
        normalized = re.sub(r"^\s*[-*]\s*", "", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"^\s*\d+\.\s*", "", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @classmethod
    def _text_supported_by_evidence(cls, answer_text: str, evidence_pack: Sequence[EvidenceItem]) -> bool:
        normalized_answer = cls._normalize_supported_text(answer_text)
        if not normalized_answer:
            return False
        answer_sentences = split_sentences(normalized_answer) or (normalized_answer,)
        normalized_evidence = [cls._normalize_supported_text(item.text) for item in evidence_pack if item.text.strip()]
        if not normalized_evidence:
            return False

        def supported(sentence: str) -> bool:
            sentence_terms = search_terms(sentence)
            required_overlap = max(2, (len(sentence_terms) + 1) // 2) if sentence_terms else 0
            if any(
                sentence in evidence_text
                or (required_overlap > 0 and keyword_overlap(sentence_terms, evidence_text) >= required_overlap)
                for evidence_text in normalized_evidence
            ):
                return True
            if required_overlap == 0:
                return False
            return keyword_overlap(sentence_terms, " ".join(normalized_evidence)) >= required_overlap

        return all(supported(sentence) for sentence in answer_sentences if sentence)

def _focus_terms(text: str) -> tuple[str, ...]:
    filtered = tuple(term for term in search_terms(text) if term not in _GENERIC_QUERY_TERMS)
    return filtered or search_terms(text)

_GENERIC_ANSWER_TERMS = {
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
    def _dedupe_providers(bindings: Sequence[ChatCapabilityBinding]) -> list[ChatCapabilityBinding]:
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
            else _answer_focus_terms(query)
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


def _answer_focus_terms(text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for term in search_terms(text):
        normalized = term.strip().lower()
        if not normalized or normalized in _GENERIC_ANSWER_TERMS or len(normalized) < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered or list(search_terms(text))


__all__ = ["AnswerGenerationResult", "AnswerGenerationService", "AnswerGenerator"]
