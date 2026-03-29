from __future__ import annotations

import re

from pkp.types.query import QueryUnderstanding
from pkp.types.text import focus_terms


class QueryUnderstandingService:
    _PAGE_PATTERN = re.compile(r"(?:第\s*(\d+)\s*页|page\s*(\d+))", re.IGNORECASE)
    _SPECIAL_SIGNAL_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("table", ("表格", "table", "指标", "数值", "统计表")),
        ("figure", ("图片", "图", "figure", "截图", "流程图")),
        ("ocr_region", ("ocr", "识别", "截图文字", "图中文字", "区域文字")),
        ("image_summary", ("图片总结", "图像摘要", "画面内容", "visual summary", "image summary")),
        ("caption", ("图注", "图题", "caption")),
        ("formula", ("公式", "equation", "formula", "latex", "数学表达式")),
    )
    _STRUCTURE_TERMS: tuple[str, ...] = (
        "架构",
        "结构",
        "模块",
        "章节",
        "标题",
        "目录",
        "分层",
        "第几层",
        "哪几层",
        "section",
        "heading",
        "module",
        "architecture",
    )
    _SOURCE_TYPE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("pdf", ("pdf", "pdf文档", "扫描件")),
        ("markdown", ("markdown", "md", "markdown文档")),
        ("docx", ("docx", "word", "word文档", "文档")),
        ("image", ("图片", "图像", "image", "截图")),
    )

    def analyze(self, query: str) -> QueryUnderstanding:
        normalized = query.strip()
        lowered = normalized.lower()
        special_targets = [
            target
            for target, keywords in self._SPECIAL_SIGNAL_MAP
            if any(keyword in lowered for keyword in keywords)
        ]
        structure_hit = any(term in normalized or term in lowered for term in self._STRUCTURE_TERMS)
        preferred_sections = self._preferred_section_terms(normalized)
        page_numbers = self._page_numbers(normalized)
        source_types = [
            source_type
            for source_type, keywords in self._SOURCE_TYPE_TERMS
            if any(keyword in lowered for keyword in keywords)
        ]
        metadata_filters: dict[str, list[str] | str | bool] = {}
        if page_numbers:
            metadata_filters["page_numbers"] = page_numbers
        if source_types:
            metadata_filters["source_types"] = source_types
        if special_targets:
            metadata_filters["special_targets"] = special_targets
        if preferred_sections:
            metadata_filters["preferred_section_terms"] = preferred_sections
        needs_metadata = bool(page_numbers or source_types)

        confidence = 0.35
        if special_targets:
            confidence += 0.25
        if structure_hit or preferred_sections:
            confidence += 0.2
        if needs_metadata:
            confidence += 0.15
        if len(normalized) >= 8:
            confidence += 0.1
        confidence = min(confidence, 0.95)

        if needs_metadata:
            intent = "localized_lookup"
            query_type = "page_lookup" if page_numbers else "metadata_lookup"
        elif special_targets:
            intent = "special_lookup"
            query_type = special_targets[0]
        elif structure_hit or preferred_sections:
            intent = "structure_lookup"
            query_type = "structure"
        else:
            intent = "semantic_lookup"
            query_type = "general"

        return QueryUnderstanding(
            intent=intent,
            query_type=query_type,
            needs_dense=True,
            needs_sparse=True,
            needs_special=bool(special_targets),
            needs_structure=structure_hit or bool(preferred_sections),
            needs_metadata=needs_metadata,
            structure_constraints={
                "preferred_section_terms": preferred_sections,
                "prefer_heading_match": structure_hit or bool(preferred_sections),
            },
            metadata_filters=metadata_filters,
            special_targets=special_targets,
            confidence=confidence,
        )

    @staticmethod
    def _preferred_section_terms(query: str) -> list[str]:
        exact_phrases = [
            phrase
            for phrase in ("系统架构", "技术架构", "组织架构", "系统设计", "工作总结", "项目计划")
            if phrase in query
        ]
        candidates = [
            term
            for term in focus_terms(query)
            if len(term) >= 2 and term not in {"什么", "哪些", "多少", "如何", "为什么"}
        ]
        section_terms: list[str] = list(exact_phrases)
        for candidate in candidates:
            if any(candidate != phrase and candidate in phrase for phrase in exact_phrases):
                continue
            if candidate.endswith(("架构", "结构", "模块", "章节", "流程", "总结", "工作")):
                section_terms.append(candidate)
        seen: set[str] = set()
        ordered: list[str] = []
        for term in section_terms:
            if term in seen:
                continue
            seen.add(term)
            ordered.append(term)
        return ordered[:3]

    @classmethod
    def _page_numbers(cls, query: str) -> list[str]:
        numbers: list[str] = []
        for direct, english in cls._PAGE_PATTERN.findall(query):
            value = direct or english
            if value and value not in numbers:
                numbers.append(value)
        return numbers
