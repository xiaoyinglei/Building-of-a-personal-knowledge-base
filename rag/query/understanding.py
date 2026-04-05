from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.schema._types.query import (
    ConfidenceBand,
    MetadataFilters,
    PageRangeConstraint,
    QueryIntent,
    QueryUnderstanding,
    RoutingHints,
    StructureConstraints,
)
from rag.schema._types.text import focus_terms, search_terms, split_sentences, text_unit_count

_NORMALIZE_SPACE_RE = re.compile(r"\s+")
_QUOTED_TERM_RE = re.compile(r"""["“”'‘’]([^"“”'‘’]{2,64})["“”'‘’]""")
_PAGE_RANGE_RE = re.compile(
    r"(?:第\s*(?P<cjk_start>\d+)\s*(?:页)?\s*(?:到|至|-|—|~)\s*(?P<cjk_end>\d+)\s*页)"
    r"|(?:pages?\s*(?P<en_start>\d+)\s*(?:to|-|—|~)\s*(?P<en_end>\d+))",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(
    r"(?:第\s*(?P<cjk>\d+)\s*页)"
    r"|(?:pages?\s*(?P<en>\d+))"
    r"|(?:p\.\s*(?P<short>\d+))",
    re.IGNORECASE,
)
_TOPIC_PATTERN_RE = re.compile(
    r"(?:关于|有关|围绕|针对|讲|说|介绍|compare|summary of|summarize|about)\s*"
    r"(?P<topic>[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff _./-]{2,48})",
    re.IGNORECASE,
)
_DOC_TITLE_RE = re.compile(
    r'(?:文档|文件|报告|pdf|docx|pptx|xlsx|slides?|deck|sheet)[^"“”]{0,8}["“](?P<title>[^"“”]{2,64})["”]',
    re.IGNORECASE,
)
_DOC_TITLE_RE_REVERSED = re.compile(
    r'["“](?P<title>[^"“”]{2,64})["”](?:文档|文件|报告|pdf|docx|pptx|xlsx|slides?|deck|sheet)',
    re.IGNORECASE,
)

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
_VAGUE_REFERENCES = {
    "这个",
    "那个",
    "这里",
    "那里",
    "it",
    "this",
    "that",
    "these",
    "those",
}

_COMPARISON_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "tradeoff",
    "trade-off",
    "difference",
    "contrast",
    "对比",
    "比较",
    "区别",
    "差异",
    "优劣",
)
_SUMMARY_MARKERS = (
    "summary",
    "summarize",
    "summarise",
    "overview",
    "key points",
    "takeaway",
    "总结",
    "概括",
    "梳理",
    "综述",
    "摘要",
    "要点",
)
_PROCESS_MARKERS = (
    "flow",
    "process",
    "pipeline",
    "workflow",
    "steps",
    "step by step",
    "how does",
    "how do",
    "流程",
    "过程",
    "链路",
    "步骤",
    "工作流",
    "怎么做",
    "如何进行",
)
_FACTUAL_MARKERS = (
    "what is",
    "what does",
    "who",
    "where",
    "when",
    "which",
    "是多少",
    "是什么",
    "做什么",
    "谁",
    "哪里",
    "哪一个",
    "哪个",
    "几层",
    "几步",
)
_SEMANTIC_MARKERS = (
    "why",
    "how",
    "meaning",
    "semantics",
    "implication",
    "explain",
    "interpret",
    "原理",
    "含义",
    "原因",
    "为什么",
    "怎么理解",
    "解释",
)
_STRUCTURE_LOCATORS = (
    "section",
    "chapter",
    "heading",
    "title",
    "part",
    "appendix",
    "章节",
    "小节",
    "标题",
    "部分",
    "目录",
    "一节",
    "哪一章",
    "哪一节",
    "哪部分",
    "哪一部分",
)
_LOCATION_MARKERS = (
    "where in",
    "located",
    "which section",
    "which chapter",
    "哪一页",
    "第几页",
    "在哪",
    "位于",
    "哪部分",
    "哪一章",
    "哪一节",
)
_GRAPH_MARKERS = (
    "dependency",
    "depend",
    "relationship",
    "relate",
    "link",
    "reason",
    "cause",
    "impact",
    "depends on",
    "依赖",
    "关系",
    "原因",
    "影响",
    "链路",
    "流程",
)


@dataclass(frozen=True, slots=True)
class ConceptSpec:
    canonical: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SectionFamilySpec:
    family: str
    canonical_terms: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SectionSemanticMatch:
    family: str
    canonical_term: str
    matched_terms: tuple[str, ...]
    score: float


@dataclass(frozen=True, slots=True)
class NormalizedQuery:
    raw_text: str
    text: str
    lowered: str
    tokens: tuple[str, ...]
    focus_terms: tuple[str, ...]
    clauses: tuple[str, ...]
    quoted_terms: tuple[str, ...]
    unit_count: int


@dataclass(frozen=True, slots=True)
class QueryFeatureSet:
    page_numbers: tuple[int, ...]
    page_ranges: tuple[PageRangeConstraint, ...]
    source_types: tuple[str, ...]
    special_targets: tuple[str, ...]
    title_hints: tuple[str, ...]
    heading_hints: tuple[str, ...]
    topical_terms: tuple[str, ...]
    structure_locators: tuple[str, ...]
    location_markers: tuple[str, ...]
    comparison_markers: tuple[str, ...]
    summary_markers: tuple[str, ...]
    process_markers: tuple[str, ...]
    factual_markers: tuple[str, ...]
    semantic_markers: tuple[str, ...]
    graph_markers: tuple[str, ...]
    document_titles: tuple[str, ...]
    file_names: tuple[str, ...]
    vague_reference: bool


@dataclass(frozen=True, slots=True)
class SemanticSignals:
    intent_scores: dict[QueryIntent, float]
    section_matches: tuple[SectionSemanticMatch, ...]
    structural_signal: float
    metadata_signal: float
    special_signal: float


@dataclass(frozen=True, slots=True)
class IntentDecision:
    intent: QueryIntent
    query_type: str


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    confidence: float
    band: ConfidenceBand


class SectionSemanticMatcher(Protocol):
    def match(self, normalized: NormalizedQuery, features: QueryFeatureSet) -> Sequence[SectionSemanticMatch]: ...


_SOURCE_TYPE_SPECS: tuple[ConceptSpec, ...] = (
    ConceptSpec("pdf", ("pdf", "pdf文档", "扫描件")),
    ConceptSpec("markdown", ("markdown", "md", "markdown文档")),
    ConceptSpec("docx", ("docx", "word", "word文档")),
    ConceptSpec("pptx", ("pptx", "ppt", "slide", "slides", "deck", "幻灯片")),
    ConceptSpec("xlsx", ("xlsx", "excel", "spreadsheet", "sheet", "工作表", "表格文件")),
    ConceptSpec("image", ("image", "images", "图片", "图像", "截图")),
)

_SPECIAL_TARGET_SPECS: tuple[ConceptSpec, ...] = (
    ConceptSpec("table", ("table", "tables", "表格", "数据表", "统计表", "指标", "数值")),
    ConceptSpec("figure", ("figure", "diagram", "chart", "image", "图片", "图像", "流程图")),
    ConceptSpec("ocr_region", ("ocr", "截图文字", "图片文字", "图中文字", "区域文字", "识别结果")),
    ConceptSpec("image_summary", ("image summary", "visual summary", "图像摘要", "画面内容", "图像说明")),
    ConceptSpec("caption", ("caption", "图注", "图题", "说明文字")),
    ConceptSpec("formula", ("formula", "equation", "latex", "公式", "数学表达式")),
)

_SECTION_FAMILY_SPECS: tuple[SectionFamilySpec, ...] = (
    SectionFamilySpec(
        family="architecture",
        canonical_terms=("系统架构", "技术架构", "architecture"),
        aliases=("系统架构", "技术架构", "架构", "分层", "层级", "模块", "组件", "architecture", "layers"),
    ),
    SectionFamilySpec(
        family="overview",
        canonical_terms=("概览", "overview"),
        aliases=("概览", "总体", "简介", "背景", "overview", "introduction", "background"),
    ),
    SectionFamilySpec(
        family="process",
        canonical_terms=("流程", "workflow"),
        aliases=("流程", "过程", "链路", "步骤", "工作流", "workflow", "pipeline", "process"),
    ),
    SectionFamilySpec(
        family="deployment",
        canonical_terms=("部署", "deployment"),
        aliases=("部署", "配置", "安装", "运维", "deployment", "setup", "configuration", "install"),
    ),
    SectionFamilySpec(
        family="evaluation",
        canonical_terms=("评估", "evaluation"),
        aliases=("评估", "实验", "结果", "性能", "benchmark", "metrics", "evaluation", "results"),
    ),
    SectionFamilySpec(
        family="conclusion",
        canonical_terms=("总结", "summary"),
        aliases=("总结", "结论", "回顾", "summary", "conclusion", "takeaway"),
    ),
)


def section_family_aliases(family: str) -> tuple[str, ...]:
    normalized_family = family.strip().lower()
    for spec in _SECTION_FAMILY_SPECS:
        if spec.family == normalized_family:
            return spec.aliases
    return ()


def special_target_aliases(target: str) -> tuple[str, ...]:
    normalized_target = target.strip().lower()
    for spec in _SPECIAL_TARGET_SPECS:
        if spec.canonical == normalized_target:
            return spec.aliases
    return ()


class QueryNormalizer:
    def normalize(self, query: str) -> NormalizedQuery:
        normalized = _NORMALIZE_SPACE_RE.sub(" ", query.strip())
        lowered = normalized.lower()
        quoted_terms = tuple(_ordered_unique(_QUOTED_TERM_RE.findall(normalized)))
        return NormalizedQuery(
            raw_text=query,
            text=normalized,
            lowered=lowered,
            tokens=search_terms(normalized),
            focus_terms=focus_terms(normalized),
            clauses=tuple(clause for clause in split_sentences(normalized) if clause),
            quoted_terms=quoted_terms,
            unit_count=text_unit_count(normalized),
        )


class QueryFeatureExtractor:
    def extract(self, normalized: NormalizedQuery) -> QueryFeatureSet:
        page_ranges = self._page_ranges(normalized.text)
        range_pages = {page for item in page_ranges for page in range(item.start, item.end + 1)}
        page_numbers = tuple(
            page
            for page in self._page_numbers(normalized.text)
            if page not in range_pages
        )
        source_types = tuple(self._concept_hits(normalized, _SOURCE_TYPE_SPECS))
        special_targets = tuple(self._concept_hits(normalized, _SPECIAL_TARGET_SPECS))
        structure_locators = self._marker_hits(normalized.lowered, _STRUCTURE_LOCATORS)
        location_markers = self._marker_hits(normalized.lowered, _LOCATION_MARKERS)
        comparison_markers = self._marker_hits(normalized.lowered, _COMPARISON_MARKERS)
        summary_markers = self._marker_hits(normalized.lowered, _SUMMARY_MARKERS)
        process_markers = self._marker_hits(normalized.lowered, _PROCESS_MARKERS)
        factual_markers = self._marker_hits(normalized.lowered, _FACTUAL_MARKERS)
        semantic_markers = self._marker_hits(normalized.lowered, _SEMANTIC_MARKERS)
        graph_markers = self._marker_hits(normalized.lowered, _GRAPH_MARKERS)
        topical_terms = self._topical_terms(normalized)
        title_hints, heading_hints = self._structural_hints(
            normalized=normalized,
            topical_terms=topical_terms,
            structure_locators=structure_locators,
        )
        document_titles, file_names = self._document_hints(normalized)
        vague_reference = self._is_vague_reference(normalized, topical_terms)
        return QueryFeatureSet(
            page_numbers=page_numbers,
            page_ranges=page_ranges,
            source_types=source_types,
            special_targets=special_targets,
            title_hints=title_hints,
            heading_hints=heading_hints,
            topical_terms=topical_terms,
            structure_locators=structure_locators,
            location_markers=location_markers,
            comparison_markers=comparison_markers,
            summary_markers=summary_markers,
            process_markers=process_markers,
            factual_markers=factual_markers,
            semantic_markers=semantic_markers,
            graph_markers=graph_markers,
            document_titles=document_titles,
            file_names=file_names,
            vague_reference=vague_reference,
        )

    def _page_ranges(self, query: str) -> tuple[PageRangeConstraint, ...]:
        ranges: list[PageRangeConstraint] = []
        for match in _PAGE_RANGE_RE.finditer(query):
            start_value = match.group("cjk_start") or match.group("en_start")
            end_value = match.group("cjk_end") or match.group("en_end")
            if start_value is None or end_value is None:
                continue
            start = int(start_value)
            end = int(end_value)
            if start > end:
                start, end = end, start
            candidate = PageRangeConstraint(start=start, end=end)
            if candidate not in ranges:
                ranges.append(candidate)
        return tuple(ranges)

    def _page_numbers(self, query: str) -> tuple[int, ...]:
        numbers: list[int] = []
        for match in _PAGE_RE.finditer(query):
            value = match.group("cjk") or match.group("en") or match.group("short")
            if value is None:
                continue
            page = int(value)
            if page not in numbers:
                numbers.append(page)
        return tuple(numbers)

    @staticmethod
    def _marker_hits(query: str, markers: Sequence[str]) -> tuple[str, ...]:
        return tuple(marker for marker in markers if marker in query)

    @staticmethod
    def _concept_hits(normalized: NormalizedQuery, concepts: Sequence[ConceptSpec]) -> list[str]:
        hits: list[str] = []
        token_set = set(normalized.tokens)
        lowered = normalized.lowered
        for concept in concepts:
            if any(alias in lowered for alias in concept.aliases):
                hits.append(concept.canonical)
                continue
            alias_terms = {alias for alias in concept.aliases if alias in token_set}
            if alias_terms:
                hits.append(concept.canonical)
        return hits

    def _topical_terms(self, normalized: NormalizedQuery) -> tuple[str, ...]:
        candidates: list[str] = list(normalized.quoted_terms)
        for term in normalized.focus_terms:
            if not self._is_viable_topic(term):
                continue
            candidates.append(term)
        for match in _TOPIC_PATTERN_RE.finditer(normalized.text):
            topic = match.group("topic").strip().strip("：:，,。.？? ")
            if self._is_viable_topic(topic):
                candidates.append(topic)
        return tuple(_ordered_unique(candidates)[:6])

    def _structural_hints(
        self,
        *,
        normalized: NormalizedQuery,
        topical_terms: Sequence[str],
        structure_locators: Sequence[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not structure_locators:
            return normalized.quoted_terms[:3], ()
        heading_hints: list[str] = list(normalized.quoted_terms)
        title_hints: list[str] = list(normalized.quoted_terms)
        for term in topical_terms:
            if not self._is_viable_topic(term):
                continue
            if term in {"总结", "概括", "对比", "流程", "过程"}:
                continue
            heading_hints.append(term)
        return tuple(_ordered_unique(title_hints)[:3]), tuple(_ordered_unique(heading_hints)[:4])

    @staticmethod
    def _document_hints(normalized: NormalizedQuery) -> tuple[tuple[str, ...], tuple[str, ...]]:
        titles = _ordered_unique(
            [*(_DOC_TITLE_RE.findall(normalized.text)), *(_DOC_TITLE_RE_REVERSED.findall(normalized.text))]
        )
        document_titles: list[str] = []
        file_names: list[str] = []
        for title in titles:
            if "." in title and title.rsplit(".", 1)[-1].lower() in {"pdf", "docx", "pptx", "xlsx", "md"}:
                file_names.append(title)
                continue
            document_titles.append(title)
        return tuple(document_titles), tuple(file_names)

    @staticmethod
    def _is_vague_reference(normalized: NormalizedQuery, topical_terms: Sequence[str]) -> bool:
        if topical_terms:
            return False
        return any(token in _VAGUE_REFERENCES for token in normalized.tokens)

    @staticmethod
    def _is_viable_topic(term: str) -> bool:
        normalized = term.strip().lower()
        units = text_unit_count(normalized)
        if units < 2 or units > 14:
            return False
        if normalized in _GENERIC_TERMS:
            return False
        if all(char.isdigit() for char in normalized):
            return False
        return True


class CatalogBackedSectionSemanticMatcher:
    def match(self, normalized: NormalizedQuery, features: QueryFeatureSet) -> Sequence[SectionSemanticMatch]:
        phrases = _ordered_unique([*features.heading_hints, *features.topical_terms, *normalized.quoted_terms])
        if not phrases:
            phrases = [term for term in normalized.tokens if 1 < text_unit_count(term) <= 12]
        matches: list[SectionSemanticMatch] = []
        for spec in _SECTION_FAMILY_SPECS:
            best_score = 0.0
            matched_terms: list[str] = []
            for phrase in phrases:
                score = self._match_phrase(phrase, normalized.lowered, spec)
                if score <= 0.0:
                    continue
                if score > best_score:
                    best_score = score
                    matched_terms = [phrase]
                elif phrase not in matched_terms:
                    matched_terms.append(phrase)
            if best_score <= 0.0:
                continue
            matches.append(
                SectionSemanticMatch(
                    family=spec.family,
                    canonical_term=spec.canonical_terms[0],
                    matched_terms=tuple(matched_terms[:3]),
                    score=round(best_score, 4),
                )
            )
        matches.sort(key=lambda item: (-item.score, item.family))
        return tuple(matches)

    @staticmethod
    def _match_phrase(phrase: str, lowered_query: str, spec: SectionFamilySpec) -> float:
        normalized_phrase = phrase.strip().lower()
        if not normalized_phrase:
            return 0.0
        if any(alias in normalized_phrase for alias in spec.aliases):
            return 0.95
        phrase_terms = set(search_terms(normalized_phrase))
        if not phrase_terms:
            return 0.0
        best = 0.0
        for alias in spec.aliases:
            alias_terms = set(search_terms(alias))
            if not alias_terms:
                continue
            overlap = len(phrase_terms & alias_terms)
            if overlap == 0:
                continue
            score = overlap / max(len(alias_terms), 1)
            if alias in lowered_query:
                score += 0.12
            best = max(best, min(score, 0.92))
        return best


class SemanticSignalAnalyzer:
    def __init__(self, *, section_matcher: SectionSemanticMatcher | None = None) -> None:
        self._section_matcher = section_matcher or CatalogBackedSectionSemanticMatcher()

    def analyze(self, normalized: NormalizedQuery, features: QueryFeatureSet) -> SemanticSignals:
        section_matches = tuple(self._section_matcher.match(normalized, features))
        structural_signal = (
            len(features.structure_locators) * 0.7
            + len(features.heading_hints) * 0.6
            + len(features.title_hints) * 0.6
            + sum(match.score for match in section_matches[:3]) * 0.9
        )
        metadata_signal = (
            len(features.page_numbers) * 0.9
            + len(features.page_ranges) * 1.1
            + len(features.source_types) * 0.9
            + len(features.document_titles) * 0.7
            + len(features.file_names) * 0.7
        )
        special_signal = len(features.special_targets) * 1.1
        intent_scores: dict[QueryIntent, float] = defaultdict(float)

        intent_scores[QueryIntent.COMPARISON_REQUEST] = (
            len(features.comparison_markers) * 2.0
            + max(len(normalized.clauses) - 1, 0) * 0.35
            + min(len(features.topical_terms), 4) * 0.12
        )
        intent_scores[QueryIntent.SUMMARY_REQUEST] = (
            len(features.summary_markers) * 1.9
            + max(len(normalized.clauses) - 1, 0) * 0.3
            + min(len(features.topical_terms), 4) * 0.1
        )
        intent_scores[QueryIntent.FLOW_PROCESS_REQUEST] = (
            len(features.process_markers) * 1.9
            + len(features.graph_markers) * 0.5
            + sum(match.score for match in section_matches if match.family == "process") * 0.8
        )
        intent_scores[QueryIntent.SECTION_LOOKUP] = (
            len(features.location_markers) * 1.4
            + len(features.heading_hints) * 0.9
            + len(features.title_hints) * 0.7
            + sum(match.score for match in section_matches[:2]) * 0.8
        )
        intent_scores[QueryIntent.STRUCTURE_LOOKUP] = (
            structural_signal * 1.15
            + sum(match.score for match in section_matches if match.family == "architecture") * 0.4
        )
        intent_scores[QueryIntent.METADATA_CONSTRAINED_LOOKUP] = metadata_signal * 1.3 + special_signal * 0.3
        intent_scores[QueryIntent.SPECIAL_CONTENT_LOOKUP] = special_signal * 1.7 + metadata_signal * 0.25
        intent_scores[QueryIntent.LOCALIZED_LOOKUP] = (
            metadata_signal * 0.8
            + len(features.location_markers) * 1.0
            + len(features.page_numbers) * 0.4
        )
        intent_scores[QueryIntent.FACTUAL_LOOKUP] = (
            len(features.factual_markers) * 1.7
            + len(features.page_numbers) * 0.25
            + len(features.source_types) * 0.18
        )
        intent_scores[QueryIntent.SEMANTIC_LOOKUP] = (
            len(features.semantic_markers) * 1.5
            + min(len(features.topical_terms), 5) * 0.22
            + (0.25 if normalized.unit_count >= 12 else 0.0)
        )
        return SemanticSignals(
            intent_scores={intent: round(score, 6) for intent, score in intent_scores.items()},
            section_matches=section_matches,
            structural_signal=round(structural_signal, 6),
            metadata_signal=round(metadata_signal, 6),
            special_signal=round(special_signal, 6),
        )


class IntentDecisionEngine:
    def decide(
        self,
        normalized: NormalizedQuery,
        features: QueryFeatureSet,
        signals: SemanticSignals,
    ) -> IntentDecision:
        ranked = sorted(signals.intent_scores.items(), key=lambda item: (-item[1], item[0].value))
        intent = ranked[0][0] if ranked else QueryIntent.SEMANTIC_LOOKUP
        if intent is QueryIntent.STRUCTURE_LOOKUP and (features.heading_hints or features.title_hints):
            intent = QueryIntent.SECTION_LOOKUP
        if intent is QueryIntent.FACTUAL_LOOKUP and signals.structural_signal >= 0.78:
            intent = QueryIntent.STRUCTURE_LOOKUP
        if intent is QueryIntent.FACTUAL_LOOKUP and signals.metadata_signal >= 0.9:
            intent = QueryIntent.METADATA_CONSTRAINED_LOOKUP
        if intent is QueryIntent.LOCALIZED_LOOKUP and signals.metadata_signal >= 1.2:
            intent = QueryIntent.METADATA_CONSTRAINED_LOOKUP
        if (
            features.special_targets
            and intent
            not in {
                QueryIntent.COMPARISON_REQUEST,
                QueryIntent.SUMMARY_REQUEST,
                QueryIntent.FLOW_PROCESS_REQUEST,
            }
            and signals.special_signal >= 1.0
            and signals.special_signal + 0.35 >= signals.metadata_signal
        ):
            intent = QueryIntent.SPECIAL_CONTENT_LOOKUP
        if intent is QueryIntent.FACTUAL_LOOKUP and signals.special_signal > 0.0:
            intent = QueryIntent.SPECIAL_CONTENT_LOOKUP
        return IntentDecision(
            intent=intent,
            query_type=self._query_type(
                intent=intent,
                normalized=normalized,
                features=features,
                signals=signals,
            ),
        )

    @staticmethod
    def _query_type(
        *,
        intent: QueryIntent,
        normalized: NormalizedQuery,
        features: QueryFeatureSet,
        signals: SemanticSignals,
    ) -> str:
        if intent is QueryIntent.SPECIAL_CONTENT_LOOKUP:
            if len(features.special_targets) == 1:
                prefix = "page_scoped_" if features.page_numbers or features.page_ranges else ""
                return f"{prefix}{features.special_targets[0]}_lookup"
            return "multi_special_lookup"
        if intent is QueryIntent.METADATA_CONSTRAINED_LOOKUP:
            if features.page_numbers or features.page_ranges:
                if features.source_types:
                    return "page_and_source_constrained_lookup"
                return "page_constrained_lookup"
            if features.source_types:
                return "source_type_constrained_lookup"
            return "metadata_constrained_lookup"
        if intent is QueryIntent.SECTION_LOOKUP:
            if features.title_hints:
                return "title_lookup"
            if features.heading_hints:
                return "heading_lookup"
            return "semantic_section_lookup"
        if intent is QueryIntent.STRUCTURE_LOOKUP:
            if any(match.family == "architecture" for match in signals.section_matches):
                return "architecture_lookup"
            return "semantic_structure_lookup"
        if intent is QueryIntent.FLOW_PROCESS_REQUEST:
            return "workflow_request"
        if intent is QueryIntent.SUMMARY_REQUEST:
            return "document_summary_request" if len(normalized.clauses) <= 2 else "multi_aspect_summary_request"
        if intent is QueryIntent.COMPARISON_REQUEST:
            return "comparative_lookup"
        if intent is QueryIntent.LOCALIZED_LOOKUP:
            return "location_scoped_lookup"
        if intent is QueryIntent.FACTUAL_LOOKUP:
            return "definition_lookup" if len(features.factual_markers) > 0 else "factual_lookup"
        return "semantic_lookup"


class ConstraintExtractor:
    def extract(
        self,
        *,
        features: QueryFeatureSet,
        signals: SemanticSignals,
        intent: IntentDecision,
    ) -> tuple[StructureConstraints, MetadataFilters, list[str]]:
        preferred_section_terms = _ordered_unique(
            [
                *features.heading_hints,
                *features.title_hints,
                *(match.canonical_term for match in signals.section_matches),
                *(term for match in signals.section_matches for term in match.matched_terms),
            ]
        )[:5]
        semantic_families = [
            match.family for match in signals.section_matches if match.score >= 0.38
        ]
        requires_structure_match = bool(
            intent.intent in {QueryIntent.STRUCTURE_LOOKUP, QueryIntent.SECTION_LOOKUP, QueryIntent.LOCALIZED_LOOKUP}
            and (preferred_section_terms or semantic_families or features.structure_locators)
        )
        prefer_heading_match = bool(
            features.heading_hints
            or features.title_hints
            or intent.intent is QueryIntent.SECTION_LOOKUP
        )
        if prefer_heading_match:
            match_strategy = "semantic_heading"
        elif semantic_families:
            match_strategy = "semantic_section"
        elif features.structure_locators:
            match_strategy = "structural_locator"
        else:
            match_strategy = "none"
        structure_constraints = StructureConstraints(
            match_strategy=match_strategy,
            requires_structure_match=requires_structure_match,
            prefer_heading_match=prefer_heading_match,
            semantic_section_families=semantic_families,
            preferred_section_terms=preferred_section_terms,
            heading_hints=list(features.heading_hints),
            title_hints=list(features.title_hints),
            locator_terms=list(features.structure_locators),
        )
        metadata_filters = MetadataFilters(
            page_numbers=list(features.page_numbers),
            page_ranges=list(features.page_ranges),
            source_types=list(features.source_types),
            document_titles=list(features.document_titles),
            file_names=list(features.file_names),
        )
        return structure_constraints, metadata_filters, preferred_section_terms


class RoutingHintDecider:
    def decide(
        self,
        *,
        normalized: NormalizedQuery,
        features: QueryFeatureSet,
        signals: SemanticSignals,
        intent: IntentDecision,
        structure_constraints: StructureConstraints,
        metadata_filters: MetadataFilters,
    ) -> RoutingHints:
        dense_priority = 0.18
        sparse_priority = 0.18
        structure_priority = 0.0
        metadata_priority = 0.0
        special_priority = 0.0
        graph_priority = 0.0

        if intent.intent in {
            QueryIntent.SEMANTIC_LOOKUP,
            QueryIntent.SUMMARY_REQUEST,
            QueryIntent.FLOW_PROCESS_REQUEST,
            QueryIntent.COMPARISON_REQUEST,
        }:
            dense_priority += 0.62
        if intent.intent in {
            QueryIntent.SUMMARY_REQUEST,
            QueryIntent.FLOW_PROCESS_REQUEST,
            QueryIntent.COMPARISON_REQUEST,
        }:
            sparse_priority += 0.28
        if intent.intent in {
            QueryIntent.FACTUAL_LOOKUP,
            QueryIntent.LOCALIZED_LOOKUP,
            QueryIntent.SECTION_LOOKUP,
            QueryIntent.STRUCTURE_LOOKUP,
            QueryIntent.METADATA_CONSTRAINED_LOOKUP,
            QueryIntent.SPECIAL_CONTENT_LOOKUP,
        }:
            sparse_priority += 0.6
            dense_priority += 0.3
        if structure_constraints.has_constraints():
            if intent.intent in {QueryIntent.SECTION_LOOKUP, QueryIntent.STRUCTURE_LOOKUP}:
                structure_priority += 0.72
            else:
                structure_priority += 0.46
            sparse_priority += 0.08
        if metadata_filters.has_constraints():
            metadata_priority += 0.9
            sparse_priority += 0.12
        if features.special_targets:
            special_priority += 0.92
            sparse_priority += 0.16
        if intent.intent in {
            QueryIntent.COMPARISON_REQUEST,
            QueryIntent.FLOW_PROCESS_REQUEST,
            QueryIntent.STRUCTURE_LOOKUP,
        }:
            graph_priority += 0.68
        graph_priority += min(len(features.graph_markers) * 0.12, 0.24)
        dense_priority += min(len(features.topical_terms) * 0.04, 0.18)
        sparse_priority += 0.08 if normalized.unit_count <= 12 else 0.0
        structure_priority += min(len(structure_constraints.semantic_section_families) * 0.08, 0.16)

        channel_scores = {
            "dense": min(dense_priority, 1.0),
            "sparse": min(sparse_priority, 1.0),
            "structure": min(structure_priority, 1.0),
            "metadata": min(metadata_priority, 1.0),
            "special": min(special_priority, 1.0),
            "graph": min(graph_priority, 1.0),
        }
        primary_channels = [
            channel
            for channel, score in sorted(channel_scores.items(), key=lambda item: (-item[1], item[0]))
            if score >= 0.45
        ]
        rewrite_focus_terms = _ordered_unique(
            [
                *structure_constraints.preferred_section_terms,
                *features.document_titles,
                *features.file_names,
                *features.topical_terms,
            ]
        )[:6]
        decomposition_axes = _ordered_unique(
            [
                *normalized.quoted_terms,
                *features.document_titles,
                *features.special_targets,
                *features.topical_terms,
            ]
        )[:6]
        return RoutingHints(
            dense_priority=round(channel_scores["dense"], 4),
            sparse_priority=round(channel_scores["sparse"], 4),
            structure_priority=round(channel_scores["structure"], 4),
            metadata_priority=round(channel_scores["metadata"], 4),
            special_priority=round(channel_scores["special"], 4),
            graph_priority=round(channel_scores["graph"], 4),
            primary_channels=primary_channels,
            rewrite_focus_terms=rewrite_focus_terms,
            decomposition_axes=decomposition_axes,
        )


class ConfidenceScorer:
    def score(
        self,
        *,
        normalized: NormalizedQuery,
        features: QueryFeatureSet,
        signals: SemanticSignals,
        routing_hints: RoutingHints,
    ) -> ConfidenceAssessment:
        ranked = sorted(signals.intent_scores.values(), reverse=True)
        top_score = ranked[0] if ranked else 0.0
        second_score = ranked[1] if len(ranked) > 1 else 0.0
        coverage = min(top_score / 2.8, 1.0)
        margin = min(max(top_score - second_score, 0.0) / 1.8, 1.0)
        specificity = 0.0
        specificity += 0.18 if features.page_numbers or features.page_ranges else 0.0
        specificity += 0.14 if features.source_types else 0.0
        specificity += 0.14 if features.special_targets else 0.0
        specificity += 0.14 if features.heading_hints or features.title_hints else 0.0
        specificity += min(len(features.topical_terms), 4) * 0.05
        specificity += 0.08 if routing_hints.primary_channels else 0.0
        if features.vague_reference:
            specificity -= 0.18
        if normalized.unit_count <= 4:
            specificity -= 0.08
        raw = 0.42 * coverage + 0.36 * margin + 0.22 * max(min(specificity, 1.0), 0.0)
        confidence = round(max(min(raw, 0.97), 0.18), 4)
        if confidence >= 0.75:
            band = ConfidenceBand.HIGH
        elif confidence >= 0.5:
            band = ConfidenceBand.MEDIUM
        else:
            band = ConfidenceBand.LOW
        return ConfidenceAssessment(confidence=confidence, band=band)


class QueryUnderstandingAssembler:
    def assemble(
        self,
        *,
        normalized: NormalizedQuery,
        features: QueryFeatureSet,
        intent: IntentDecision,
        structure_constraints: StructureConstraints,
        metadata_filters: MetadataFilters,
        preferred_section_terms: Sequence[str],
        routing_hints: RoutingHints,
        confidence: ConfidenceAssessment,
    ) -> QueryUnderstanding:
        needs_dense = routing_hints.dense_priority >= 0.45
        needs_sparse = routing_hints.sparse_priority >= 0.45
        if confidence.band is ConfidenceBand.LOW:
            needs_dense = True
            needs_sparse = True
        needs_structure = routing_hints.structure_priority >= 0.45 or structure_constraints.has_constraints()
        needs_metadata = routing_hints.metadata_priority >= 0.55 or metadata_filters.has_constraints()
        needs_special = routing_hints.special_priority >= 0.55 or bool(features.special_targets)
        needs_graph_expansion = routing_hints.graph_priority >= 0.58 and confidence.band is not ConfidenceBand.LOW
        should_decompose_query = self._should_decompose_query(
            normalized=normalized,
            intent=intent.intent,
            features=features,
        )
        should_rewrite_query = self._should_rewrite_query(
            normalized=normalized,
            features=features,
            confidence=confidence,
        )
        return QueryUnderstanding(
            intent=intent.intent,
            query_type=intent.query_type,
            confidence=confidence.confidence,
            confidence_band=confidence.band,
            needs_dense=needs_dense,
            needs_sparse=needs_sparse,
            needs_special=needs_special,
            needs_structure=needs_structure,
            needs_metadata=needs_metadata,
            needs_graph_expansion=needs_graph_expansion,
            should_rewrite_query=should_rewrite_query,
            should_decompose_query=should_decompose_query,
            structure_constraints=structure_constraints,
            metadata_filters=metadata_filters,
            special_targets=list(features.special_targets),
            preferred_section_terms=list(preferred_section_terms),
            routing_hints=routing_hints,
        )

    @staticmethod
    def _should_decompose_query(
        *,
        normalized: NormalizedQuery,
        intent: QueryIntent,
        features: QueryFeatureSet,
    ) -> bool:
        if intent is QueryIntent.COMPARISON_REQUEST:
            return True
        if intent in {QueryIntent.SUMMARY_REQUEST, QueryIntent.FLOW_PROCESS_REQUEST}:
            return len(normalized.clauses) > 1 or len(features.topical_terms) >= 3
        return len(features.source_types) > 1 or len(features.special_targets) > 1

    @staticmethod
    def _should_rewrite_query(
        *,
        normalized: NormalizedQuery,
        features: QueryFeatureSet,
        confidence: ConfidenceAssessment,
    ) -> bool:
        if features.page_numbers or features.page_ranges or features.source_types:
            return False
        if features.special_targets or features.heading_hints or features.title_hints:
            return False
        if confidence.band is ConfidenceBand.LOW:
            return True
        if features.vague_reference:
            return True
        if normalized.unit_count >= 18 and len(features.topical_terms) <= 1:
            return True
        return False


class QueryUnderstandingService:
    def __init__(
        self,
        *,
        normalizer: QueryNormalizer | None = None,
        feature_extractor: QueryFeatureExtractor | None = None,
        semantic_signal_analyzer: SemanticSignalAnalyzer | None = None,
        intent_engine: IntentDecisionEngine | None = None,
        constraint_extractor: ConstraintExtractor | None = None,
        routing_hint_decider: RoutingHintDecider | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        assembler: QueryUnderstandingAssembler | None = None,
    ) -> None:
        self._normalizer = normalizer or QueryNormalizer()
        self._feature_extractor = feature_extractor or QueryFeatureExtractor()
        self._semantic_signal_analyzer = semantic_signal_analyzer or SemanticSignalAnalyzer()
        self._intent_engine = intent_engine or IntentDecisionEngine()
        self._constraint_extractor = constraint_extractor or ConstraintExtractor()
        self._routing_hint_decider = routing_hint_decider or RoutingHintDecider()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._assembler = assembler or QueryUnderstandingAssembler()

    def analyze(self, query: str) -> QueryUnderstanding:
        normalized = self._normalizer.normalize(query)
        features = self._feature_extractor.extract(normalized)
        signals = self._semantic_signal_analyzer.analyze(normalized, features)
        intent = self._intent_engine.decide(normalized, features, signals)
        structure_constraints, metadata_filters, preferred_section_terms = self._constraint_extractor.extract(
            features=features,
            signals=signals,
            intent=intent,
        )
        routing_hints = self._routing_hint_decider.decide(
            normalized=normalized,
            features=features,
            signals=signals,
            intent=intent,
            structure_constraints=structure_constraints,
            metadata_filters=metadata_filters,
        )
        confidence = self._confidence_scorer.score(
            normalized=normalized,
            features=features,
            signals=signals,
            routing_hints=routing_hints,
        )
        return self._assembler.assemble(
            normalized=normalized,
            features=features,
            intent=intent,
            structure_constraints=structure_constraints,
            metadata_filters=metadata_filters,
            preferred_section_terms=preferred_section_terms,
            routing_hints=routing_hints,
            confidence=confidence,
        )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


__all__ = [
    "CatalogBackedSectionSemanticMatcher",
    "QueryFeatureExtractor",
    "QueryNormalizer",
    "QueryUnderstandingService",
    "SemanticSignalAnalyzer",
    "section_family_aliases",
    "special_target_aliases",
]
