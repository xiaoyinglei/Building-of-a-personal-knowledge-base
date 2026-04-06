from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag.schema._types.access import AccessPolicy, RuntimeMode
from rag.schema._types.query import (
    ComplexityLevel,
    MetadataFilters,
    PageRangeConstraint,
    QueryUnderstanding,
    StructureConstraints,
    TaskType,
)
from rag.schema._types.text import focus_terms

_NORMALIZE_SPACE_RE = re.compile(r"\s+")
_QUOTED_TERM_RE = re.compile(r'''["“”'‘’]([^"“”'‘’]{2,64})["“”'‘’]''')
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
_COMPARISON_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "difference",
    "contrast",
    "对比",
    "比较",
    "区别",
    "差异",
)
_SUMMARY_MARKERS = (
    "summary",
    "summarize",
    "overview",
    "总结",
    "概括",
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
    "流程",
    "过程",
    "链路",
    "步骤",
    "工作流",
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
)
_STRUCTURE_MARKERS = (
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
_SOURCE_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "pdf": ("pdf", "pdf文档", "扫描件"),
    "markdown": ("markdown", "md", "markdown文档"),
    "docx": ("docx", "word", "word文档"),
    "pptx": ("pptx", "ppt", "slide", "slides", "deck", "幻灯片"),
    "xlsx": ("xlsx", "excel", "spreadsheet", "sheet", "工作表", "表格文件"),
    "image": ("image", "images", "图片", "图像", "截图"),
}
_SPECIAL_TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "table": ("table", "tables", "表格", "数据表", "统计表", "指标", "数值"),
    "figure": ("figure", "diagram", "chart", "image", "图片", "图像", "流程图"),
    "ocr_region": ("ocr", "截图文字", "图片文字", "图中文字", "区域文字", "识别结果"),
    "image_summary": ("image summary", "visual summary", "图像摘要", "画面内容", "图像说明"),
    "caption": ("caption", "图注", "图题", "说明文字"),
    "formula": ("formula", "equation", "latex", "公式", "数学表达式"),
}
_SECTION_FAMILY_TERMS: dict[str, tuple[str, ...]] = {
    "architecture": ("系统架构", "技术架构", "architecture"),
    "overview": ("概览", "overview"),
    "process": ("流程", "workflow"),
    "deployment": ("部署", "deployment"),
    "evaluation": ("评估", "evaluation"),
    "conclusion": ("总结", "summary"),
}
_SECTION_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "architecture": ("系统架构", "技术架构", "架构", "分层", "层级", "模块", "组件", "architecture", "layers"),
    "overview": ("概览", "总体", "简介", "背景", "overview", "introduction", "background"),
    "process": ("流程", "过程", "链路", "步骤", "工作流", "workflow", "pipeline", "process"),
    "deployment": ("部署", "配置", "安装", "运维", "deployment", "setup", "configuration", "install"),
    "evaluation": ("评估", "实验", "结果", "性能", "benchmark", "metrics", "evaluation", "results"),
    "conclusion": ("总结", "结论", "回顾", "summary", "conclusion", "takeaway"),
}
_DEEP_TASK_TYPES = {TaskType.COMPARISON, TaskType.SYNTHESIS, TaskType.TIMELINE, TaskType.RESEARCH}


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType
    complexity_level: ComplexityLevel
    runtime_mode: RuntimeMode
    source_scope: list[str] = Field(default_factory=list)
    web_search_allowed: bool = False
    graph_expansion_allowed: bool = False
    rerank_required: bool = True


class QueryUnderstandingService:
    def analyze(self, query: str) -> QueryUnderstanding:
        normalized = _NORMALIZE_SPACE_RE.sub(" ", query.strip())
        lowered = normalized.lower()
        quoted_terms = _ordered_unique(_QUOTED_TERM_RE.findall(normalized))[:4]
        topical_terms = self._topical_terms(normalized, quoted_terms)
        metadata_filters = self._metadata_filters(normalized, lowered)
        special_targets = self._concept_hits(lowered, _SPECIAL_TARGET_ALIASES)
        structure_constraints, preferred_section_terms = self._structure_constraints(
            lowered=lowered,
            quoted_terms=quoted_terms,
            topical_terms=topical_terms,
        )
        query_type = self._query_type(
            lowered=lowered,
            metadata_filters=metadata_filters,
            structure_constraints=structure_constraints,
            special_targets=special_targets,
        )
        return QueryUnderstanding(
            query_type=query_type,
            needs_special=bool(special_targets),
            needs_structure=structure_constraints.has_constraints(),
            needs_metadata=metadata_filters.has_constraints(),
            needs_graph_expansion=(query_type == "process" or any(marker in lowered for marker in _GRAPH_MARKERS)),
            structure_constraints=structure_constraints,
            metadata_filters=metadata_filters,
            special_targets=special_targets,
            preferred_section_terms=preferred_section_terms,
        )

    def _query_type(
        self,
        *,
        lowered: str,
        metadata_filters: MetadataFilters,
        structure_constraints: StructureConstraints,
        special_targets: list[str],
    ) -> str:
        if any(marker in lowered for marker in _COMPARISON_MARKERS):
            return "comparison"
        if any(marker in lowered for marker in _SUMMARY_MARKERS):
            return "summary"
        if any(marker in lowered for marker in _PROCESS_MARKERS):
            return "process"
        if special_targets:
            return "special_lookup"
        if structure_constraints.has_constraints():
            if structure_constraints.locator_terms:
                return "section_lookup"
            return "structure_lookup"
        if metadata_filters.has_constraints():
            return "scoped_lookup"
        return "lookup"

    def _topical_terms(self, normalized: str, quoted_terms: Sequence[str]) -> list[str]:
        candidates = [*quoted_terms]
        for term in focus_terms(normalized):
            stripped = term.strip()
            if not self._is_viable_topic(stripped):
                continue
            candidates.append(stripped)
        return _ordered_unique(candidates)[:6]

    def _metadata_filters(self, normalized: str, lowered: str) -> MetadataFilters:
        page_ranges = self._page_ranges(normalized)
        range_pages = {page for item in page_ranges for page in range(item.start, item.end + 1)}
        page_numbers = [page for page in self._page_numbers(normalized) if page not in range_pages]
        document_titles, file_names = self._document_hints(normalized)
        return MetadataFilters(
            page_numbers=page_numbers,
            page_ranges=page_ranges,
            source_types=self._concept_hits(lowered, _SOURCE_TYPE_ALIASES),
            document_titles=document_titles,
            file_names=file_names,
        )

    def _structure_constraints(
        self,
        *,
        lowered: str,
        quoted_terms: Sequence[str],
        topical_terms: Sequence[str],
    ) -> tuple[StructureConstraints, list[str]]:
        locator_terms = [marker for marker in (*_STRUCTURE_MARKERS, *_LOCATION_MARKERS) if marker in lowered]
        matched_families = [
            family
            for family, aliases in _SECTION_FAMILY_ALIASES.items()
            if any(alias in lowered for alias in aliases)
        ]
        has_structure_signal = bool(locator_terms or matched_families or quoted_terms)
        preferred_section_terms = (
            _ordered_unique(
                [
                    *(term for family in matched_families for term in _SECTION_FAMILY_TERMS.get(family, ())),
                    *quoted_terms,
                    *topical_terms,
                ]
            )[:6]
            if has_structure_signal
            else []
        )
        heading_hints = preferred_section_terms[:4] if (locator_terms or matched_families or quoted_terms) else []
        title_hints = list(quoted_terms[:3])
        requires_structure = bool(locator_terms or matched_families or heading_hints)
        prefer_heading_match = bool(locator_terms or heading_hints)
        constraints = StructureConstraints(
            match_strategy=(
                "heading" if prefer_heading_match else "semantic" if matched_families else "none"
            ),
            requires_structure_match=requires_structure,
            prefer_heading_match=prefer_heading_match,
            semantic_section_families=matched_families,
            preferred_section_terms=preferred_section_terms,
            heading_hints=heading_hints,
            title_hints=title_hints,
            locator_terms=locator_terms,
        )
        return constraints, preferred_section_terms

    @staticmethod
    def _page_ranges(query: str) -> list[PageRangeConstraint]:
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
        return ranges

    @staticmethod
    def _page_numbers(query: str) -> list[int]:
        numbers: list[int] = []
        for match in _PAGE_RE.finditer(query):
            value = match.group("cjk") or match.group("en") or match.group("short")
            if value is None:
                continue
            page = int(value)
            if page not in numbers:
                numbers.append(page)
        return numbers

    @staticmethod
    def _document_hints(query: str) -> tuple[list[str], list[str]]:
        titles = _ordered_unique([*(_DOC_TITLE_RE.findall(query)), *(_DOC_TITLE_RE_REVERSED.findall(query))])
        document_titles: list[str] = []
        file_names: list[str] = []
        for title in titles:
            if "." in title and title.rsplit(".", 1)[-1].lower() in {"pdf", "docx", "pptx", "xlsx", "md"}:
                file_names.append(title)
            else:
                document_titles.append(title)
        return document_titles, file_names

    @staticmethod
    def _concept_hits(query: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
        return [name for name, values in aliases.items() if any(alias in query for alias in values)]

    @staticmethod
    def _is_viable_topic(term: str) -> bool:
        normalized = term.strip().lower()
        return bool(normalized) and normalized not in _GENERIC_TERMS and len(normalized) >= 2


class RoutingService:
    def route(
        self,
        query: str,
        *,
        query_understanding: QueryUnderstanding,
        source_scope: Sequence[str] = (),
        access_policy: AccessPolicy | None = None,
    ) -> RoutingDecision:
        del query, access_policy
        task_type = self._task_type(query_understanding, source_scope)
        complexity_level = self._complexity_level(task_type, source_scope, query_understanding)
        runtime_mode = self._runtime_mode(task_type, source_scope, query_understanding)
        return RoutingDecision(
            task_type=task_type,
            complexity_level=complexity_level,
            runtime_mode=runtime_mode,
            source_scope=list(source_scope),
            web_search_allowed=not source_scope and task_type in _DEEP_TASK_TYPES,
            graph_expansion_allowed=(
                runtime_mode is RuntimeMode.DEEP and query_understanding.needs_graph_expansion
            ),
            rerank_required=True,
        )

    @staticmethod
    def _task_type(query_understanding: QueryUnderstanding, source_scope: Sequence[str]) -> TaskType:
        if query_understanding.query_type == "comparison":
            return TaskType.COMPARISON
        if query_understanding.query_type == "summary":
            return TaskType.SYNTHESIS if source_scope else TaskType.RESEARCH
        if query_understanding.query_type == "process":
            return TaskType.SYNTHESIS if source_scope else TaskType.RESEARCH
        if len(source_scope) == 1 or query_understanding.metadata_filters.has_constraints():
            return TaskType.SINGLE_DOC_QA
        return TaskType.LOOKUP

    @staticmethod
    def _complexity_level(
        task_type: TaskType,
        source_scope: Sequence[str],
        query_understanding: QueryUnderstanding,
    ) -> ComplexityLevel:
        if task_type is TaskType.COMPARISON:
            return ComplexityLevel.L3_COMPARATIVE
        if task_type in {TaskType.SYNTHESIS, TaskType.RESEARCH, TaskType.TIMELINE}:
            return ComplexityLevel.L4_RESEARCH if len(source_scope) != 1 else ComplexityLevel.L2_SCOPED
        if len(source_scope) == 1 or query_understanding.has_explicit_constraints():
            return ComplexityLevel.L2_SCOPED
        return ComplexityLevel.L1_DIRECT

    @staticmethod
    def _runtime_mode(
        task_type: TaskType,
        source_scope: Sequence[str],
        query_understanding: QueryUnderstanding,
    ) -> RuntimeMode:
        if query_understanding.needs_graph_expansion:
            return RuntimeMode.DEEP
        if task_type in {TaskType.COMPARISON, TaskType.RESEARCH}:
            return RuntimeMode.DEEP
        if task_type is TaskType.SYNTHESIS and len(source_scope) != 1:
            return RuntimeMode.DEEP
        return RuntimeMode.FAST


def section_family_aliases(family: str) -> tuple[str, ...]:
    return _SECTION_FAMILY_ALIASES.get(family.strip().lower(), ())


def special_target_aliases(target: str) -> tuple[str, ...]:
    return _SPECIAL_TARGET_ALIASES.get(target.strip().lower(), ())


def _ordered_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


__all__ = [
    "RoutingDecision",
    "RoutingService",
    "QueryUnderstandingService",
    "section_family_aliases",
    "special_target_aliases",
]
