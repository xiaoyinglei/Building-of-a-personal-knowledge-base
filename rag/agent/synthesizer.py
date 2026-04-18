"""Cross-subtask synthesis before final report rendering."""

from __future__ import annotations

from dataclasses import dataclass

from rag.agent.report import AgentReportBuilder
from rag.agent.state import AgentRunState
from rag.providers.generation import AnswerGenerationService, AnswerGenerator
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


@dataclass(frozen=True, slots=True)
class SynthesizedSections:
    executive_summary: str
    key_findings: list[str]
    risks: list[str]
    unknowns: list[str]
    recommendations: list[str]


class AgentSynthesizer:
    """Merge subtask outcomes before rendering the final report."""

    def __init__(
        self,
        *,
        answer_generator: AnswerGenerator | None = None,
        report_builder: AgentReportBuilder | None = None,
        verbosity: str = "balanced",
    ) -> None:
        self._answer_generator = answer_generator or AnswerGenerator(
            answer_generation_service=AnswerGenerationService()
        )
        self._report_builder = report_builder or AgentReportBuilder()
        self._verbosity = verbosity

    def synthesize(
        self,
        *,
        state: AgentRunState,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference,
    ):
        sections = self._programmatic_sections(state)
        executive_summary = self._executive_summary(
            sections=sections,
            access_policy=access_policy,
            execution_location_preference=execution_location_preference,
        )
        return self._report_builder.build(
            state=state,
            executive_summary=executive_summary,
            key_findings=sections.key_findings,
            risks=sections.risks,
            unknowns=sections.unknowns,
            recommendations=sections.recommendations,
        )

    @staticmethod
    def _programmatic_sections(state: AgentRunState) -> SynthesizedSections:
        confirmed_findings: list[str] = []
        risks: list[str] = []
        unknowns: list[str] = []
        recommendations: list[str] = []
        for result in state.subtask_results:
            if result.status == "completed":
                confirmed_findings.extend(result.findings[:2])
            risks.extend(result.evidence_assessment.conflicts)
            unknowns.extend(result.unresolved_questions)
        if not confirmed_findings:
            confirmed_findings.append("The agent could not confirm any high-confidence finding from the available evidence.")
        if not recommendations:
            if unknowns:
                recommendations.append("Expand evidence collection before making a high-confidence conclusion.")
            else:
                recommendations.append("Use the confirmed findings and citation map as the basis for downstream analysis.")
        return SynthesizedSections(
            executive_summary="",
            key_findings=_ordered_unique(confirmed_findings),
            risks=_ordered_unique(risks),
            unknowns=_ordered_unique(unknowns),
            recommendations=_ordered_unique(recommendations),
        )

    def _executive_summary(
        self,
        *,
        sections: SynthesizedSections,
        access_policy: AccessPolicy,
        execution_location_preference: ExecutionLocationPreference,
    ) -> str:
        prompt = self._summary_prompt(sections)
        if self._answer_generator.chat_bindings:
            generated = self._answer_generator.generate_direct(
                query="Summarize the structured agent findings.",
                prompt=prompt,
                access_policy=access_policy,
                execution_location_preference=execution_location_preference,
            )
            answer_text = generated.answer.answer_text.strip()
            if answer_text and "No chat-capable provider available" not in answer_text:
                return answer_text
        return self._deterministic_summary(sections, verbosity=self._verbosity)

    @staticmethod
    def _summary_prompt(sections: SynthesizedSections) -> str:
        risk_lines = [f"- {item}" for item in sections.risks] or ["- None"]
        unknown_lines = [f"- {item}" for item in sections.unknowns] or ["- None"]
        lines = [
            "You are writing an executive summary for an evidence-grounded analysis report.",
            "Write 2-4 concise sentences.",
            "Do not invent new claims.",
            "Confirmed findings:",
            *[f"- {item}" for item in sections.key_findings],
            "Risks:",
            *risk_lines,
            "Unknowns:",
            *unknown_lines,
        ]
        return "\n".join(lines)

    @staticmethod
    def _deterministic_summary(sections: SynthesizedSections, *, verbosity: str) -> str:
        summary = sections.key_findings[0]
        if verbosity == "concise":
            return summary
        if sections.unknowns:
            return f"{summary} Evidence gaps remain in {sections.unknowns[0]}."
        return summary


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


__all__ = ["AgentSynthesizer", "SynthesizedSections"]
