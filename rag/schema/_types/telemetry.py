from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str 
    category: str 
    payload: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationMetricInput(BaseModel):
    """Structured evaluation sample consumed by the telemetry harness."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    citation_precision: float = Field(ge=0.0, le=1.0) # 相关且支持答案的引用证据比例
    evidence_sufficient: bool # 答案是否有充分的证据支持
    conflict_detected: bool # 是否检测到证据之间的冲突
    simple_query_latency_seconds: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("latency_seconds", "simple_query_latency_seconds"),
    ) # 简单查询的响应时间，单位为秒
    deep_query_completion_quality: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("deep_quality", "deep_query_completion_quality"),
    ) # 深度查询的完成质量评分，范围从0到1
    preservation_useful: bool # 是否认为保存建议有用


class EvaluationMetricSummary(BaseModel):
    """Aggregate evaluation quality metrics."""

    model_config = ConfigDict(frozen=True)

    citation_precision: float
    evidence_sufficiency_rate: float
    conflict_detection_quality: float
    simple_query_latency: float
    deep_query_completion_quality: float
    preservation_usefulness: float

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


def coerce_evaluation_metric_input(
    item: EvaluationMetricInput | Mapping[str, Any],
) -> EvaluationMetricInput:
    if isinstance(item, EvaluationMetricInput):
        return item
    return EvaluationMetricInput.model_validate(item)
