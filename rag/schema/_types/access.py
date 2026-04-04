from __future__ import annotations

# 枚举类型（字符串枚举，既是枚举又能当字符串用）
from enum import StrEnum

# Pydantic：用于定义数据模型
from pydantic import BaseModel, ConfigDict, Field


# =========================
# 数据驻留策略（数据能不能上云）
# =========================
class Residency(StrEnum):
    CLOUD_ALLOWED = "cloud_allowed"      # 允许数据在云端处理
    LOCAL_PREFERRED = "local_preferred"  # 优先本地，但不强制
    LOCAL_REQUIRED = "local_required"    # 必须本地，不能上云


# =========================
# 外部检索策略（能不能联网 / 调外部数据）
# =========================
class ExternalRetrievalPolicy(StrEnum):
    ALLOW = "allow"  # 允许外部检索（如搜索引擎、外部API）
    DENY = "deny"    # 禁止外部检索


# =========================
# 运行模式（影响速度 vs 深度）
# =========================
class RuntimeMode(StrEnum):
    FAST = "fast"  # 快速模式（低延迟，可能效果一般）
    DEEP = "deep"  # 深度模式（更慢，但效果更好）


# =========================
# 执行位置（在哪跑）
# =========================
class ExecutionLocation(StrEnum):
    CLOUD = "cloud"  # 云端执行
    LOCAL = "local"  # 本地执行


# =========================
# 执行位置偏好（优先级策略）
# =========================
class ExecutionLocationPreference(StrEnum):
    CLOUD_FIRST = "cloud_first"  # 优先云端
    LOCAL_FIRST = "local_first"  # 优先本地
    LOCAL_ONLY = "local_only"    # 只允许本地


# =========================
# Residency 严格程度排序（用于比较哪个更“严格”）
# 数字越大，限制越强
# =========================
_RESIDENCY_ORDER: dict[Residency, int] = {
    Residency.CLOUD_ALLOWED: 0,      # 最宽松
    Residency.LOCAL_PREFERRED: 1,    # 中等
    Residency.LOCAL_REQUIRED: 2,     # 最严格
}


# =========================
# 核心策略类（访问/执行策略）
# =========================
class AccessPolicy(BaseModel):
    # Pydantic配置：冻结模型（不可修改）
    model_config = ConfigDict(frozen=True)

    # 数据驻留策略（是否允许上云）
    residency: Residency = Residency.CLOUD_ALLOWED

    # 是否允许外部检索（联网）
    external_retrieval: ExternalRetrievalPolicy = ExternalRetrievalPolicy.ALLOW

    # 允许的运行模式集合（FAST / DEEP）
    # 使用 frozenset：不可变集合（更安全）
    allowed_runtimes: frozenset[RuntimeMode] = Field(
        default_factory=lambda: frozenset({RuntimeMode.FAST, RuntimeMode.DEEP})
    )

    # 允许的执行位置（CLOUD / LOCAL）
    allowed_locations: frozenset[ExecutionLocation] = Field(
        default_factory=lambda: frozenset({ExecutionLocation.CLOUD, ExecutionLocation.LOCAL})
    )

    # 敏感标签（例如：pii / confidential / finance 等）
    sensitivity_tags: frozenset[str] = Field(default_factory=frozenset)

    # =========================
    # 默认策略
    # =========================
    @classmethod
    def default(cls) -> AccessPolicy:
        return cls()  # 返回默认配置

    # =========================
    # 策略收紧（最核心方法）
    # 把两个策略合并成一个“更严格”的策略
    # =========================
    def narrow(self, other: AccessPolicy) -> AccessPolicy:
        # 1. 运行模式：取交集（只能保留双方都允许的）
        allowed_runtimes = self.allowed_runtimes & other.allowed_runtimes
        if not allowed_runtimes:
            raise ValueError("allowed_runtimes cannot become empty during narrowing")

        # 2. 执行位置：取交集
        allowed_locations = self.allowed_locations & other.allowed_locations
        if not allowed_locations:
            raise ValueError("allowed_locations cannot become empty during narrowing")

        # 3. residency：选更严格的（通过排序表比较）
        residency = max((self.residency, other.residency), key=_RESIDENCY_ORDER.__getitem__)

        # 4. 外部检索：只要有一个 DENY → 最终就是 DENY（保守策略）
        external_retrieval = (
            ExternalRetrievalPolicy.DENY
            if ExternalRetrievalPolicy.DENY in {self.external_retrieval, other.external_retrieval}
            else ExternalRetrievalPolicy.ALLOW
        )

        # 5. 敏感标签：合并（并集）
        return AccessPolicy(
            residency=residency,
            external_retrieval=external_retrieval,
            allowed_runtimes=allowed_runtimes,
            allowed_locations=allowed_locations,
            sensitivity_tags=self.sensitivity_tags | other.sensitivity_tags,
        )

    # =========================
    # 是否“必须本地执行”
    # =========================
    @property
    def local_only(self) -> bool:
        return self.residency is Residency.LOCAL_REQUIRED

    # =========================
    # 判断是否允许某种运行模式
    # =========================
    def allows_runtime(self, mode: RuntimeMode) -> bool:
        return mode in self.allowed_runtimes

    # =========================
    # 判断是否允许某个执行位置
    # =========================
    def allows_location(self, location: ExecutionLocation) -> bool:
        return location in self.allowed_locations