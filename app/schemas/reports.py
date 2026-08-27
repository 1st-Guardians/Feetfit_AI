from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_FUNGAL_SUSPICION_SAFETY_DESCRIPTION = (
    "현재 업로드된 발 이미지 기준으로는 무좀으로 강하게 의심되는 뚜렷한 위험 신호가 크게 관찰되지 않습니다. "
    "일부 영역에서 색상이나 질감 차이가 보일 수 있으나 전반적으로는 안정적인 상태로 판단됩니다. "
    "다만 발가락 사이와 발바닥처럼 습기가 쉽게 머무는 부위는 평소처럼 건조하게 관리하고, "
    "가려움이나 각질 증가 같은 변화가 이어지는지 가볍게 확인해 주면 좋겠습니다."
)
DEFAULT_SKIN_REACTION_SAFETY_DESCRIPTION = (
    "피부 자극이나 염증 반응은 전반적으로 심하지 않은 편으로 보입니다. "
    "눈에 띄게 붉거나 넓게 번지는 양상은 제한적으로 판단되며, 현재 상태만 놓고 보면 급하게 걱정할 수준은 아닙니다. "
    "다만 장시간 신발 착용, 땀, 마찰로 인해 일시적인 자극이 생길 수 있으니 발을 깨끗하게 씻고 충분히 말리는 습관을 유지해 주세요."
)
DEFAULT_TOTAL_SCORE_DESCRIPTION = (
    "종합적으로 봤을 때 현재 발 상태는 비교적 양호한 편입니다. "
    "무좀 의심 부위와 피부 자극 반응 모두 크게 두드러지지 않아 일상적인 관리만 잘 유지해도 괜찮을 가능성이 높습니다. "
    "발을 건조하게 유지하고 통풍이 잘 되는 양말과 신발을 선택하는 것이 도움이 됩니다. "
    "이 결과는 이미지 기반의 참고용 분석이므로, 통증이나 심한 가려움, 갈라짐, 진물 같은 증상이 지속되면 전문 진료를 권장합니다."
)


class TineaPedisReportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    fungal_suspicion_safety_score: int = Field(alias="fungalSuspicionSafetyScore", ge=0, le=100)
    skin_reaction_safety_score: int = Field(alias="skinReactionSafetyScore", ge=0, le=100)
    fungal_suspicion_safety_description: str = Field(
        default=DEFAULT_FUNGAL_SUSPICION_SAFETY_DESCRIPTION,
        alias="fungalSuspicionSafetyDescription",
    )
    skin_reaction_safety_description: str = Field(
        default=DEFAULT_SKIN_REACTION_SAFETY_DESCRIPTION,
        alias="skinReactionSafetyDescription",
    )
    total_score_description: str = Field(
        default=DEFAULT_TOTAL_SCORE_DESCRIPTION,
        alias="totalScoreDescription",
    )


class HalluxValgusReportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    left_toe_angle_degree: float = Field(alias="leftToeAngleDegree", ge=0)
    right_toe_angle_degree: float = Field(alias="rightToeAngleDegree", ge=0)
    score_analysis_text: str = Field(alias="scoreAnalysisText")


class FootTypeAnalysisContext(BaseModel):
    """Already-classified facts that may be shown above the shoe list.

    Raw length/width measurements are retained as traceable context, but they
    are never converted into an arch or width category inside Feetfit_AI.  A
    categorical sentence is allowed only when the upstream analysis supplies
    the corresponding enum explicitly.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    arch_type: Literal["LOW", "NORMAL", "HIGH", "UNKNOWN"] = Field(
        default="UNKNOWN", alias="archType"
    )
    foot_width_type: Literal["NARROW", "NORMAL", "WIDE", "UNKNOWN"] = Field(
        default="UNKNOWN", alias="footWidthType"
    )
    pressure_balance_type: Literal[
        "LEFT_DOMINANT", "BALANCED", "RIGHT_DOMINANT", "UNKNOWN"
    ] = Field(default="UNKNOWN", alias="pressureBalanceType")
    measured_left_foot_size_mm: float | None = Field(
        default=None, alias="measuredLeftFootSizeMm", ge=0
    )
    measured_right_foot_size_mm: float | None = Field(
        default=None, alias="measuredRightFootSizeMm", ge=0
    )
    left_foot_width_mm: float | None = Field(
        default=None, alias="leftFootWidthMm", ge=0
    )
    right_foot_width_mm: float | None = Field(
        default=None, alias="rightFootWidthMm", ge=0
    )
    left_pressure_percent: float | None = Field(
        default=None, alias="leftPressurePercent", ge=0, le=100
    )
    right_pressure_percent: float | None = Field(
        default=None, alias="rightPressurePercent", ge=0, le=100
    )
    plantar_footprint_analysis_text: str | None = Field(
        default=None, alias="plantarFootprintAnalysisText", max_length=1000
    )

    @field_validator("plantar_footprint_analysis_text")
    @classmethod
    def compact_plantar_analysis_text(cls, value: str | None) -> str | None:
        compacted = " ".join((value or "").split())
        return compacted or None

    @model_validator(mode="after")
    def require_classified_evidence(self):
        has_classification = not all(
            value == "UNKNOWN"
            for value in (
                self.arch_type,
                self.foot_width_type,
                self.pressure_balance_type,
            )
        )
        has_left_pressure = self.left_pressure_percent is not None
        has_right_pressure = self.right_pressure_percent is not None
        if has_left_pressure != has_right_pressure:
            raise ValueError(
                "leftPressurePercent and rightPressurePercent must be supplied together."
            )
        has_pressure_pair = has_left_pressure and has_right_pressure
        if has_pressure_pair:
            total = self.left_pressure_percent + self.right_pressure_percent
            if not 99.0 <= total <= 101.0:
                raise ValueError("Foot pressure percentages must total approximately 100.")
        if not (
            has_classification
            or has_pressure_pair
            or self.plantar_footprint_analysis_text is not None
        ):
            raise ValueError(
                "At least one classified foot-type analysis result is required."
            )
        return self


class FootTypeTextGenerationRequest(BaseModel):
    """Authoritative completed-session facts supplied by Feetfit_Server."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    measurement_status: Literal["COMPLETED"] = Field(alias="measurementStatus")
    facts_hash: str = Field(alias="factsHash", pattern=r"^[0-9a-f]{64}$")
    analysis: FootTypeAnalysisContext


class FootTypeTextGenerationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    facts_hash: str = Field(alias="factsHash", pattern=r"^[0-9a-f]{64}$")
    type_text: str = Field(alias="typeText", min_length=1, max_length=500)
    evidence_id: str = Field(alias="evidenceId", min_length=1, max_length=80)
    source: Literal["OPENAI", "FALLBACK"]

    @field_validator("type_text")
    @classmethod
    def reject_measurement_session_preface(cls, value: str) -> str:
        if value.lstrip().startswith("이번 측정에서는"):
            raise ValueError("typeText must not start with a measurement-session preface.")
        return value
