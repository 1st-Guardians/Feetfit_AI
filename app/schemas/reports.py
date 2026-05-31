from pydantic import BaseModel, ConfigDict, Field


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
