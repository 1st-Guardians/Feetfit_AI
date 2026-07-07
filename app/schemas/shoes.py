from pydantic import BaseModel, ConfigDict, Field


class ShoeRecommendationTriggerRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)


class ShoeRecommendationReasonPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reason_type: str = Field(alias="reasonType")
    title: str
    risk_level: str = Field(alias="riskLevel")
    review_ids: list[int] = Field(alias="reviewIds")


class ShoeRecommendationItemPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    shoe_id: int = Field(alias="shoeId")
    fit_score: float = Field(alias="fitScore")
    reasons: list[ShoeRecommendationReasonPayload]


class ShoeRecommendationForwardRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId")
    recommendations: list[ShoeRecommendationItemPayload]


class ShoeSummaryTriggerRequest(BaseModel):
    """Feetfit_Server가 신발 상세 조회 시 pointSummary가 비어 있으면 호출하는 트리거 요청."""

    model_config = ConfigDict(populate_by_name=True)

    shoe_id: int = Field(alias="shoeId", ge=1)
    user_id: int = Field(alias="userId", ge=1)


class ShoeSummaryReasonPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reason_type: str = Field(alias="reasonType")
    review_summary: str = Field(alias="reviewSummary")


class ShoeSummaryForwardRequest(BaseModel):
    """생성된 pointSummary/reviewSummary를 Feetfit_Server의
    POST /api/shoes/{shoeId}/summaries로 저장 요청할 때 쓰는 바디."""

    model_config = ConfigDict(populate_by_name=True)

    point_summary: str = Field(alias="pointSummary")
    reasons: list[ShoeSummaryReasonPayload]
