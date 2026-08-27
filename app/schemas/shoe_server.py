from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReasonType = Literal["FOREFOOT", "HEEL", "INSOLE"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
CharacteristicLevel = Literal["LOW", "MEDIUM", "HIGH"]
ReviewSource = Literal["MUSINSA"]
CanonicalShoeCharacteristic = Literal[
    "CUSHION",
    "SHOCK_ABSORPTION",
    "ENERGY_RETURN",
    "WIDTH_SPACE",
    "TOEBOX_SPACE",
    "HEEL_HOLD",
    "BREATHABILITY",
]

ResultT = TypeVar("ResultT")


class ServerApiResponse(BaseModel, Generic[ResultT]):
    """Feetfit_Server의 공통 ApiResponse<T> 응답."""

    model_config = ConfigDict(populate_by_name=True)

    is_success: bool = Field(alias="isSuccess")
    code: str
    message: str
    result: ResultT | None = None


class ServerShoeReview(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(alias="reviewId", ge=1)
    rating: float
    review_text: str = Field(alias="reviewText", min_length=1)
    source: ReviewSource
    collected_at: datetime | None = Field(default=None, alias="collectedAt")


class ServerShoeRawMetric(BaseModel):
    """RunRepeat의 원문 의미와 측정 방법을 잃지 않는 raw metric 계약."""

    model_config = ConfigDict(populate_by_name=True)

    metric_id: int = Field(alias="metricId", ge=1)
    canonical_characteristic: CanonicalShoeCharacteristic = Field(alias="canonicalCharacteristic")
    source_metric_name: str = Field(alias="sourceMetricName")
    value: Decimal | None = None
    average_value: Decimal | None = Field(default=None, alias="averageValue")
    source_min_value: Decimal | None = Field(default=None, alias="sourceMinValue")
    source_max_value: Decimal | None = Field(default=None, alias="sourceMaxValue")
    unit: str | None = None
    tested_size: str | None = Field(default=None, alias="testedSize")
    method_name: str | None = Field(default=None, alias="methodName")
    method_version: str | None = Field(default=None, alias="methodVersion")
    location: str | None = None
    variant: str | None = None
    comparison_sample_count: int | None = Field(default=None, alias="comparisonSampleCount", ge=0)
    comparison_cohort: str | None = Field(default=None, alias="comparisonCohort")
    raw_value_text: str | None = Field(default=None, alias="rawValueText")


class ServerShoeLabMeasurement(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    measurement_id: int = Field(alias="measurementId", ge=1)
    source: str
    source_url: str = Field(alias="sourceUrl")
    source_brand_name: str | None = Field(default=None, alias="sourceBrandName")
    source_shoe_name: str | None = Field(default=None, alias="sourceShoeName")
    source_model_code: str | None = Field(default=None, alias="sourceModelCode")
    tested_size: str | None = Field(default=None, alias="testedSize")
    captured_at: datetime | None = Field(default=None, alias="capturedAt")
    parser_version: str | None = Field(default=None, alias="parserVersion")
    internal_length_mm: float | None = Field(default=None, alias="internalLengthMm")
    width_mm: float | None = Field(default=None, alias="widthMm")
    toebox_width_mm: float | None = Field(default=None, alias="toeboxWidthMm")
    toebox_height_mm: float | None = Field(default=None, alias="toeboxHeightMm")
    insole_thickness_mm: float | None = Field(default=None, alias="insoleThicknessMm")
    heel_stack_mm: float | None = Field(default=None, alias="heelStackMm")
    forefoot_stack_mm: float | None = Field(default=None, alias="forefootStackMm")
    raw_metrics: list[ServerShoeRawMetric] = Field(alias="rawMetrics")


class ServerShoe(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(alias="shoeId", ge=1)
    brand_name: str = Field(alias="brandName")
    shoe_name: str = Field(alias="shoeName")
    model_code: str = Field(alias="modelCode")
    musinsa_url: str = Field(alias="musinsaUrl")
    price: int | None = None
    image_url: str | None = Field(default=None, alias="imageUrl")
    overall_rating: float | None = Field(default=None, alias="overallRating")
    review_count: int = Field(alias="reviewCount", ge=0)
    reviews: list[ServerShoeReview]
    lab_measurements: list[ServerShoeLabMeasurement] = Field(alias="labMeasurements")


class ServerDailyFootAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    balance_score: float | None = Field(default=None, alias="balanceScore")
    left_pressure_percent: float | None = Field(default=None, alias="leftPressurePercent")
    right_pressure_percent: float | None = Field(default=None, alias="rightPressurePercent")
    measured_left_foot_size_mm: float | None = Field(default=None, alias="measuredLeftFootSizeMm")
    measured_right_foot_size_mm: float | None = Field(default=None, alias="measuredRightFootSizeMm")
    left_foot_width_mm: float | None = Field(default=None, alias="leftFootWidthMm")
    right_foot_width_mm: float | None = Field(default=None, alias="rightFootWidthMm")
    avg_temperature_celsius: float | None = Field(default=None, alias="avgTemperatureCelsius")
    avg_humidity_percent: float | None = Field(default=None, alias="avgHumidityPercent")
    type_text: str | None = Field(default=None, alias="typeText")


class ServerTinaPedisAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fungal_suspicion_safety_score: int = Field(alias="fungalSuspicionSafetyScore", ge=0, le=100)
    skin_reaction_safety_score: int = Field(alias="skinReactionSafetyScore", ge=0, le=100)


class ServerHalluxValgusAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    left_toe_angle_degree: float | None = Field(default=None, alias="leftToeAngleDegree")
    right_toe_angle_degree: float | None = Field(default=None, alias="rightToeAngleDegree")
    risk_score: float | None = Field(default=None, alias="riskScore")


class ServerStaticPressureAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    analysis_id: int = Field(alias="analysisId", ge=1)
    foot_side: Literal["LEFT", "RIGHT"] = Field(alias="footSide")
    left_pressure_ratio: float | None = Field(default=None, alias="leftPressureRatio")
    right_pressure_ratio: float | None = Field(default=None, alias="rightPressureRatio")
    forefoot_pressure_ratio: float | None = Field(default=None, alias="forefootPressureRatio")
    rearfoot_pressure_ratio: float | None = Field(default=None, alias="rearfootPressureRatio")
    center_of_pressure_x: float = Field(alias="centerOfPressureX")
    center_of_pressure_y: float = Field(alias="centerOfPressureY")
    balance_score: float = Field(alias="balanceScore")
    balance_status: Literal["NEED_IMPROVEMENT", "ATTENTION_NEEDED", "VERY_GOOD"] = Field(
        alias="balanceStatus"
    )
    analysis_text: str | None = Field(default=None, alias="analysisText")


class ServerPressureSensorReading(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reading_id: int = Field(alias="readingId", ge=1)
    foot_side: Literal["LEFT", "RIGHT"] = Field(alias="footSide")
    foot_region: str = Field(alias="footRegion", pattern=r"^PRESSURE_(?:[0-9]|[12][0-9]|3[01])$")
    sensor_index: int | None = Field(default=None, alias="sensorIndex")
    pressure_value: float | None = Field(default=None, alias="pressureValue")
    pressure_unit: float | None = Field(default=None, alias="pressureUnit")
    recorded_at: datetime | None = Field(default=None, alias="recordedAt")


class ServerUserFootState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    daily_foot_analysis: ServerDailyFootAnalysis | None = Field(default=None, alias="dailyFootAnalysis")
    tina_pedis_analysis: ServerTinaPedisAnalysis | None = Field(default=None, alias="tinaPedisAnalysis")
    hallux_valgus_analysis: ServerHalluxValgusAnalysis | None = Field(default=None, alias="halluxValgusAnalysis")
    # Phase A는 전달 계약만 보존한다. 세부 계산 매핑은 Phase D에서 확정한다.
    static_pressure_analyses: list[ServerStaticPressureAnalysis] = Field(alias="staticPressureAnalyses")
    pressure_sensor_readings: list[ServerPressureSensorReading] = Field(alias="pressureSensorReadings")


class ServerRecommendationContext(BaseModel):
    """모든 Server page를 합친, 한 measurement session의 추천 계산 입력."""

    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    user_id: int = Field(alias="userId", ge=1)
    measurement_status: Literal["COMPLETED"] = Field(alias="measurementStatus")
    foot_state: ServerUserFootState = Field(alias="footState")
    shoes: list[ServerShoe]


class ServerRecommendationContextPage(ServerRecommendationContext):
    """Feetfit_Server가 한 page씩 반환하는 실제 ApiResponse.result."""

    current_page: int = Field(alias="currentPage", ge=0)
    total_pages: int = Field(alias="totalPages", ge=0)
    total_elements: int = Field(alias="totalElements", ge=0)
    has_next: bool = Field(alias="hasNext")


class ServerEvidenceReview(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(alias="reviewId", ge=1)
    review_text: str = Field(alias="reviewText", min_length=1)
    source: ReviewSource


class ServerSavedReason(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reason_type: ReasonType = Field(alias="reasonType")
    title: str
    risk_level: RiskLevel = Field(alias="riskLevel")
    review_summary: str | None = Field(default=None, alias="reviewSummary")
    reviews: list[ServerEvidenceReview] = Field(max_length=3)


class ServerSavedRecommendation(BaseModel):
    """Ollama 문장 생성에 필요한, 이미 계산·저장된 사실만 담는 Server 응답."""

    model_config = ConfigDict(populate_by_name=True)

    measurement_session_id: int = Field(alias="measurementSessionId", ge=1)
    user_id: int = Field(alias="userId", ge=1)
    shoe_id: int = Field(alias="shoeId", ge=1)
    brand_name: str = Field(alias="brandName")
    shoe_name: str = Field(alias="shoeName")
    fit_score: float = Field(alias="fitScore")
    point_summary: str | None = Field(default=None, alias="pointSummary")
    analyzed_at: datetime = Field(alias="analyzedAt")
    reasons: list[ServerSavedReason] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_reason_types(self):
        reason_types = [reason.reason_type for reason in self.reasons]
        if set(reason_types) != {"FOREFOOT", "HEEL", "INSOLE"}:
            raise ValueError("reasons must contain FOREFOOT, HEEL, and INSOLE")
        return self


class ServerShoeCharacteristic(BaseModel):
    """Public RunRepeat characteristic returned by Feetfit_Server."""

    model_config = ConfigDict(populate_by_name=True)

    type: CanonicalShoeCharacteristic
    level: CharacteristicLevel | None = None
    value: Decimal | None = None
    average_value: Decimal | None = Field(default=None, alias="averageValue")
    min_value: Decimal | None = Field(default=None, alias="minValue")
    max_value: Decimal | None = Field(default=None, alias="maxValue")
    unit: str | None = None
    tested_size: str | None = Field(default=None, alias="testedSize")


class ServerShoeCharacteristics(BaseModel):
    """Intrinsic shoe facts used to ground the product-level fit summary."""

    model_config = ConfigDict(populate_by_name=True)

    shoe_id: int = Field(alias="shoeId", ge=1)
    summary: str
    characteristics: list[ServerShoeCharacteristic]

    @model_validator(mode="after")
    def validate_unique_characteristics(self):
        types = [item.type for item in self.characteristics]
        if len(types) != len(set(types)):
            raise ValueError("shoe characteristic types must be unique")
        return self
