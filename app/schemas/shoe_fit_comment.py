from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShoePointSummaryClaim(BaseModel):
    """Internal citation metadata; it is intentionally not forwarded to Server."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(alias="evidenceIds", max_length=8)

    @model_validator(mode="after")
    def validate_evidence_ids(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("point summary evidence ids must be unique")
        return self


class ShoeFitSummary(BaseModel):
    """Ollama LLM이 생성하는, 기존 pointSummary/reviewSummary 자리를 그대로 대체하는 응답 타입."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    point_summary: str = Field(alias="pointSummary", min_length=1, max_length=1200)
    point_summary_claims: list[ShoePointSummaryClaim] = Field(
        default_factory=list, alias="pointSummaryClaims", max_length=4
    )
    forefoot_summary: str = Field(alias="forefootSummary", min_length=1, max_length=600)
    heel_summary: str = Field(alias="heelSummary", min_length=1, max_length=600)
    insole_summary: str = Field(alias="insoleSummary", min_length=1, max_length=600)
    forefoot_review_ids: list[int] = Field(alias="forefootReviewIds", max_length=3)
    heel_review_ids: list[int] = Field(alias="heelReviewIds", max_length=3)
    insole_review_ids: list[int] = Field(alias="insoleReviewIds", max_length=3)

    @model_validator(mode="after")
    def validate_review_ids(self):
        for review_ids in (
            self.forefoot_review_ids,
            self.heel_review_ids,
            self.insole_review_ids,
        ):
            if any(review_id < 1 for review_id in review_ids):
                raise ValueError("selected review ids must be positive")
            if len(review_ids) != len(set(review_ids)):
                raise ValueError("selected review ids must be unique")
        return self
