from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ShoeFitSummary(BaseModel):
    """Ollama LLM이 생성하는, 기존 pointSummary/reviewSummary 자리를 그대로 대체하는 응답 타입."""

    model_config = ConfigDict(populate_by_name=True)

    point_summary: str = Field(alias="pointSummary")
    forefoot_summary: str = Field(alias="forefootSummary")
    heel_summary: str = Field(alias="heelSummary")
    insole_summary: str = Field(alias="insoleSummary")
