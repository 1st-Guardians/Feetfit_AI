from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Feetfit AI 무좀 분석 API"
    app_version: str = "0.1.0"
    weights_dir: Path = Field(default=PROJECT_ROOT / "weights")
    sam_source_dir: Path = Field(default=Path(r"D:\tinea pedis\sam\segment-anything"))
    tinea_report_endpoint: str = "http://54.184.58.176/api/reports/tina-pedis"
    report_proxy_timeout_seconds: float = 60.0
    max_upload_size_bytes: int = 10 * 1024 * 1024
    combined_image_max_height: int = 1600
    combined_image_gap_pixels: int = 16

    analysis_device: str = "auto"
    foot_confidence: float = 0.25
    foot_min_component_area: int = 2000
    foot_iou_duplicate_threshold: float = 0.6
    fungal_threshold: float | None = 0.78
    inflammation_threshold: float | None = None
    score_hard_weight: float = 0.7
    score_sensitivity: float = 7.0

    suspicion_line_thickness: int = 3
    min_suspicion_area: int = 18
    circle_scale: float = 1.08
    min_circle_radius: int = 5
    max_circle_radius: int = 42
    max_dot_area_for_suspicion_map: int = 450
    inner_confidence_percentile: float = 78.0
    inner_min_area_ratio: float = 0.18


settings = Settings()
