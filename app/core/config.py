from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Feetfit AI 무좀 분석 API"
    app_version: str = "0.1.0"
    weights_dir: Path = Field(default=PROJECT_ROOT / "weights")
    sam_source_dir: Path = Field(default=Path(r"D:\tinea pedis\sam\segment-anything"))
    tinea_report_endpoint: str = "http://35.94.253.151/api/reports/tina-pedis"
    hallux_valgus_report_endpoint: str = "http://35.94.253.151/api/reports/hallux-valgus"
    shoe_recommendation_endpoint: str = Field(
        default="http://127.0.0.1:8080/api/shoes/recommendations",
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_RECOMMENDATION_ENDPOINT",
            "SHOE_RECOMMENDATION_ENDPOINT",
        ),
    )
    shoe_summary_save_endpoint_template: str = Field(
        default="http://127.0.0.1:8080/api/shoes/{shoe_id}/summaries",
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_SUMMARY_SAVE_ENDPOINT_TEMPLATE",
            "SHOE_SUMMARY_SAVE_ENDPOINT_TEMPLATE",
        ),
    )
    shoe_recommendation_context_endpoint: str = Field(
        default="http://127.0.0.1:8080/api/internal/shoe-analysis/recommendation-context",
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_RECOMMENDATION_CONTEXT_ENDPOINT",
            "SHOE_RECOMMENDATION_CONTEXT_ENDPOINT",
        ),
    )
    shoe_summary_context_endpoint_template: str = Field(
        default=(
            "http://127.0.0.1:8080/api/internal/shoe-analysis/shoes/"
            "{shoe_id}/recommendation-summary-context"
        ),
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_SUMMARY_CONTEXT_ENDPOINT_TEMPLATE",
            "SHOE_SUMMARY_CONTEXT_ENDPOINT_TEMPLATE",
        ),
    )
    shoe_characteristics_endpoint_template: str = Field(
        default="http://127.0.0.1:8080/api/shoes/{shoe_id}/characteristics",
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_CHARACTERISTICS_ENDPOINT_TEMPLATE",
            "SHOE_CHARACTERISTICS_ENDPOINT_TEMPLATE",
        ),
    )
    shoe_recommendation_context_page_size: int = Field(default=100, ge=1, le=200)
    feetfit_server_internal_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_INTERNAL_API_KEY",
            "INTERNAL_API_KEY",
        ),
    )
    report_proxy_timeout_seconds: float = Field(default=60.0, gt=0, le=3600)
    feetfit_server_callback_timeout_seconds: float = Field(default=900.0, ge=900.0)

    openai_api_key: str | None = Field(default=None, repr=False)
    openai_report_model: str = "gpt-4.1-mini"
    openai_report_timeout_seconds: float = 20.0
    openai_report_text_enabled: bool = True
    openai_foot_type_text_enabled: bool = True
    foot_type_pressure_balance_tolerance_percent: float = Field(
        default=5.0, ge=0, le=25
    )
    openai_report_include_images: bool = True

    shoe_db_url: str = ""
    shoe_db_username: str = ""
    shoe_db_password: str = ""

    shoe_embedding_model_name: str = "BAAI/bge-m3"
    shoe_embedding_device: str = "auto"
    shoe_review_embedding_cache_path: Path = Field(default=PROJECT_ROOT / ".cache" / "shoe_review_embeddings.npz")
    shoe_release_embedding_model_after_batch: bool = True
    shoe_recommendation_batch_lock_timeout_seconds: float = Field(default=1.0, ge=0, le=30)
    shoe_max_candidate_reviews_per_reason: int = Field(default=40, ge=1, le=200)
    shoe_reviews_per_reason: int = Field(default=3, ge=1, le=3)
    shoe_review_semantic_min_score: float = Field(default=0.42, ge=-1, le=1)

    # Phase D scoring policy. These are intentionally configurable and remain
    # TEMPORARY_HEURISTIC / NOT_CLINICALLY_VALIDATED until a validated policy replaces them.
    shoe_risk_low_min_score: float = Field(default=75.0, ge=0, le=100)
    shoe_risk_medium_min_score: float = Field(default=50.0, ge=0, le=100)
    shoe_forefoot_area_weight: float = Field(default=0.40, gt=0)
    shoe_heel_area_weight: float = Field(default=0.30, gt=0)
    shoe_insole_area_weight: float = Field(default=0.30, gt=0)
    shoe_forefoot_width_ratio_neutral: float = Field(default=0.36, gt=0)
    shoe_forefoot_width_ratio_high: float = Field(default=0.43, gt=0)
    shoe_hallux_angle_neutral_degree: float = Field(default=10.0, ge=0)
    shoe_hallux_angle_high_degree: float = Field(default=30.0, gt=0)
    shoe_forefoot_width_allowance_mm: float = Field(default=5.0, ge=0)
    shoe_forefoot_width_extra_allowance_mm: float = Field(default=4.0, ge=0)
    shoe_forefoot_width_excess_free_mm: float = Field(default=8.0, ge=0)
    shoe_forefoot_width_shortfall_penalty_per_mm: float = Field(default=9.0, ge=0)
    shoe_forefoot_width_excess_penalty_per_mm: float = Field(default=2.0, ge=0)
    shoe_pressure_imbalance_neutral_percent: float = Field(default=5.0, ge=0)
    shoe_pressure_imbalance_high_percent: float = Field(default=20.0, gt=0)
    shoe_balance_neutral_score: float = Field(default=85.0, ge=0, le=100)
    shoe_balance_high_risk_score: float = Field(default=50.0, ge=0, le=100)
    shoe_rearfoot_pressure_neutral_percent: float = Field(default=50.0, ge=0, le=100)
    shoe_rearfoot_pressure_high_percent: float = Field(default=75.0, ge=0, le=100)
    shoe_forefoot_pressure_neutral_percent: float = Field(default=50.0, ge=0, le=100)
    shoe_forefoot_pressure_high_percent: float = Field(default=75.0, ge=0, le=100)
    shoe_skin_safety_neutral_score: float = Field(default=85.0, ge=0, le=100)
    shoe_skin_safety_high_risk_score: float = Field(default=45.0, ge=0, le=100)
    shoe_humidity_neutral_percent: float = Field(default=50.0, ge=0, le=100)
    shoe_humidity_high_percent: float = Field(default=75.0, ge=0, le=100)
    shoe_fungal_safety_neutral_score: float = Field(default=85.0, ge=0, le=100)
    shoe_fungal_safety_high_risk_score: float = Field(default=45.0, ge=0, le=100)
    shoe_metric_target_tolerance: float = Field(default=0.75, gt=0, le=1)
    shoe_forefoot_width_component_weight: float = Field(default=0.55, ge=0)
    shoe_forefoot_toebox_component_weight: float = Field(default=0.45, ge=0)
    shoe_heel_hold_component_weight: float = Field(default=0.40, ge=0)
    shoe_heel_shock_component_weight: float = Field(default=0.35, ge=0)
    shoe_heel_energy_component_weight: float = Field(default=0.15, ge=0)
    shoe_heel_cushion_component_weight: float = Field(default=0.10, ge=0)
    shoe_insole_breathability_component_weight: float = Field(default=0.45, ge=0)
    shoe_insole_cushion_component_weight: float = Field(default=0.30, ge=0)
    shoe_insole_shock_component_weight: float = Field(default=0.25, ge=0)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_temperature: float = Field(default=0.0, ge=0, le=2)
    ollama_request_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    ollama_num_gpu: int = Field(default=-1, ge=-1)
    ollama_cpu_fallback_enabled: bool = True
    ollama_max_concurrency: int = Field(default=1, ge=1, le=4)
    ollama_queue_timeout_seconds: float = Field(default=5.0, gt=0, le=120)

    max_upload_size_bytes: int = 10 * 1024 * 1024
    combined_image_max_height: int = 1600
    combined_image_gap_pixels: int = 16

    analysis_device: str = "auto"
    foot_confidence: float = 0.25
    foot_min_component_area: int = 2000
    foot_iou_duplicate_threshold: float = 0.6
    fungal_threshold: float | None = 0.78
    inflammation_threshold: float | None = None
    tinea_sliding_window_enabled: bool = False
    tinea_sliding_window_tile_size: int = 768
    tinea_sliding_window_overlap: float = 0.3
    tinea_sliding_window_padding: int = 32
    tinea_sliding_window_max_tiles: int = 12
    tinea_preprocess_enhance_enabled: bool = True
    tinea_preprocess_contrast_gain: float = 1.05
    tinea_preprocess_clahe_clip_limit: float = 1.6
    tinea_preprocess_red_saturation_gain: float = 1.10
    tinea_preprocess_red_value_gain: float = 1.02
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
    photo_cutout_background: bool = True
    photo_cutout_padding: int = 28

    hallux_model_source_dir: Path = Field(default=Path(r"D:\Hallux valgus"))
    hallux_landmark_weights: Path = Field(
        default=Path(
            r"D:\Hallux valgus\runs\real_foot_landmark_swin\teacher_student_top2_angle_from_swin\weights\best_top2_angle.pt"
        )
    )
    hallux_img_size: int = 384
    hallux_seg_conf: float = 0.25
    hallux_crop_padding: int = 28
    hallux_visual_lower_segment_ratio: float = 0.85
    hallux_visual_lower_segment_foot_ratio: float = 0.24

    # Single-photo ArUco measurement pipeline.  The implementation is loaded
    # from the separately cloned calibration repository so its validated
    # geometry code and camera assets stay the single source of truth.
    aruco_source_dir: Path = Field(default=Path(r"D:\ArUco-marker-code"))
    aruco_camera_calibration: Path | None = None
    aruco_foot_segmentation_weights: Path | None = None
    aruco_dictionary: str = "DICT_4X4_50"
    aruco_expected_image_width: int = 1280
    aruco_expected_image_height: int = 720
    aruco_allow_calibration_resize: bool = False
    aruco_undistort_balance: float = 0.0
    aruco_marker_size_mm: float = 20.0
    aruco_marker_row_spacing_mm: float = 171.0
    aruco_marker_column_spacing_mm: float = 140.0
    aruco_fixed_offset_mm: float = 113.0
    aruco_visible_length_scale: float = 1.0
    aruco_reference_edge: str = "outer"
    aruco_yolo_confidence: float = 0.25
    aruco_yolo_mask_threshold: float = 0.5
    aruco_yolo_image_size: int = 1280
    aruco_yolo_device: str = ""
    aruco_toe_refinement: str = "auto"
    aruco_toe_max_extension_mm: float = 2.0
    aruco_forefoot_start_fraction: float = 0.32
    aruco_forefoot_end_fraction: float = 0.38
    aruco_mtp_target_fraction: float = 0.35
    aruco_ball_slice_band_mm: float = 5.0
    aruco_ball_slice_step_mm: float = 0.5
    aruco_crop_padding_mm: float = 8.0
    # ArUco's image_left/image_right names are board bays, not anatomy.  Set
    # which anatomical side is placed in the canonical image-left bay.
    aruco_image_left_anatomical_side: str = "left"

    @model_validator(mode="after")
    def validate_shoe_fit_policy(self):
        if self.shoe_risk_low_min_score <= self.shoe_risk_medium_min_score:
            raise ValueError("shoe risk LOW threshold must be greater than MEDIUM threshold")
        increasing_pairs = (
            (self.shoe_forefoot_width_ratio_neutral, self.shoe_forefoot_width_ratio_high),
            (self.shoe_hallux_angle_neutral_degree, self.shoe_hallux_angle_high_degree),
            (self.shoe_pressure_imbalance_neutral_percent, self.shoe_pressure_imbalance_high_percent),
            (self.shoe_rearfoot_pressure_neutral_percent, self.shoe_rearfoot_pressure_high_percent),
            (self.shoe_forefoot_pressure_neutral_percent, self.shoe_forefoot_pressure_high_percent),
            (self.shoe_humidity_neutral_percent, self.shoe_humidity_high_percent),
        )
        if any(high <= neutral for neutral, high in increasing_pairs):
            raise ValueError("shoe fit high thresholds must be greater than neutral thresholds")
        decreasing_pairs = (
            (self.shoe_balance_neutral_score, self.shoe_balance_high_risk_score),
            (self.shoe_skin_safety_neutral_score, self.shoe_skin_safety_high_risk_score),
            (self.shoe_fungal_safety_neutral_score, self.shoe_fungal_safety_high_risk_score),
        )
        if any(high_risk >= neutral for neutral, high_risk in decreasing_pairs):
            raise ValueError("shoe safety high-risk thresholds must be below neutral thresholds")
        weight_groups = (
            (
                self.shoe_forefoot_area_weight,
                self.shoe_heel_area_weight,
                self.shoe_insole_area_weight,
            ),
            (
                self.shoe_forefoot_width_component_weight,
                self.shoe_forefoot_toebox_component_weight,
            ),
            (
                self.shoe_heel_hold_component_weight,
                self.shoe_heel_shock_component_weight,
                self.shoe_heel_energy_component_weight,
                self.shoe_heel_cushion_component_weight,
            ),
            (
                self.shoe_insole_breathability_component_weight,
                self.shoe_insole_cushion_component_weight,
                self.shoe_insole_shock_component_weight,
            ),
        )
        if any(sum(group) <= 0 for group in weight_groups):
            raise ValueError("each shoe fit weight group must have a positive sum")
        return self


settings = Settings()
