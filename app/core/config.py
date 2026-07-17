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
    tinea_report_endpoint: str = "http://35.94.253.151/api/reports/tina-pedis"
    hallux_valgus_report_endpoint: str = "http://35.94.253.151/api/reports/hallux-valgus"
    report_proxy_timeout_seconds: float = 60.0
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_report_model: str = "gpt-4.1-mini"
    openai_report_timeout_seconds: float = 20.0
    openai_report_text_enabled: bool = True
    openai_report_include_images: bool = True
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


settings = Settings()
