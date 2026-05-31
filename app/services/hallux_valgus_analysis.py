from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
import torch

from app.core.config import PROJECT_ROOT, settings
from app.core.weights import weights
from app.services.tinea_analysis import AnalysisError, load_image, predict_foot_mask


os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

FOOT_OUTLINE_COLOR = (165, 165, 165)
FOOT_FILL_COLOR = (255, 255, 255)
FOOT_OUTLINE_COLOR_RGBA = (165, 165, 165, 255)
FOOT_FILL_COLOR_RGBA = (255, 255, 255, 255)
KEYPOINT_COLOR_RGBA = (255, 120, 0, 255)
LINE_COLOR_RGBA = (255, 170, 80, 255)
LEFT_SIDE_VALUE = -1.0
RIGHT_SIDE_VALUE = 1.0


@dataclass(frozen=True)
class HalluxValgusAnalysisResult:
    analysis_png: bytes
    original_filename: str
    angle_degree: float
    analysis_text: str
    score: float
    metrics: dict


def ensure_hallux_source_path() -> None:
    source_dir = settings.hallux_model_source_dir
    if not source_dir.exists():
        raise AnalysisError(f"Hallux model source directory not found: {source_dir}")
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg not in {"cpu", "cuda"}:
        raise AnalysisError(f"Unsupported analysis device: {device_arg}")
    return device_arg


def load_hallux_model(weights_path: Path, device: str) -> torch.nn.Module:
    if not weights_path.exists():
        raise AnalysisError(f"Hallux landmark weights not found: {weights_path}")

    ensure_hallux_source_path()
    from real_foot_landmark_model.net import build_real_foot_model

    model = build_real_foot_model(
        backbone="swin",
        img_size=settings.hallux_img_size,
        pretrained=False,
        prior_hidden=32,
        coord_temperature=60.0,
        coord_window_radius=6,
        coord_mode="softargmax",
    ).to(device)
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def bbox_from_mask(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + padding + 1)
    y2 = min(h, int(ys.max()) + padding + 1)
    return x1, y1, x2, y2


def crop_foot(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox_from_mask(mask, settings.hallux_crop_padding)
    masked = image_bgr.copy()
    masked[mask == 0] = 0
    return masked[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def side_value(foot_side: str) -> float:
    return LEFT_SIDE_VALUE if foot_side == "left" else RIGHT_SIDE_VALUE


def hallux_angle(points: np.ndarray) -> float:
    hallux = points[0] - points[1]
    met = points[1] - points[2]
    norm = float(np.linalg.norm(hallux) * np.linalg.norm(met))
    if norm <= 1e-8:
        return float("nan")
    cos_v = float(np.clip(np.dot(hallux, met) / norm, -1.0, 1.0))
    raw = float(np.degrees(np.arccos(cos_v)))
    return min(raw, 180.0 - raw)


def preprocess_crop(
    crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
    flip_for_model: bool,
) -> tuple[torch.Tensor, torch.Tensor, object]:
    ensure_hallux_source_path()
    from real_foot_landmark_model.common import KEYPOINT_NAMES
    from real_foot_landmark_model.data import MEAN, STD, build_priors_from_mask, letterbox_image, letterbox_mask

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    dummy = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32)
    canvas, _, info = letterbox_image(rgb, dummy, img_size=settings.hallux_img_size)
    mask_canvas = letterbox_mask(crop_mask.astype(np.uint8), info, img_size=settings.hallux_img_size)
    if flip_for_model:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        mask_canvas = np.ascontiguousarray(mask_canvas[:, ::-1])
    priors = build_priors_from_mask(mask_canvas)
    image = canvas.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image_t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    priors_t = torch.from_numpy(priors).unsqueeze(0).float()
    return image_t, priors_t, info


@torch.no_grad()
def predict_crop(
    model: torch.nn.Module,
    crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
    device: str,
    foot_side: str,
) -> tuple[np.ndarray, float, str, bool]:
    ensure_hallux_source_path()
    from real_foot_landmark_model.data import invert_letterbox_coords

    candidates = []
    for side_mode, flip_for_model in (("hallux-left", False), ("hallux-right", True)):
        image_t, priors_t, info = preprocess_crop(crop_bgr, crop_mask, flip_for_model=flip_for_model)
        side_t = torch.tensor([side_value(foot_side)], dtype=torch.float32, device=device)
        outputs = model(image_t.to(device), priors_t.to(device), side_t)
        coords_norm = outputs["coords"][0].detach().cpu().numpy()
        if flip_for_model:
            coords_norm[:, 0] = 1.0 - coords_norm[:, 0]
        coords_model_px = coords_norm * float(settings.hallux_img_size)
        coords_crop = invert_letterbox_coords(coords_model_px, info)
        heatmaps = torch.sigmoid(torch.nan_to_num(outputs["heatmap_logits"], nan=0.0, posinf=20.0, neginf=-20.0))
        score = float(heatmaps.amax(dim=(-1, -2)).mean().detach().cpu())
        if not np.isfinite(coords_crop).all():
            score = -1.0
        candidates.append((score, coords_crop, side_mode, flip_for_model))

    score, coords_crop, side_mode, flip_for_model = max(candidates, key=lambda item: item[0])
    return coords_crop.astype(np.float32), float(score), side_mode, bool(flip_for_model)


def resize_for_visual(mask: np.ndarray, points: np.ndarray, max_side: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return mask, points
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized_mask, points * scale


def clipped_visual_points(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    visual = points.copy()
    upper_len = float(np.linalg.norm(points[0] - points[1]))
    lower_len = float(np.linalg.norm(points[2] - points[1]))
    if upper_len <= 1e-6 or lower_len <= 1e-6:
        return visual

    ys = np.where(mask > 0)[0]
    foot_h = float(ys.max() - ys.min() + 1) if ys.size else float(mask.shape[0])
    max_by_upper = upper_len * float(settings.hallux_visual_lower_segment_ratio)
    max_by_foot = foot_h * float(settings.hallux_visual_lower_segment_foot_ratio)
    max_lower_len = max(24.0, min(max_by_upper, max_by_foot))
    if lower_len > max_lower_len:
        direction = (points[2] - points[1]) / lower_len
        visual[2] = points[1] + direction * max_lower_len
    return visual


def render_analysis_image(mask: np.ndarray, points: np.ndarray) -> bytes:
    mask, points = resize_for_visual(mask, points)
    h, w = mask.shape[:2]
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    foot_region = mask > 0
    canvas[foot_region] = FOOT_FILL_COLOR_RGBA

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        thickness = max(3, int(round(max(h, w) / 360)))
        cv2.drawContours(canvas, contours, -1, FOOT_OUTLINE_COLOR_RGBA, thickness=thickness, lineType=cv2.LINE_AA)

    visual_points = clipped_visual_points(points, mask)
    line_thickness = max(3, int(round(max(h, w) / 320)))
    radius = max(7, int(round(max(h, w) / 95)))
    p0, p1, p2 = [tuple(int(round(v)) for v in pt) for pt in visual_points]
    cv2.line(canvas, p0, p1, LINE_COLOR_RGBA, line_thickness, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, LINE_COLOR_RGBA, line_thickness, cv2.LINE_AA)
    for point in (p0, p1, p2):
        cv2.circle(canvas, point, radius, KEYPOINT_COLOR_RGBA, -1, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise AnalysisError("Failed to encode hallux valgus analysis image.")
    return encoded.tobytes()


def angle_stage(angle: float) -> str:
    if angle < 15.0:
        return "정상 또는 변형이 뚜렷하지 않은 범위(15° 미만)"
    if angle < 20.0:
        return "경미한 변형 범위(15~20°)"
    if angle < 40.0:
        return "변형이 진행된 범위(20~40°)"
    return "심한 변형 범위(40° 이상)"


def side_analysis_text(foot_side: str, angle: float) -> str:
    side_label = "왼발" if foot_side == "left" else "오른발"
    return (
        f"{side_label} 엄지발가락이 두 번째 발가락 쪽으로 기울어진 각도(HVA)가 "
        f"{angle:.1f}°로 측정되었습니다. {angle_stage(angle)}에 해당합니다."
    )


def risk_score_from_angles(left_angle: float, right_angle: float) -> float:
    max_angle = max(left_angle, right_angle)
    min_angle = min(left_angle, right_angle)
    if max_angle < 15.0:
        score = 15.0 + (max_angle / 15.0) * 25.0
    elif max_angle < 20.0:
        score = 45.0 + ((max_angle - 15.0) / 5.0) * 20.0
    elif max_angle < 40.0:
        score = 70.0 + ((max_angle - 20.0) / 20.0) * 20.0
    else:
        score = 90.0 + min(10.0, (max_angle - 40.0) * 0.5)
    if min_angle >= 15.0:
        score += 4.0
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def score_analysis_text(left_angle: float, right_angle: float) -> str:
    left_abnormal = left_angle >= 15.0
    right_abnormal = right_angle >= 15.0
    if left_abnormal and right_abnormal:
        return "양쪽 발 모두 무지외반 진행 가능성이 있어 관리가 필요합니다."
    if left_abnormal:
        return "왼발 중심으로 무지외반 진행 가능성이 있어 관리가 필요합니다."
    if right_abnormal:
        return "오른발 중심으로 무지외반 진행 가능성이 있어 관리가 필요합니다."
    return "현재 이미지 기준으로는 무지외반 위험 신호가 크게 두드러지지 않습니다."


class HalluxValgusAnalyzer:
    def __init__(self) -> None:
        self.device = resolve_device(settings.analysis_device)
        self.model = load_hallux_model(weights.hallux_landmark, self.device)
        if not weights.foot_segmentation.exists():
            raise AnalysisError(f"Foot segmentation model not found: {weights.foot_segmentation}")
        from ultralytics import YOLO

        self.foot_model = YOLO(str(weights.foot_segmentation))

    def analyze(self, image_bytes: bytes, filename: str | None, foot_side: str) -> HalluxValgusAnalysisResult:
        _, image_bgr = load_image(image_bytes)
        foot_mask, foot_found = predict_foot_mask(
            foot_model=self.foot_model,
            image_bgr=image_bgr,
            conf=settings.hallux_seg_conf,
            min_component_area=settings.foot_min_component_area,
            iou_dup_threshold=settings.foot_iou_duplicate_threshold,
        )
        crop_bgr, crop_mask, bbox = crop_foot(image_bgr, foot_mask)
        coords_crop, score, side_mode, flip_for_model = predict_crop(
            self.model,
            crop_bgr,
            crop_mask,
            self.device,
            foot_side=foot_side,
        )
        angle = hallux_angle(coords_crop)
        if not np.isfinite(angle):
            raise AnalysisError("Hallux angle prediction failed.")

        analysis_png = render_analysis_image(crop_mask, coords_crop)
        return HalluxValgusAnalysisResult(
            analysis_png=analysis_png,
            original_filename=Path(filename or "foot.jpg").name,
            angle_degree=round(float(angle), 1),
            analysis_text=side_analysis_text(foot_side, float(angle)),
            score=score,
            metrics={
                "foot_outline_found": bool(foot_found),
                "bbox": bbox,
                "score": score,
                "side_mode": side_mode,
                "flip_for_model": flip_for_model,
                "hallux_tip": coords_crop[0].tolist(),
                "met_head": coords_crop[1].tolist(),
                "met_base": coords_crop[2].tolist(),
                "visualization": "segmented_foot_silhouette_with_blue_keypoints_and_clipped_lower_segment",
            },
        )


_hallux_analyzer: HalluxValgusAnalyzer | None = None
_hallux_analyzer_lock = threading.Lock()


def get_hallux_valgus_analyzer() -> HalluxValgusAnalyzer:
    global _hallux_analyzer
    if _hallux_analyzer is not None:
        return _hallux_analyzer

    with _hallux_analyzer_lock:
        if _hallux_analyzer is None:
            _hallux_analyzer = HalluxValgusAnalyzer()
    return _hallux_analyzer


def analyze_hallux_valgus_image(image_bytes: bytes, filename: str | None, foot_side: str) -> HalluxValgusAnalysisResult:
    if foot_side not in {"left", "right"}:
        raise AnalysisError(f"Unsupported foot side: {foot_side}")
    analyzer = get_hallux_valgus_analyzer()
    return analyzer.analyze(image_bytes, filename, foot_side)
