from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import re
import sys
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import PROJECT_ROOT, settings
from app.core.weights import weights


os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".ultralytics").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO  # noqa: E402

NUM_CLASSES = 3
PHOTO_FUNGAL_COLOR = (255, 0, 0)  # BGR blue
PHOTO_INFLAMMATION_COLOR = (0, 0, 255)  # BGR red
MAP_FUNGAL_COLOR = (255, 220, 155)  # BGR pastel sky blue
MAP_INFLAMMATION_COLOR = (195, 170, 255)  # BGR pastel pink
FOOT_OUTLINE_COLOR = (165, 165, 165)


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class TineaAnalysisResult:
    suspicion_map_jpeg: bytes
    photo_overlay_jpeg: bytes
    original_filename: str
    fungal_safety_score: int
    skin_reaction_safety_score: int
    metrics: dict


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SamSemanticHead(nn.Module):
    def __init__(self, in_channels: int = 256, hidden_channels: int = 128, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.proj = DecoderBlock(in_channels, hidden_channels)
        self.up1 = DecoderBlock(hidden_channels, hidden_channels // 2)
        self.up2 = DecoderBlock(hidden_channels // 2, hidden_channels // 4)
        self.out = nn.Conv2d(hidden_channels // 4, num_classes, kernel_size=1)

    def forward(self, image_embeddings: torch.Tensor, output_size: int) -> torch.Tensor:
        x = self.proj(image_embeddings)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        logits = self.out(x)
        return F.interpolate(logits, size=(output_size, output_size), mode="bilinear", align_corners=False)


class SamMultiClassSegModel(nn.Module):
    def __init__(self, sam: nn.Module, output_size: int, head_channels: int) -> None:
        super().__init__()
        self.sam = sam
        self.output_size = output_size
        self.head = SamSemanticHead(hidden_channels=head_channels)

        for parameter in self.sam.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.sam.preprocess(images)
        embeddings = self.sam.image_encoder(images)
        return self.head(embeddings, self.output_size)


def sanitize_filename(filename: str | None) -> str:
    if not filename:
        return "foot.jpg"

    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "foot.jpg"


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg not in {"cpu", "cuda"}:
        raise AnalysisError(f"Unsupported analysis device: {device_arg}")
    return device_arg


def load_image(upload_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    try:
        image = Image.open(BytesIO(upload_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise AnalysisError("Uploaded file is not a readable image.") from exc

    image_rgb = np.asarray(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image, image_bgr


def resize_to_square(image: Image.Image, target_size: int) -> Image.Image:
    return image.resize((target_size, target_size), Image.Resampling.BILINEAR)


def load_tinea_model(model_path: Path, device: str) -> tuple[SamMultiClassSegModel, int, float, float]:
    if not model_path.exists():
        raise AnalysisError(f"Tinea model not found: {model_path}")
    if not weights.sam_checkpoint.exists():
        raise AnalysisError(f"SAM checkpoint not found: {weights.sam_checkpoint}")
    if not settings.sam_source_dir.exists():
        raise AnalysisError(f"segment-anything source directory not found: {settings.sam_source_dir}")

    if str(settings.sam_source_dir) not in sys.path:
        sys.path.insert(0, str(settings.sam_source_dir))

    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise AnalysisError(f"Could not import segment_anything from {settings.sam_source_dir}") from exc

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    ckpt_args = checkpoint.get("args", {}) or {}
    model_type = ckpt_args.get("model_type", "vit_b")
    output_size = int(ckpt_args.get("output_size", 384))
    head_channels = int(ckpt_args.get("head_channels", 128))
    fungal_threshold = (
        settings.fungal_threshold
        if settings.fungal_threshold is not None
        else float(ckpt_args.get("fg_threshold", 0.86))
    )
    inflammation_threshold = (
        settings.inflammation_threshold
        if settings.inflammation_threshold is not None
        else float(ckpt_args.get("inflammation_threshold", 0.88))
    )

    sam = sam_model_registry[model_type](checkpoint=str(weights.sam_checkpoint))
    model = SamMultiClassSegModel(sam=sam, output_size=output_size, head_channels=head_channels)
    model.head.load_state_dict(checkpoint["head_state_dict"])
    if "image_encoder_state_dict" in checkpoint:
        model.sam.image_encoder.load_state_dict(checkpoint["image_encoder_state_dict"])
    model.to(device)
    model.eval()

    sam_input_size = int(model.sam.image_encoder.img_size)
    return model, sam_input_size, float(fungal_threshold), float(inflammation_threshold)


def image_to_sam_tensor(image: Image.Image, sam_input_size: int, device: str) -> torch.Tensor:
    sam_image = resize_to_square(image, sam_input_size)
    arr = np.asarray(sam_image, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def predict_tinea_probs(
    model: SamMultiClassSegModel,
    image: Image.Image,
    sam_input_size: int,
    device: str,
) -> np.ndarray:
    width, height = image.size
    image_tensor = image_to_sam_tensor(image, sam_input_size, device)
    with torch.inference_mode():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    return np.stack(
        [cv2.resize(probs[class_idx], (width, height), interpolation=cv2.INTER_LINEAR) for class_idx in range(probs.shape[0])],
        axis=0,
    )


def threshold_predictions(
    probs: np.ndarray,
    fungal_threshold: float,
    inflammation_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    fungal_prob = probs[1]
    inflammation_prob = probs[2]
    fungal_mask = fungal_prob >= fungal_threshold
    inflammation_mask = inflammation_prob >= inflammation_threshold

    overlap = fungal_mask & inflammation_mask
    if overlap.any():
        fungal_wins = fungal_prob >= inflammation_prob
        fungal_mask = fungal_mask & (~overlap | fungal_wins)
        inflammation_mask = inflammation_mask & (~overlap | ~fungal_wins)

    return fungal_mask, inflammation_mask


def masks_for_visualization(
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    foot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    foot_region = foot_mask > 0
    return fungal_mask & foot_region, inflammation_mask & foot_region


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def split_connected_components(binary_mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        comp_mask[labels == label_idx] = 1
        components.append(comp_mask)
    return components


def deduplicate_masks(mask_items: list[dict], iou_threshold: float) -> list[dict]:
    mask_items = sorted(mask_items, key=lambda x: x["score"], reverse=True)
    kept = []
    for item in mask_items:
        if not any(mask_iou(item["mask"], kept_item["mask"]) >= iou_threshold for kept_item in kept):
            kept.append(item)
    return kept


def predict_foot_mask(
    foot_model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    min_component_area: int,
    iou_dup_threshold: float,
) -> tuple[np.ndarray, bool]:
    h, w = image_bgr.shape[:2]
    fallback = np.ones((h, w), dtype=np.uint8)
    results = foot_model.predict(source=image_bgr, conf=conf, save=False, verbose=False)
    if not results:
        return fallback, False

    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return fallback, False

    raw_masks = result.masks.data.detach().cpu().numpy()
    if result.boxes is not None and result.boxes.conf is not None:
        scores = result.boxes.conf.detach().cpu().numpy()
    else:
        scores = np.ones(len(raw_masks), dtype=np.float32)

    components = []
    for mask_idx, raw_mask in enumerate(raw_masks):
        score = float(scores[mask_idx]) if mask_idx < len(scores) else 1.0
        mask_resized = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
        for comp in split_connected_components(binary_mask, min_component_area):
            components.append({"mask": comp, "score": score})

    final_components = deduplicate_masks(components, iou_dup_threshold)
    if not final_components:
        return fallback, False

    combined = np.zeros((h, w), dtype=np.uint8)
    for item in final_components:
        combined[item["mask"] > 0] = 1
    return combined, True


def score_from_burden(burden: float, sensitivity: float) -> float:
    score = 100.0 * float(np.exp(-sensitivity * max(0.0, burden)))
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def class_score_metrics(
    prob: np.ndarray,
    mask: np.ndarray,
    foot_mask: np.ndarray,
    threshold: float,
) -> dict:
    foot_region = foot_mask > 0
    foot_pixels = int(foot_region.sum())
    if foot_pixels == 0:
        return {
            "pixels": 0,
            "pixel_ratio": 0.0,
            "confidence_weighted_ratio": 0.0,
            "burden": 0.0,
            "score": 100.0,
        }

    hard_weight = float(np.clip(settings.score_hard_weight, 0.0, 1.0))
    class_region = mask & foot_region
    class_pixels = int(class_region.sum())
    pixel_ratio = class_pixels / foot_pixels

    denom = max(1e-6, 1.0 - threshold)
    confidence_excess = np.clip((prob - threshold) / denom, 0.0, 1.0)
    confidence_weighted_ratio = float(confidence_excess[foot_region].sum() / foot_pixels)
    burden = hard_weight * pixel_ratio + (1.0 - hard_weight) * confidence_weighted_ratio

    return {
        "pixels": class_pixels,
        "pixel_ratio": round(float(pixel_ratio), 6),
        "confidence_weighted_ratio": round(confidence_weighted_ratio, 6),
        "burden": round(float(burden), 6),
        "score": score_from_burden(burden, settings.score_sensitivity),
    }


def score_label(score: float) -> str:
    if score >= 90.0:
        return "healthy"
    if score >= 70.0:
        return "mild"
    if score >= 40.0:
        return "moderate"
    return "high_suspicion"


def compute_health_scores(
    probs: np.ndarray,
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    foot_mask: np.ndarray,
    foot_found: bool,
    fungal_threshold: float,
    inflammation_threshold: float,
) -> dict:
    fungal = class_score_metrics(probs[1], fungal_mask, foot_mask, fungal_threshold)
    inflammation = class_score_metrics(probs[2], inflammation_mask, foot_mask, inflammation_threshold)
    overall_score = round(min(fungal["score"], inflammation["score"]), 2)

    return {
        "foot_area_pixels": int((foot_mask > 0).sum()),
        "score_reliability": "normal" if foot_found else "low_full_image_fallback",
        "fungal_pixels_in_foot": fungal["pixels"],
        "fungal_pixel_ratio": fungal["pixel_ratio"],
        "fungal_confidence_weighted_ratio": fungal["confidence_weighted_ratio"],
        "fungal_burden": fungal["burden"],
        "fungal_score": fungal["score"],
        "fungal_score_label": score_label(fungal["score"]),
        "inflammation_pixels_in_foot": inflammation["pixels"],
        "inflammation_pixel_ratio": inflammation["pixel_ratio"],
        "inflammation_confidence_weighted_ratio": inflammation["confidence_weighted_ratio"],
        "inflammation_burden": inflammation["burden"],
        "inflammation_score": inflammation["score"],
        "inflammation_score_label": score_label(inflammation["score"]),
        "overall_health_score": overall_score,
        "overall_health_label": score_label(overall_score),
    }


def confidence_to_alpha(confidence: float, threshold: float, min_alpha: float, max_alpha: float) -> float:
    denom = max(1e-6, 1.0 - threshold)
    scaled = np.clip((confidence - threshold) / denom, 0.0, 1.0)
    return float(min_alpha + scaled * (max_alpha - min_alpha))


def blend_mask_by_confidence(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    color: tuple[int, int, int],
    min_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    if not mask.any():
        return image_bgr

    out = image_bgr.copy()
    denom = max(1e-6, 1.0 - threshold)
    alpha = min_alpha + np.clip((prob - threshold) / denom, 0.0, 1.0) * (max_alpha - min_alpha)
    color_arr = np.zeros_like(out, dtype=np.float32)
    color_arr[:, :] = np.array(color, dtype=np.float32)

    mask_3 = mask[:, :, None]
    blended = out.astype(np.float32) * (1.0 - alpha[:, :, None]) + color_arr * alpha[:, :, None]
    out[mask_3.repeat(3, axis=2)] = blended[mask_3.repeat(3, axis=2)].astype(np.uint8)
    return out


def render_photo_overlay(
    image_bgr: np.ndarray,
    probs: np.ndarray,
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    fungal_threshold: float,
    inflammation_threshold: float,
) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay = blend_mask_by_confidence(
        overlay,
        fungal_mask,
        probs[1],
        fungal_threshold,
        PHOTO_FUNGAL_COLOR,
        min_alpha=0.28,
        max_alpha=0.68,
    )
    overlay = blend_mask_by_confidence(
        overlay,
        inflammation_mask,
        probs[2],
        inflammation_threshold,
        PHOTO_INFLAMMATION_COLOR,
        min_alpha=0.26,
        max_alpha=0.64,
    )

    contour_thickness = max(3, settings.suspicion_line_thickness + 1)
    for mask, color in [
        (fungal_mask, PHOTO_FUNGAL_COLOR),
        (inflammation_mask, PHOTO_INFLAMMATION_COLOR),
    ]:
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, color, thickness=contour_thickness, lineType=cv2.LINE_AA)

    return overlay


def apply_cutout_background(
    image_bgr: np.ndarray,
    foot_mask: np.ndarray,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    mask = (foot_mask > 0).astype(np.float32)
    if mask.all():
        return image_bgr

    alpha = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0, sigmaY=1.0)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    background = np.full_like(image_bgr, background_color, dtype=np.uint8)
    cutout = image_bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return cutout.astype(np.uint8)


def circle_from_mask(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    circle_scale: float,
    min_radius: int,
    max_radius: int,
    layer: int,
    level: float,
) -> dict | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    values = prob[ys, xs]
    weights_for_center = np.clip(values - threshold, 0.0, None) + 1e-3
    cx = float(np.average(xs, weights=weights_for_center))
    cy = float(np.average(ys, weights=weights_for_center))
    area = int(len(xs))
    area_radius = np.sqrt(area / np.pi) * circle_scale
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    spread_radius = float(np.percentile(distances, 70)) * 0.96 if distances.size else 0.0
    radius = int(round(max(area_radius, spread_radius)))
    radius = int(np.clip(radius, min_radius, max_radius))

    return {
        "center": (int(round(cx)), int(round(cy))),
        "radius": radius,
        "confidence": float(np.percentile(values, 90)),
        "threshold": threshold,
        "level": float(level),
        "layer": layer,
        "area": area,
    }


def hierarchical_class_circles(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> list[dict]:
    if not mask.any():
        return []

    area = int(mask.sum())
    if area < settings.min_suspicion_area:
        return []
    if area <= settings.max_dot_area_for_suspicion_map:
        return []

    outer = circle_from_mask(
        mask=mask,
        prob=prob,
        threshold=threshold,
        circle_scale=settings.circle_scale,
        min_radius=settings.min_circle_radius,
        max_radius=settings.max_circle_radius,
        layer=0,
        level=threshold,
    )
    if outer is None:
        return []

    circles = [outer]
    inner_level = max(threshold, float(np.percentile(prob[mask], settings.inner_confidence_percentile)))
    if inner_level < threshold + 0.03:
        return circles

    inner_mask = mask & (prob >= inner_level)
    inner_area = int(inner_mask.sum())
    min_inner_area = max(settings.min_suspicion_area, int(round(area * settings.inner_min_area_ratio)))
    if inner_area < min_inner_area:
        return circles

    inner = circle_from_mask(
        mask=inner_mask,
        prob=prob,
        threshold=threshold,
        circle_scale=settings.circle_scale,
        min_radius=2,
        max_radius=max(2, int(round(outer["radius"] * 0.58))),
        layer=1,
        level=inner_level,
    )
    if inner is None:
        return circles

    outer_cx, outer_cy = outer["center"]
    inner_cx, inner_cy = inner["center"]
    center_distance = float(np.hypot(inner_cx - outer_cx, inner_cy - outer_cy))
    max_inner_radius = int(np.floor(outer["radius"] - center_distance - 1))
    if max_inner_radius < 2:
        inner["center"] = outer["center"]
        max_inner_radius = max(2, int(round(outer["radius"] * 0.52)))

    inner["radius"] = int(min(inner["radius"], max_inner_radius))
    if inner["radius"] >= 2:
        circles.append(inner)
    return circles


def blend_circle(
    canvas: np.ndarray,
    foot_mask: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    h, w = canvas.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    circle_mask[foot_mask == 0] = 0

    if not circle_mask.any():
        return

    color_img = np.zeros_like(canvas, dtype=np.uint8)
    color_img[:, :] = np.array(color, dtype=np.uint8)
    blended = cv2.addWeighted(canvas, 1.0 - alpha, color_img, alpha, 0.0)
    canvas[circle_mask > 0] = blended[circle_mask > 0]


def circle_contains_point(circle: dict, point: tuple[int, int], margin: float = 0.92) -> bool:
    cx, cy = circle["center"]
    px, py = point
    return float(np.hypot(px - cx, py - cy)) <= float(circle["radius"]) * margin


def evidence_dots_from_mask(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    existing_circles: list[dict],
    color: tuple[int, int, int],
) -> list[dict]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    dots = []

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component = labels == label_idx
        ys, xs = np.where(component)
        values = prob[ys, xs]
        weights = np.clip(values - threshold, 0.0, None) + 1e-3
        center = (int(round(float(np.average(xs, weights=weights)))), int(round(float(np.average(ys, weights=weights)))))

        if any(circle_contains_point(circle, center) for circle in existing_circles):
            continue

        radius = int(np.clip(round(np.sqrt(area / np.pi) * 1.4), 3, 7))
        confidence = float(np.percentile(values, 90))
        alpha = confidence_to_alpha(confidence, threshold, min_alpha=0.42, max_alpha=0.68)
        dots.append(
            {
                "center": center,
                "radius": radius,
                "color": color,
                "alpha": alpha,
                "area": area,
                "confidence": confidence,
            }
        )

    dots.sort(key=lambda item: (item["area"], item["confidence"]), reverse=True)
    return dots


def connected_components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label_idx
        components.append(component)

    components.sort(key=lambda component: int(component.sum()), reverse=True)
    return components


def circles_and_dots_from_overlay_mask(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    color: tuple[int, int, int],
) -> tuple[list[dict], list[dict]]:
    circles = []
    dots = []

    for component in connected_components_from_mask(mask):
        area = int(component.sum())
        if area < settings.min_suspicion_area:
            continue

        if area > settings.max_dot_area_for_suspicion_map and not circles:
            component_circles = hierarchical_class_circles(component, prob, threshold)
            for circle in component_circles:
                circle["color"] = color
                circles.append(circle)
            if component_circles:
                continue

        dots.extend(evidence_dots_from_mask(component, prob, threshold, circles, color))

    return circles, dots


def render_suspicion_map(
    image_shape: tuple[int, int],
    foot_mask: np.ndarray,
    probs: np.ndarray,
    fungal_mask: np.ndarray,
    inflammation_mask: np.ndarray,
    fungal_threshold: float,
    inflammation_threshold: float,
) -> tuple[np.ndarray, int, int]:
    h, w = image_shape
    canvas = np.full((h, w, 3), (250, 253, 255), dtype=np.uint8)
    canvas[foot_mask > 0] = (255, 255, 255)

    circles = []
    evidence_dots = []
    for mask, prob, threshold, color in [
        (fungal_mask, probs[1], fungal_threshold, MAP_FUNGAL_COLOR),
        (inflammation_mask, probs[2], inflammation_threshold, MAP_INFLAMMATION_COLOR),
    ]:
        draw_mask = mask & (foot_mask > 0)
        class_circles, class_dots = circles_and_dots_from_overlay_mask(draw_mask, prob, threshold, color)
        circles.extend(class_circles)
        evidence_dots.extend(class_dots)

    circles.sort(key=lambda item: (item["layer"], -item["radius"]))
    for circle in circles:
        if circle["layer"] == 0:
            alpha = confidence_to_alpha(circle["confidence"], circle["threshold"], min_alpha=0.16, max_alpha=0.34)
        else:
            alpha = confidence_to_alpha(circle["confidence"], circle["threshold"], min_alpha=0.32, max_alpha=0.58)
        blend_circle(
            canvas=canvas,
            foot_mask=foot_mask,
            center=circle["center"],
            radius=circle["radius"],
            color=circle["color"],
            alpha=alpha,
        )

    for dot in evidence_dots:
        blend_circle(
            canvas=canvas,
            foot_mask=foot_mask,
            center=dot["center"],
            radius=dot["radius"],
            color=dot["color"],
            alpha=dot["alpha"],
        )

    foot_contours, _ = cv2.findContours(foot_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if foot_contours:
        cv2.drawContours(
            canvas,
            foot_contours,
            -1,
            FOOT_OUTLINE_COLOR,
            thickness=settings.suspicion_line_thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas, len(circles), len(evidence_dots)


def encode_jpeg(image_bgr: np.ndarray, quality: int = 95) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise AnalysisError("Failed to encode analysis image.")
    return encoded.tobytes()


def decode_jpeg(image_jpeg: bytes) -> np.ndarray:
    arr = np.frombuffer(image_jpeg, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise AnalysisError("Failed to decode analysis image.")
    return image


def resize_to_height(image_bgr: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h == target_height:
        return image_bgr

    scale = target_height / h
    target_width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=interpolation)


def combine_jpeg_images_side_by_side(
    left_jpeg: bytes,
    right_jpeg: bytes,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    left = decode_jpeg(left_jpeg)
    right = decode_jpeg(right_jpeg)

    target_height = min(max(left.shape[0], right.shape[0]), settings.combined_image_max_height)
    left = resize_to_height(left, target_height)
    right = resize_to_height(right, target_height)

    gap = max(0, settings.combined_image_gap_pixels)
    combined_height = target_height
    combined_width = left.shape[1] + gap + right.shape[1]
    canvas = np.full((combined_height, combined_width, 3), background_color, dtype=np.uint8)
    canvas[:, : left.shape[1]] = left
    canvas[:, left.shape[1] + gap : left.shape[1] + gap + right.shape[1]] = right
    return encode_jpeg(canvas)


class TineaAnalyzer:
    def __init__(self) -> None:
        self.device = resolve_device(settings.analysis_device)
        self.tinea_model, self.sam_input_size, self.fungal_threshold, self.inflammation_threshold = load_tinea_model(
            weights.tinea_pedis_segmentation,
            self.device,
        )
        if not weights.foot_segmentation.exists():
            raise AnalysisError(f"Foot segmentation model not found: {weights.foot_segmentation}")
        self.foot_model = YOLO(str(weights.foot_segmentation))

    def analyze(self, image_bytes: bytes, filename: str | None) -> TineaAnalysisResult:
        pil_image, image_bgr = load_image(image_bytes)
        h, w = image_bgr.shape[:2]
        probs = predict_tinea_probs(self.tinea_model, pil_image, self.sam_input_size, self.device)
        fungal_mask, inflammation_mask = threshold_predictions(
            probs,
            self.fungal_threshold,
            self.inflammation_threshold,
        )
        foot_mask, foot_found = predict_foot_mask(
            foot_model=self.foot_model,
            image_bgr=image_bgr,
            conf=settings.foot_confidence,
            min_component_area=settings.foot_min_component_area,
            iou_dup_threshold=settings.foot_iou_duplicate_threshold,
        )
        overlay_fungal_mask, overlay_inflammation_mask = masks_for_visualization(
            fungal_mask,
            inflammation_mask,
            foot_mask,
        )
        metrics = compute_health_scores(
            probs=probs,
            fungal_mask=overlay_fungal_mask,
            inflammation_mask=overlay_inflammation_mask,
            foot_mask=foot_mask,
            foot_found=foot_found,
            fungal_threshold=self.fungal_threshold,
            inflammation_threshold=self.inflammation_threshold,
        )
        suspicion_map, circle_count, evidence_dot_count = render_suspicion_map(
            image_shape=(h, w),
            foot_mask=foot_mask,
            probs=probs,
            fungal_mask=overlay_fungal_mask,
            inflammation_mask=overlay_inflammation_mask,
            fungal_threshold=self.fungal_threshold,
            inflammation_threshold=self.inflammation_threshold,
        )
        photo_overlay = render_photo_overlay(
            image_bgr=image_bgr,
            probs=probs,
            fungal_mask=overlay_fungal_mask,
            inflammation_mask=overlay_inflammation_mask,
            fungal_threshold=self.fungal_threshold,
            inflammation_threshold=self.inflammation_threshold,
        )
        if settings.photo_cutout_background:
            photo_overlay = apply_cutout_background(photo_overlay, foot_mask)
        metrics.update(
            {
                "foot_outline_found": foot_found,
                "fungal_pixels": int(overlay_fungal_mask.sum()),
                "inflammation_pixels": int(overlay_inflammation_mask.sum()),
                "circle_count": circle_count,
                "evidence_dot_count": evidence_dot_count,
                "fungal_threshold": self.fungal_threshold,
                "inflammation_threshold": self.inflammation_threshold,
                "suspicion_map_source": "photo_overlay_segmentation_mask",
                "suspicious_area_map_visualization": "circle_suspicion_map_with_evidence_dots",
                "original_foot_image_visualization": (
                    "photo_overlay_fungal_blue_inflammation_red_cutout_background"
                    if settings.photo_cutout_background
                    else "photo_overlay_fungal_blue_inflammation_red"
                ),
                "max_fungal_prob": float(probs[1].max()),
                "max_inflammation_prob": float(probs[2].max()),
            }
        )

        return TineaAnalysisResult(
            suspicion_map_jpeg=encode_jpeg(suspicion_map),
            photo_overlay_jpeg=encode_jpeg(photo_overlay),
            original_filename=sanitize_filename(filename),
            fungal_safety_score=int(round(float(metrics["fungal_score"]))),
            skin_reaction_safety_score=int(round(float(metrics["inflammation_score"]))),
            metrics=metrics,
        )


_analyzer: TineaAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_tinea_analyzer() -> TineaAnalyzer:
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    with _analyzer_lock:
        if _analyzer is None:
            _analyzer = TineaAnalyzer()
    return _analyzer


def analyze_foot_image(image_bytes: bytes, filename: str | None) -> TineaAnalysisResult:
    analyzer = get_tinea_analyzer()
    return analyzer.analyze(image_bytes, filename)
