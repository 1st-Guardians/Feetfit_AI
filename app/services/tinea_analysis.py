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
MAP_INFLAMMATION_COLOR = (135, 105, 255)  # BGR stronger pink-red
FOOT_OUTLINE_COLOR = (165, 165, 165)


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class TineaAnalysisResult:
    suspicion_map_png: bytes
    photo_overlay_png: bytes
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


def enhance_tinea_input_image(image: Image.Image) -> Image.Image:
    if not settings.tinea_preprocess_enhance_enabled:
        return image

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    clahe_clip = max(0.1, float(settings.tinea_preprocess_clahe_clip_limit))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    contrast_gain = max(0.0, float(settings.tinea_preprocess_contrast_gain))
    if abs(contrast_gain - 1.0) > 1e-3:
        arr = np.clip((arr.astype(np.float32) - 127.5) * contrast_gain + 127.5, 0, 255).astype(np.uint8)

    red_sat_gain = max(0.0, float(settings.tinea_preprocess_red_saturation_gain))
    red_value_gain = max(0.0, float(settings.tinea_preprocess_red_value_gain))
    if red_sat_gain > 1.0 or red_value_gain > 1.0:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        red_mask = ((hue <= 12.0) | (hue >= 168.0)) & (saturation >= 35.0) & (value >= 40.0)
        if red_mask.any():
            saturation[red_mask] = np.clip(saturation[red_mask] * red_sat_gain, 0, 255)
            value[red_mask] = np.clip(value[red_mask] * red_value_gain, 0, 255)
            hsv[:, :, 1] = saturation
            hsv[:, :, 2] = value
            arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return Image.fromarray(arr)


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
    sam_image = resize_to_square(enhance_tinea_input_image(image), sam_input_size)
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


def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]

    step = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _sliding_window_boxes(
    image_shape: tuple[int, int],
    foot_mask: np.ndarray,
    tile_size: int,
    overlap: float,
    padding: int,
    max_tiles: int,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    h, w = image_shape
    x1, y1, x2, y2 = bbox_from_mask(foot_mask, padding)
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    tile_w = min(max(1, tile_size), crop_w)
    tile_h = min(max(1, tile_size), crop_h)
    overlap = float(np.clip(overlap, 0.0, 0.85))

    boxes = []
    for local_y in _tile_starts(crop_h, tile_h, overlap):
        for local_x in _tile_starts(crop_w, tile_w, overlap):
            bx1 = int(np.clip(x1 + local_x, 0, w - 1))
            by1 = int(np.clip(y1 + local_y, 0, h - 1))
            bx2 = int(np.clip(bx1 + tile_w, bx1 + 1, w))
            by2 = int(np.clip(by1 + tile_h, by1 + 1, h))
            boxes.append((bx1, by1, bx2, by2))

    if len(boxes) > max_tiles > 0:
        boxes.sort(
            key=lambda box: int((foot_mask[box[1] : box[3], box[0] : box[2]] > 0).sum()),
            reverse=True,
        )
        boxes = boxes[:max_tiles]
        boxes.sort(key=lambda box: (box[1], box[0]))

    return boxes, (x1, y1, x2, y2)


def _tile_blend_weight(height: int, width: int) -> np.ndarray:
    if height <= 1 or width <= 1:
        return np.ones((height, width), dtype=np.float32)

    y = 1.0 - np.abs(np.linspace(-1.0, 1.0, height, dtype=np.float32))
    x = 1.0 - np.abs(np.linspace(-1.0, 1.0, width, dtype=np.float32))
    weight = y[:, None] * x[None, :]
    return np.clip(weight, 0.18, 1.0).astype(np.float32)


def predict_tinea_probs_with_sliding_window(
    model: SamMultiClassSegModel,
    image: Image.Image,
    sam_input_size: int,
    device: str,
    foot_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    full_probs = predict_tinea_probs(model, image, sam_input_size, device)
    metrics = {
        "tinea_sliding_window_enabled": bool(settings.tinea_sliding_window_enabled),
        "tinea_sliding_window_applied": False,
        "tinea_sliding_window_tile_count": 0,
        "tinea_sliding_window_merge": "class_max",
        "tinea_preprocess_enhance_enabled": bool(settings.tinea_preprocess_enhance_enabled),
        "tinea_preprocess_contrast_gain": float(settings.tinea_preprocess_contrast_gain),
        "tinea_preprocess_clahe_clip_limit": float(settings.tinea_preprocess_clahe_clip_limit),
        "tinea_preprocess_red_saturation_gain": float(settings.tinea_preprocess_red_saturation_gain),
        "tinea_preprocess_red_value_gain": float(settings.tinea_preprocess_red_value_gain),
    }
    if not settings.tinea_sliding_window_enabled:
        return full_probs, metrics

    width, height = image.size
    tile_size = max(64, int(settings.tinea_sliding_window_tile_size))
    overlap = float(np.clip(settings.tinea_sliding_window_overlap, 0.0, 0.85))
    padding = max(0, int(settings.tinea_sliding_window_padding))
    max_tiles = max(1, int(settings.tinea_sliding_window_max_tiles))
    boxes, foot_bbox = _sliding_window_boxes(
        image_shape=(height, width),
        foot_mask=foot_mask,
        tile_size=tile_size,
        overlap=overlap,
        padding=padding,
        max_tiles=max_tiles,
    )
    if not boxes:
        return full_probs, metrics

    accum = np.zeros_like(full_probs, dtype=np.float32)
    weights_accum = np.zeros((height, width), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        tile_image = image.crop((x1, y1, x2, y2))
        tile_probs = predict_tinea_probs(model, tile_image, sam_input_size, device)
        tile_h, tile_w = tile_probs.shape[1:]
        weight = _tile_blend_weight(tile_h, tile_w)
        accum[:, y1:y2, x1:x2] += tile_probs[:, : y2 - y1, : x2 - x1] * weight[: y2 - y1, : x2 - x1]
        weights_accum[y1:y2, x1:x2] += weight[: y2 - y1, : x2 - x1]

    covered = weights_accum > 1e-6
    tile_merged = full_probs.copy()
    tile_merged[:, covered] = accum[:, covered] / weights_accum[covered]

    combined = full_probs.copy()
    for class_idx in (1, 2):
        combined[class_idx, covered] = np.maximum(full_probs[class_idx, covered], tile_merged[class_idx, covered])

    metrics.update(
        {
            "tinea_sliding_window_applied": True,
            "tinea_sliding_window_tile_count": len(boxes),
            "tinea_sliding_window_tile_size": tile_size,
            "tinea_sliding_window_overlap": overlap,
            "tinea_sliding_window_padding": padding,
            "tinea_sliding_window_max_tiles": max_tiles,
            "tinea_sliding_window_foot_bbox": foot_bbox,
        }
    )
    return combined, metrics


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


def validate_precomputed_foot_mask(
    foot_mask: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Normalize a shared upstream foot mask without running YOLO again."""

    mask = np.asarray(foot_mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2 or mask.shape != image_shape:
        raise AnalysisError(
            "Precomputed foot mask must be a 2D array matching the input image."
        )
    if not np.isfinite(mask).all():
        raise AnalysisError("Precomputed foot mask contains non-finite values.")
    binary = (mask > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(binary)) < 10:
        raise AnalysisError("Precomputed foot mask is empty or too small.")
    return binary


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
        min_alpha=0.34,
        max_alpha=0.78,
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


def apply_cutout_alpha(image_bgr: np.ndarray, foot_mask: np.ndarray) -> np.ndarray:
    alpha = (foot_mask > 0).astype(np.float32) * 255.0
    if not alpha.all():
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0, sigmaY=1.0)

    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    return np.dstack([image_bgr, alpha])


def crop_to_foot(image: np.ndarray, foot_mask: np.ndarray, padding: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_from_mask(foot_mask, padding)
    return image[y1:y2, x1:x2].copy()


def _toe_names_left_to_right(foot_side: str | None) -> list[str]:
    if foot_side == "left":
        return ["새끼발가락", "넷째 발가락", "셋째 발가락", "둘째 발가락", "엄지발가락"]
    if foot_side == "right":
        return ["엄지발가락", "둘째 발가락", "셋째 발가락", "넷째 발가락", "새끼발가락"]
    return ["가쪽 발가락", "넷째 발가락", "가운데 발가락", "둘째 발가락", "안쪽 발가락"]


def _horizontal_region_label(x_norm: float) -> str:
    if x_norm < 0.33:
        return "왼쪽"
    if x_norm > 0.67:
        return "오른쪽"
    return "중앙"


def _toe_region_label(foot_side: str | None, x_norm: float, x_min_norm: float, x_max_norm: float) -> str:
    names = _toe_names_left_to_right(foot_side)
    center_idx = int(np.clip(np.floor(x_norm * len(names)), 0, len(names) - 1))
    start_idx = int(np.clip(np.floor(x_min_norm * len(names)), 0, len(names) - 1))
    end_idx = int(np.clip(np.floor(x_max_norm * len(names)), 0, len(names) - 1))

    if end_idx > start_idx:
        span = names[start_idx : end_idx + 1]
        if len(span) == 2:
            return f"{span[0]}과 {span[1]} 사이"
        return f"{span[0]}~{span[-1]} 주변"
    return f"{names[center_idx]} 주변"


def _component_location_label(component: np.ndarray, foot_mask: np.ndarray, foot_side: str | None) -> str:
    x1, y1, x2, y2 = bbox_from_mask(foot_mask, 0)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    ys, xs = np.where(component)
    if xs.size == 0 or ys.size == 0:
        return "발 영역"

    cx = float(xs.mean())
    cy = float(ys.mean())
    x_norm = float(np.clip((cx - x1) / width, 0.0, 0.999))
    y_norm = float(np.clip((cy - y1) / height, 0.0, 0.999))
    x_min_norm = float(np.clip((float(xs.min()) - x1) / width, 0.0, 0.999))
    x_max_norm = float(np.clip((float(xs.max()) - x1) / width, 0.0, 0.999))

    if y_norm < 0.34:
        return _toe_region_label(foot_side, x_norm, x_min_norm, x_max_norm)
    if y_norm < 0.52:
        return f"{_horizontal_region_label(x_norm)} 앞발/발등"
    if y_norm < 0.76:
        return f"{_horizontal_region_label(x_norm)} 발 중앙부"
    return f"{_horizontal_region_label(x_norm)} 뒤꿈치 주변"


def summarize_suspicion_regions(
    mask: np.ndarray,
    prob: np.ndarray,
    foot_mask: np.ndarray,
    threshold: float,
    foot_side: str | None,
    max_regions: int = 5,
) -> list[dict]:
    foot_region = foot_mask > 0
    foot_pixels = int(foot_region.sum())
    if foot_pixels == 0:
        return []

    regions = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask & foot_region).astype(np.uint8), 8)
    min_area = max(1, int(settings.min_suspicion_area))
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component = labels == label_idx
        ys, xs = np.where(component)
        if xs.size == 0 or ys.size == 0:
            continue

        values = prob[ys, xs]
        weights = np.clip(values - threshold, 0.0, None) + 1e-3
        cx = float(np.average(xs, weights=weights))
        cy = float(np.average(ys, weights=weights))
        x1, y1, x2, y2 = bbox_from_mask(foot_mask, 0)
        regions.append(
            {
                "location": _component_location_label(component, foot_mask, foot_side),
                "area_pixels": area,
                "foot_area_ratio": round(float(area / foot_pixels), 6),
                "confidence_p90": round(float(np.percentile(values, 90)), 4),
                "center_x_ratio": round(float(np.clip((cx - x1) / max(1, x2 - x1), 0.0, 1.0)), 4),
                "center_y_ratio": round(float(np.clip((cy - y1) / max(1, y2 - y1), 0.0, 1.0)), 4),
            }
        )

    regions.sort(key=lambda item: (item["area_pixels"], item["confidence_p90"]), reverse=True)
    return regions[:max_regions]


def render_hallux_style_png(image_bgr: np.ndarray, foot_mask: np.ndarray) -> bytes:
    cutout = apply_cutout_alpha(image_bgr, foot_mask)
    cropped = crop_to_foot(cutout, foot_mask, settings.photo_cutout_padding)
    return encode_png(cropped)


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


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise AnalysisError("Failed to encode analysis image.")
    return encoded.tobytes()


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AnalysisError("Failed to decode analysis image.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    return image


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_height:
        return image

    scale = target_height / h
    target_width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def ensure_bgra(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 4:
        return image
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        return np.dstack([image, alpha])
    raise AnalysisError(f"Unsupported image channel count: {image.shape[2]}")


def combine_png_images_side_by_side(
    left_image: bytes,
    right_image: bytes,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    left = decode_image(left_image)
    right = decode_image(right_image)
    use_alpha = (left.ndim == 3 and left.shape[2] == 4) or (right.ndim == 3 and right.shape[2] == 4)
    if use_alpha:
        left = ensure_bgra(left)
        right = ensure_bgra(right)

    target_height = min(max(left.shape[0], right.shape[0]), settings.combined_image_max_height)
    left = resize_to_height(left, target_height)
    right = resize_to_height(right, target_height)

    gap = max(0, settings.combined_image_gap_pixels)
    combined_height = target_height
    combined_width = left.shape[1] + gap + right.shape[1]
    if use_alpha:
        canvas = np.zeros((combined_height, combined_width, 4), dtype=np.uint8)
        canvas[:, :, :3] = np.array(background_color, dtype=np.uint8)
    else:
        canvas = np.full((combined_height, combined_width, 3), background_color, dtype=np.uint8)
    canvas[:, : left.shape[1]] = left
    canvas[:, left.shape[1] + gap : left.shape[1] + gap + right.shape[1]] = right
    return encode_png(canvas)


class TineaAnalyzer:
    def __init__(self) -> None:
        self.device = resolve_device(settings.analysis_device)
        self.tinea_model, self.sam_input_size, self.fungal_threshold, self.inflammation_threshold = load_tinea_model(
            weights.tinea_pedis_segmentation,
            self.device,
        )
        self.foot_model = None
        self._foot_model_lock = threading.Lock()

    def get_foot_model(self):
        if self.foot_model is not None:
            return self.foot_model
        with self._foot_model_lock:
            if self.foot_model is None:
                if not weights.foot_segmentation.exists():
                    raise AnalysisError(
                        f"Foot segmentation model not found: {weights.foot_segmentation}"
                    )
                self.foot_model = YOLO(str(weights.foot_segmentation))
        return self.foot_model

    def analyze(
        self,
        image_bytes: bytes,
        filename: str | None,
        foot_side: str | None = None,
        *,
        foot_mask: np.ndarray | None = None,
    ) -> TineaAnalysisResult:
        pil_image, image_bgr = load_image(image_bytes)
        h, w = image_bgr.shape[:2]
        if foot_mask is None:
            analyzed_foot_mask, foot_found = predict_foot_mask(
                foot_model=self.get_foot_model(),
                image_bgr=image_bgr,
                conf=settings.foot_confidence,
                min_component_area=settings.foot_min_component_area,
                iou_dup_threshold=settings.foot_iou_duplicate_threshold,
            )
            mask_source = "tinea_yolo"
        else:
            analyzed_foot_mask = validate_precomputed_foot_mask(
                foot_mask,
                (h, w),
            )
            foot_found = True
            mask_source = "shared_aruco_segmentation"
        probs, inference_metrics = predict_tinea_probs_with_sliding_window(
            self.tinea_model,
            pil_image,
            self.sam_input_size,
            self.device,
            analyzed_foot_mask,
        )
        fungal_mask, inflammation_mask = threshold_predictions(
            probs,
            self.fungal_threshold,
            self.inflammation_threshold,
        )
        overlay_fungal_mask, overlay_inflammation_mask = masks_for_visualization(
            fungal_mask,
            inflammation_mask,
            analyzed_foot_mask,
        )
        metrics = compute_health_scores(
            probs=probs,
            fungal_mask=overlay_fungal_mask,
            inflammation_mask=overlay_inflammation_mask,
            foot_mask=analyzed_foot_mask,
            foot_found=foot_found,
            fungal_threshold=self.fungal_threshold,
            inflammation_threshold=self.inflammation_threshold,
        )
        suspicion_map, circle_count, evidence_dot_count = render_suspicion_map(
            image_shape=(h, w),
            foot_mask=analyzed_foot_mask,
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
            suspicion_map_png = render_hallux_style_png(
                suspicion_map, analyzed_foot_mask
            )
            photo_overlay_png = render_hallux_style_png(
                photo_overlay, analyzed_foot_mask
            )
        else:
            suspicion_map_png = encode_png(suspicion_map)
            photo_overlay_png = encode_png(photo_overlay)
        metrics.update(
            {
                "foot_outline_found": foot_found,
                "foot_mask_source": mask_source,
                **inference_metrics,
                "fungal_pixels": int(overlay_fungal_mask.sum()),
                "inflammation_pixels": int(overlay_inflammation_mask.sum()),
                "circle_count": circle_count,
                "evidence_dot_count": evidence_dot_count,
                "fungal_threshold": self.fungal_threshold,
                "inflammation_threshold": self.inflammation_threshold,
                "suspicion_map_source": "photo_overlay_segmentation_mask",
                "suspicious_area_map_visualization": "circle_suspicion_map_with_evidence_dots",
                "original_foot_image_visualization": (
                    "photo_overlay_fungal_blue_inflammation_red_transparent_cutout_crop"
                    if settings.photo_cutout_background
                    else "photo_overlay_fungal_blue_inflammation_red"
                ),
                "max_fungal_prob": float(probs[1].max()),
                "max_inflammation_prob": float(probs[2].max()),
                "foot_side": foot_side or "unknown",
                "fungal_regions": summarize_suspicion_regions(
                    overlay_fungal_mask,
                    probs[1],
                    analyzed_foot_mask,
                    self.fungal_threshold,
                    foot_side,
                ),
                "inflammation_regions": summarize_suspicion_regions(
                    overlay_inflammation_mask,
                    probs[2],
                    analyzed_foot_mask,
                    self.inflammation_threshold,
                    foot_side,
                ),
            }
        )

        return TineaAnalysisResult(
            suspicion_map_png=suspicion_map_png,
            photo_overlay_png=photo_overlay_png,
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


def analyze_foot_image(
    image_bytes: bytes,
    filename: str | None,
    foot_side: str | None = None,
    *,
    foot_mask: np.ndarray | None = None,
) -> TineaAnalysisResult:
    analyzer = get_tinea_analyzer()
    return analyzer.analyze(
        image_bytes,
        filename,
        foot_side,
        foot_mask=foot_mask,
    )
