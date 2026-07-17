from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Callable

import cv2
import numpy as np

from app.core.config import settings
from app.services.hallux_valgus_analysis import (
    HalluxValgusAnalysisResult,
    analyze_hallux_valgus_image,
)
from app.services.tinea_analysis import (
    AnalysisError,
    TineaAnalysisResult,
    analyze_foot_image,
    load_image,
)


class IntegratedAnalysisError(AnalysisError):
    """Raised when the shared photo/measurement pipeline cannot finish."""


@dataclass(frozen=True)
class PreparedFoot:
    anatomical_side: str
    board_side: str
    image_png: bytes
    mask: np.ndarray
    length_mm: float
    ball_width_mm: float
    segmentation_confidence: float | None
    length_details: dict[str, Any]
    ball_width_details: dict[str, Any]


@dataclass(frozen=True)
class FootGeometryResult:
    original_filename: str
    input_width: int
    input_height: int
    measurement_valid: bool
    measurement_status: str
    measurement_invalid_reasons: tuple[str, ...]
    orientation_transform: str
    detected_marker_ids: tuple[int, ...]
    missing_marker_ids: tuple[int, ...]
    lens_correction: dict[str, Any]
    global_calibration: dict[str, Any]
    feet: dict[str, PreparedFoot]


@dataclass(frozen=True)
class IntegratedFootAnalysisResult:
    geometry: FootGeometryResult
    tinea: dict[str, TineaAnalysisResult]
    hallux_valgus: dict[str, HalluxValgusAnalysisResult]


@dataclass(frozen=True)
class _ArucoRuntime:
    source_dir: Path
    photo_module: ModuleType
    yolo_module: ModuleType
    ball_width_module: ModuleType
    detector_type: type
    world_points_mm: Any
    measurement_error_type: type[Exception]


@dataclass
class _CachedYolo:
    model: Any
    inference_lock: threading.Lock


_runtime: _ArucoRuntime | None = None
_runtime_lock = threading.Lock()
_segmentation_capture = threading.local()
_yolo_cache: dict[tuple[str, str], _CachedYolo] = {}
_yolo_cache_lock = threading.Lock()


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _module_from_source(name: str, source_dir: Path) -> ModuleType:
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if not module_file or not _path_is_within(Path(module_file), source_dir):
        raise IntegratedAnalysisError(
            f"ArUco module '{name}' was imported from an unexpected location: "
            f"{module_file or 'unknown'}"
        )
    return module


def _get_cached_yolo(model_path: Path, device: str | None) -> _CachedYolo:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_file():
        raise IntegratedAnalysisError(
            f"ArUco foot segmentation weights not found: {resolved}"
        )
    key = (str(resolved), device or "")
    cached = _yolo_cache.get(key)
    if cached is not None:
        return cached

    with _yolo_cache_lock:
        cached = _yolo_cache.get(key)
        if cached is None:
            try:
                from ultralytics import YOLO

                model = YOLO(str(resolved))
            except Exception as exc:  # pragma: no cover - backend-specific detail
                raise IntegratedAnalysisError(
                    f"Failed to load ArUco foot segmentation model: {exc}"
                ) from exc
            cached = _CachedYolo(model=model, inference_lock=threading.Lock())
            _yolo_cache[key] = cached
    return cached


def _cached_predictor(
    yolo_module: ModuleType,
    model_path: Path,
    configured_device: str | None,
) -> Callable[..., list[Any]]:
    def predict(image_bgr: np.ndarray, **options: Any) -> list[Any]:
        device = options.get("device") or configured_device
        cached = _get_cached_yolo(model_path, device)
        predict_options: dict[str, Any] = {
            "source": image_bgr,
            "conf": float(options.get("conf", settings.aruco_yolo_confidence)),
            "imgsz": int(options.get("imgsz", settings.aruco_yolo_image_size)),
            "retina_masks": True,
            "save": False,
            "verbose": False,
        }
        if device:
            predict_options["device"] = device
        try:
            with cached.inference_lock:
                results = cached.model.predict(**predict_options)
            return yolo_module._extract_ultralytics_detections(
                results,
                fallback_names=getattr(cached.model, "names", None),
            )
        except Exception as exc:  # pragma: no cover - backend-specific detail
            raise IntegratedAnalysisError(
                f"ArUco foot segmentation inference failed: {exc}"
            ) from exc

    return predict


def _install_segmentation_capture(
    photo_module: ModuleType,
    yolo_module: ModuleType,
) -> None:
    if getattr(photo_module, "_feetfit_capture_installed", False):
        return

    original = photo_module.segment_feet_yolo

    def capture(*args: Any, **kwargs: Any) -> Any:
        model_path = Path(
            kwargs.get("model_path")
            or settings.aruco_foot_segmentation_weights
            or (settings.weights_dir / "foot_seg_yolo11n_best.pt")
        )
        configured_device = str(kwargs.get("device") or "").strip() or None
        result = original(
            *args,
            **kwargs,
            predictor=_cached_predictor(
                yolo_module,
                model_path,
                configured_device,
            ),
        )
        _segmentation_capture.value = result
        return result

    photo_module.segment_feet_yolo = capture
    photo_module._feetfit_capture_installed = True


def _load_aruco_runtime() -> _ArucoRuntime:
    global _runtime
    source_dir = settings.aruco_source_dir.expanduser().resolve()
    if _runtime is not None:
        if _runtime.source_dir != source_dir:
            raise IntegratedAnalysisError(
                "ARUCO_SOURCE_DIR changed after the ArUco runtime was loaded; "
                "restart the server to apply it."
            )
        return _runtime

    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        if not (source_dir / "measure_foot_photo.py").is_file():
            raise IntegratedAnalysisError(
                f"ArUco source repository not found or incomplete: {source_dir}"
            )
        source_text = str(source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        try:
            photo_module = _module_from_source("measure_foot_photo", source_dir)
            yolo_module = _module_from_source(
                "aruco_foot_measure.yolo_foot", source_dir
            )
            ball_width_module = _module_from_source(
                "aruco_foot_measure.ball_width", source_dir
            )
            app_module = _module_from_source("aruco_foot_measure.app", source_dir)
            pencil_module = _module_from_source(
                "aruco_foot_measure.pencil", source_dir
            )
            _install_segmentation_capture(photo_module, yolo_module)
        except IntegratedAnalysisError:
            raise
        except Exception as exc:
            raise IntegratedAnalysisError(
                f"Failed to load ArUco measurement code: {exc}"
            ) from exc

        _runtime = _ArucoRuntime(
            source_dir=source_dir,
            photo_module=photo_module,
            yolo_module=yolo_module,
            ball_width_module=ball_width_module,
            detector_type=app_module.ArucoMarkerDetector,
            world_points_mm=pencil_module.WORLD_POINTS_MM,
            measurement_error_type=pencil_module.PencilMeasurementError,
        )
    return _runtime


def _calibration_path(runtime: _ArucoRuntime) -> Path:
    path = settings.aruco_camera_calibration
    return (
        path.expanduser().resolve()
        if path is not None
        else runtime.source_dir / "models" / "hybrid_best_1280x720.npz"
    )


def _segmentation_weights_path() -> Path:
    path = settings.aruco_foot_segmentation_weights
    return (
        path.expanduser().resolve()
        if path is not None
        else (settings.weights_dir / "foot_seg_yolo11n_best.pt").resolve()
    )


def _build_measurement_args(filename: str | None, runtime: _ArucoRuntime) -> argparse.Namespace:
    yolo_device = settings.aruco_yolo_device.strip() or None
    return argparse.Namespace(
        image=Path(filename or "foot-photo.jpg").name,
        camera_calibration=_calibration_path(runtime),
        undistort_balance=settings.aruco_undistort_balance,
        allow_calibration_resize=settings.aruco_allow_calibration_resize,
        undistorted_output=None,
        feet="both",
        mirror=False,
        marker_size_mm=settings.aruco_marker_size_mm,
        marker_row_spacing_mm=settings.aruco_marker_row_spacing_mm,
        marker_column_spacing_mm=settings.aruco_marker_column_spacing_mm,
        fixed_offset_mm=settings.aruco_fixed_offset_mm,
        visible_length_scale=settings.aruco_visible_length_scale,
        reference_edge=settings.aruco_reference_edge,
        foot_segmentation="yolo",
        yolo_model=_segmentation_weights_path(),
        yolo_conf=settings.aruco_yolo_confidence,
        yolo_mask_threshold=settings.aruco_yolo_mask_threshold,
        yolo_imgsz=settings.aruco_yolo_image_size,
        yolo_device=yolo_device,
        toe_refinement=settings.aruco_toe_refinement,
        toe_max_extension_mm=settings.aruco_toe_max_extension_mm,
    )


def _anatomical_side_by_board_side() -> dict[str, str]:
    image_left_side = settings.aruco_image_left_anatomical_side.strip().lower()
    if image_left_side not in {"left", "right"}:
        raise IntegratedAnalysisError(
            "ARUCO_IMAGE_LEFT_ANATOMICAL_SIDE must be 'left' or 'right'."
        )
    return {
        "image_left": image_left_side,
        "image_right": "right" if image_left_side == "left" else "left",
    }


def _encode_png(image: np.ndarray, label: str) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise IntegratedAnalysisError(f"Failed to encode {label} as PNG.")
    return encoded.tobytes()


def _crop_foot(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    pixels_per_mm: float,
) -> tuple[bytes, np.ndarray]:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if binary.shape != image_bgr.shape[:2]:
        raise IntegratedAnalysisError(
            "Rectified foot mask shape does not match the rectified photo."
        )
    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        raise IntegratedAnalysisError("ArUco foot segmentation returned an empty mask.")

    padding = max(0, int(round(settings.aruco_crop_padding_mm * pixels_per_mm)))
    height, width = binary.shape
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(width, int(xs.max()) + padding + 1)
    y2 = min(height, int(ys.max()) + padding + 1)
    crop_bgr = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = binary[y1:y2, x1:x2].copy()
    return _encode_png(crop_bgr, "rectified foot crop"), crop_mask


def _ball_width_details(value: Any) -> dict[str, Any]:
    return {
        "ball_width_mm": round(float(value.width_mm), 3),
        "ball_width_cm": round(float(value.width_mm) / 10.0, 3),
        "position_from_toe_mm": round(float(value.distance_from_toe_mm), 3),
        "position_fraction_from_toe": round(float(value.fraction_from_toe), 6),
        "position_fraction_from_heel": round(float(value.fraction_from_heel), 6),
        "candidate_cross_section_count": int(value.candidate_count),
        "candidate_width_spread_mm": round(
            float(value.candidate_width_spread_mm), 3
        ),
        "method": str(value.method),
        "measurement_definition": "straight-line MTP-zone width, not girth",
    }


def prepare_foot_geometry(
    image_bytes: bytes,
    filename: str | None,
) -> FootGeometryResult:
    """Run lens, marker, shared segmentation, split, length, and ball width once."""

    runtime = _load_aruco_runtime()
    _, source_bgr = load_image(image_bytes)
    height, width = source_bgr.shape[:2]
    expected_size = (
        settings.aruco_expected_image_width,
        settings.aruco_expected_image_height,
    )
    if (
        not settings.aruco_allow_calibration_resize
        and (width, height) != expected_size
    ):
        raise IntegratedAnalysisError(
            "ArUco calibration requires an unmodified "
            f"{expected_size[0]}x{expected_size[1]} image; received {width}x{height}."
        )

    args = _build_measurement_args(filename, runtime)
    calibration_path = Path(args.camera_calibration)
    if not calibration_path.is_file():
        raise IntegratedAnalysisError(
            f"ArUco camera calibration not found: {calibration_path}"
        )

    try:
        corrected, lens_metadata, undistortion_maps = (
            runtime.photo_module._apply_optional_lens_correction(
                args,
                source_bgr,
                return_maps=True,
            )
        )
        detector = runtime.detector_type(
            settings.aruco_dictionary,
            expected_marker_ids=runtime.world_points_mm,
        )
        _segmentation_capture.value = None
        bundle = runtime.photo_module._measure_marker_offset_layout(
            args,
            corrected,
            detector,
            lens_metadata,
            distorted_source_image=source_bgr,
            undistortion_maps=undistortion_maps,
        )
        segmentation = getattr(_segmentation_capture, "value", None)
        _segmentation_capture.value = None
        if segmentation is None:
            raise IntegratedAnalysisError(
                "ArUco pipeline did not expose the shared foot segmentation."
            )

        rectified = bundle["rectified"]
        measurements = bundle["measurements"]
        widths = {
            board_side: runtime.ball_width_module.measure_foot_ball_width(
                rectified,
                measurement,
                search_start_fraction_from_toe=(
                    settings.aruco_forefoot_start_fraction
                ),
                search_end_fraction_from_toe=settings.aruco_forefoot_end_fraction,
                target_fraction_from_toe=settings.aruco_mtp_target_fraction,
                slice_band_mm=settings.aruco_ball_slice_band_mm,
                slice_step_mm=settings.aruco_ball_slice_step_mm,
            )
            for board_side, measurement in measurements.items()
        }
    except IntegratedAnalysisError:
        raise
    except Exception as exc:
        raise IntegratedAnalysisError(
            f"ArUco geometry analysis failed: {exc}"
        ) from exc
    finally:
        _segmentation_capture.value = None

    side_mapping = _anatomical_side_by_board_side()
    prepared_feet: dict[str, PreparedFoot] = {}
    base_result = bundle["result"]
    for board_side in ("image_left", "image_right"):
        if board_side not in measurements or board_side not in segmentation.by_side:
            raise IntegratedAnalysisError(
                f"ArUco pipeline did not produce both feet; missing {board_side}."
            )
        anatomical_side = side_mapping[board_side]
        measurement = measurements[board_side]
        prediction = segmentation.by_side[board_side]
        image_png, crop_mask = _crop_foot(
            rectified.image_bgr,
            prediction.mask,
            float(rectified.pixels_per_mm),
        )
        width_value = widths[board_side]
        prepared_feet[anatomical_side] = PreparedFoot(
            anatomical_side=anatomical_side,
            board_side=board_side,
            image_png=image_png,
            mask=crop_mask,
            length_mm=round(float(measurement.length_mm), 3),
            ball_width_mm=round(float(width_value.width_mm), 3),
            segmentation_confidence=round(float(prediction.confidence), 6),
            length_details=dict(base_result["feet"][board_side]),
            ball_width_details=_ball_width_details(width_value),
        )

    if set(prepared_feet) != {"left", "right"}:
        raise IntegratedAnalysisError(
            "The configured ArUco board-to-anatomy mapping is not one-to-one."
        )

    return FootGeometryResult(
        original_filename=Path(filename or "foot-photo.jpg").name,
        input_width=width,
        input_height=height,
        measurement_valid=bool(bundle["measurement_valid"]),
        measurement_status=str(base_result["measurement_status"]),
        measurement_invalid_reasons=tuple(bundle["measurement_invalid_reasons"]),
        orientation_transform=str(bundle["transform_name"]),
        detected_marker_ids=tuple(base_result["detected_marker_ids"]),
        missing_marker_ids=tuple(base_result["missing_marker_ids"]),
        lens_correction=dict(base_result["lens_correction"]),
        global_calibration=dict(base_result["global_calibration"]),
        feet=prepared_feet,
    )


def analyze_integrated_foot_photo(
    image_bytes: bytes,
    filename: str | None,
) -> IntegratedFootAnalysisResult:
    """Analyze one calibrated two-foot photo with every downstream model."""

    geometry = prepare_foot_geometry(image_bytes, filename)

    tinea_results: dict[str, TineaAnalysisResult] = {}
    for side in ("left", "right"):
        foot = geometry.feet[side]
        tinea_results[side] = analyze_foot_image(
            foot.image_png,
            f"{side}_{geometry.original_filename}.png",
            side,
            foot_mask=foot.mask,
        )

    hallux_results: dict[str, HalluxValgusAnalysisResult] = {}
    for side in ("left", "right"):
        foot = geometry.feet[side]
        hallux_results[side] = analyze_hallux_valgus_image(
            foot.image_png,
            f"{side}_{geometry.original_filename}.png",
            side,
            foot_mask=foot.mask,
        )

    return IntegratedFootAnalysisResult(
        geometry=geometry,
        tinea=tinea_results,
        hallux_valgus=hallux_results,
    )
