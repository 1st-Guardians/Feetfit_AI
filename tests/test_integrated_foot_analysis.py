from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.integrated_foot_analysis import (
    FootGeometryResult,
    IntegratedFootAnalysisResult,
    IntegratedAnalysisError,
    PreparedFoot,
    _anatomical_side_by_board_side,
    _crop_foot,
    prepare_foot_geometry,
)
from app.services.hallux_valgus_analysis import (
    HalluxValgusAnalysisResult,
    score_analysis_text,
)
from app.services.report_text_generation import TineaReportText
from app.services.tinea_analysis import (
    AnalysisError,
    TineaAnalysisResult,
    validate_precomputed_foot_mask,
)


class SharedMaskTests(unittest.TestCase):
    def test_precomputed_mask_is_normalized(self) -> None:
        source = np.zeros((12, 10), dtype=np.float32)
        source[2:9, 3:8] = 0.8

        result = validate_precomputed_foot_mask(source, (12, 10))

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(set(np.unique(result)), {0, 255})
        self.assertEqual(int(np.count_nonzero(result)), 35)

    def test_precomputed_mask_must_match_image(self) -> None:
        with self.assertRaises(AnalysisError):
            validate_precomputed_foot_mask(np.ones((10, 10)), (11, 10))

    def test_rectified_crop_keeps_image_and_mask_aligned(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, :, 1] = 150
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:15, 8:22] = 255

        with patch.object(settings, "aruco_crop_padding_mm", 0.0):
            image_png, crop_mask = _crop_foot(image, mask, pixels_per_mm=5.0)
        decoded = cv2.imdecode(
            np.frombuffer(image_png, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        self.assertEqual(decoded.shape[:2], crop_mask.shape)
        self.assertEqual(crop_mask.shape, (10, 14))
        self.assertTrue(np.all(crop_mask == 255))


class SideMappingTests(unittest.TestCase):
    def test_board_mapping_is_explicit_and_bijective(self) -> None:
        with patch.object(settings, "aruco_image_left_anatomical_side", "right"):
            mapping = _anatomical_side_by_board_side()

        self.assertEqual(mapping["image_left"], "right")
        self.assertEqual(mapping["image_right"], "left")

    def test_invalid_board_mapping_is_rejected(self) -> None:
        with patch.object(settings, "aruco_image_left_anatomical_side", "auto"):
            with self.assertRaises(IntegratedAnalysisError):
                _anatomical_side_by_board_side()


class ApiContractTests(unittest.TestCase):
    def test_hallux_local_fallback_text_is_valid_korean(self) -> None:
        text = score_analysis_text(18.0, 10.0)

        self.assertIn("왼발", text)
        self.assertIn("18.0도", text)
        self.assertNotIn("?", text)

    def test_integrated_route_is_registered_without_removing_legacy_routes(self) -> None:
        paths = app.openapi()["paths"]

        self.assertIn("/api/reports/tina-pedis", paths)
        self.assertIn("/api/reports/hallux-valgus", paths)
        self.assertIn("/api/reports/integrated-foot-analysis", paths)
        content = paths["/api/reports/integrated-foot-analysis"]["post"][
            "requestBody"
        ]["content"]
        self.assertIn("multipart/form-data", content)

    def test_single_upload_runs_both_report_contracts(self) -> None:
        ok, encoded = cv2.imencode(
            ".png",
            np.full((8, 8, 3), 180, dtype=np.uint8),
        )
        self.assertTrue(ok)
        image_png = encoded.tobytes()
        mask = np.full((8, 8), 255, dtype=np.uint8)
        feet = {
            side: PreparedFoot(
                anatomical_side=side,
                board_side=f"image_{side}",
                image_png=image_png,
                mask=mask,
                length_mm=240.0 if side == "left" else 238.0,
                ball_width_mm=92.0 if side == "left" else 91.0,
                segmentation_confidence=0.98,
                length_details={"length_mm": 240.0},
                ball_width_details={"ball_width_mm": 92.0},
            )
            for side in ("left", "right")
        }
        geometry = FootGeometryResult(
            original_filename="feet.jpg",
            input_width=1280,
            input_height=720,
            measurement_valid=True,
            measurement_status="valid_metric_measurement",
            measurement_invalid_reasons=(),
            orientation_transform="rotate_180",
            detected_marker_ids=(0, 1, 2, 3, 4, 5),
            missing_marker_ids=(),
            lens_correction={"applied": True},
            global_calibration={"all_marker_validation_world_rmse_mm": 1.0},
            feet=feet,
        )
        tinea = {
            side: TineaAnalysisResult(
                suspicion_map_png=image_png,
                photo_overlay_png=image_png,
                original_filename=f"{side}.png",
                fungal_safety_score=95,
                skin_reaction_safety_score=94,
                metrics={"foot_mask_source": "shared_aruco_segmentation"},
            )
            for side in ("left", "right")
        }
        hallux = {
            side: HalluxValgusAnalysisResult(
                analysis_png=image_png,
                original_filename=f"{side}.png",
                angle_degree=12.0,
                analysis_text="normal",
                score=0.9,
                metrics={"foot_mask_source": "shared_aruco_segmentation"},
            )
            for side in ("left", "right")
        }
        analysis = IntegratedFootAnalysisResult(
            geometry=geometry,
            tinea=tinea,
            hallux_valgus=hallux,
        )

        class FakeAsyncClient:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict]] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback) -> None:
                return None

            async def post(self, url: str, **kwargs):
                self.calls.append((url, kwargs))
                return httpx.Response(
                    201,
                    json={"saved": True},
                    request=httpx.Request("POST", url),
                )

        fake_client = FakeAsyncClient()
        tinea_text = TineaReportText("fungal", "skin", "total")
        with (
            patch(
                "app.api.routes.reports.analyze_integrated_foot_photo",
                return_value=analysis,
            ),
            patch(
                "app.api.routes.reports.generate_tinea_report_text",
                new=AsyncMock(return_value=tinea_text),
            ),
            patch(
                "app.api.routes.reports.generate_hallux_score_analysis_text",
                new=AsyncMock(return_value="hallux"),
            ),
            patch(
                "app.api.routes.reports.httpx.AsyncClient",
                return_value=fake_client,
            ),
        ):
            response = TestClient(app).post(
                "/api/reports/integrated-foot-analysis",
                headers={"Authorization": "Bearer test-token"},
                data={"measurementSessionId": "7"},
                files={"footImage": ("feet.jpg", b"image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["measurementSessionId"], 7)
        self.assertEqual(payload["analysis"]["feet"]["left"]["lengthMm"], 240.0)
        self.assertEqual(len(fake_client.calls), 2)
        self.assertEqual(
            {call[0] for call in fake_client.calls},
            {settings.tinea_report_endpoint, settings.hallux_valgus_report_endpoint},
        )
        tinea_call = next(
            call for call in fake_client.calls if call[0] == settings.tinea_report_endpoint
        )
        hallux_call = next(
            call
            for call in fake_client.calls
            if call[0] == settings.hallux_valgus_report_endpoint
        )
        self.assertEqual(
            set(tinea_call[1]["files"]),
            {"suspiciousAreaMapImage", "originalFootImage"},
        )
        self.assertEqual(
            set(hallux_call[1]["files"]),
            {"leftFootImage", "rightFootImage"},
        )


@unittest.skipUnless(
    os.getenv("RUN_ARUCO_SMOKE_TESTS") == "1",
    "Set RUN_ARUCO_SMOKE_TESTS=1 to run model-backed ArUco smoke tests.",
)
class ArucoModelSmokeTests(unittest.TestCase):
    def test_reference_photo_produces_two_valid_feet(self) -> None:
        image_path = (
            settings.aruco_source_dir
            / "examples"
            / "photos"
            / "photo_20260715_045510.jpg"
        )
        self.assertTrue(image_path.is_file(), image_path)

        result = prepare_foot_geometry(
            Path(image_path).read_bytes(),
            image_path.name,
        )

        self.assertTrue(result.measurement_valid)
        self.assertEqual(set(result.feet), {"left", "right"})
        self.assertEqual(result.detected_marker_ids, (0, 1, 2, 3, 4, 5))
        for foot in result.feet.values():
            self.assertGreater(foot.length_mm, 100.0)
            self.assertGreater(foot.ball_width_mm, 40.0)
            self.assertEqual(foot.image_png[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
