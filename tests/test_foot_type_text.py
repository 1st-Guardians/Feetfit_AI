from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import settings
from app.main import app
from app.schemas.reports import (
    FootTypeAnalysisContext,
    FootTypeTextGenerationResponse,
)
from app.services.report_text_generation import (
    FootTypeReportText,
    _create_structured_response,
    build_fallback_foot_type_text,
    generate_foot_type_text,
)


LOW_ARCH_TEXT = (
    "발의 아치가 낮아 발바닥이 넓게 닿는 편이에요. "
    "오래 걷거나 서 있으면 피로가 커질 수 있어 아치를 잘 받쳐주는 신발이 더 편안할 수 있어요."
)
FACTS_HASH = "a" * 64


class FootTypeTextGenerationTests(unittest.IsolatedAsyncioTestCase):
    def test_low_arch_fallback_matches_shoe_list_copy(self) -> None:
        result = build_fallback_foot_type_text(
            FootTypeAnalysisContext(archType="LOW")
        )

        self.assertEqual(result.type_text, LOW_ARCH_TEXT)
        self.assertEqual(result.evidence_id, "ARCH_LOW")
        self.assertEqual(result.source, "FALLBACK")

    def test_every_supported_fallback_avoids_measurement_session_preface(self) -> None:
        contexts = [
            *(
                FootTypeAnalysisContext(archType=value)
                for value in ("LOW", "NORMAL", "HIGH")
            ),
            *(
                FootTypeAnalysisContext(footWidthType=value)
                for value in ("NARROW", "NORMAL", "WIDE")
            ),
            *(
                FootTypeAnalysisContext(pressureBalanceType=value)
                for value in ("LEFT_DOMINANT", "BALANCED", "RIGHT_DOMINANT")
            ),
            FootTypeAnalysisContext(plantarFootprintAnalysisText="압력 분포 차이"),
        ]

        for context in contexts:
            with self.subTest(context=context.model_dump()):
                result = build_fallback_foot_type_text(context)
                self.assertFalse(
                    result.type_text.lstrip().startswith("이번 측정에서는")
                )

    def test_response_contract_rejects_measurement_session_preface(self) -> None:
        with self.assertRaises(ValidationError):
            FootTypeTextGenerationResponse(
                measurementSessionId=30,
                factsHash=FACTS_HASH,
                typeText="이번 측정에서는 오른발에 압력이 더 실려요.",
                evidenceId="PRESSURE_RIGHT_DOMINANT",
                source="FALLBACK",
            )

    def test_raw_dimensions_alone_cannot_create_a_foot_type(self) -> None:
        with self.assertRaises(ValidationError):
            FootTypeAnalysisContext(
                measuredLeftFootSizeMm=250.0,
                measuredRightFootSizeMm=249.0,
                leftFootWidthMm=110.0,
                rightFootWidthMm=109.0,
            )

    def test_pressure_percentages_create_copy_without_banned_opening(self) -> None:
        result = build_fallback_foot_type_text(
            FootTypeAnalysisContext(
                leftPressurePercent=46.0,
                rightPressurePercent=54.0,
            )
        )

        self.assertEqual(result.evidence_id, "PRESSURE_RIGHT_DOMINANT")
        self.assertTrue(result.type_text.startswith("오른발에 압력이 조금 더"))
        self.assertFalse(result.type_text.startswith("이번 측정에서는"))

    def test_near_equal_pressure_is_balanced_by_configured_tolerance(self) -> None:
        context = FootTypeAnalysisContext(
            leftPressurePercent=48.0,
            rightPressurePercent=52.0,
        )
        with patch.object(
            settings, "foot_type_pressure_balance_tolerance_percent", 5.0
        ):
            result = build_fallback_foot_type_text(context)

        self.assertEqual(result.evidence_id, "PRESSURE_BALANCED")
        self.assertFalse(result.type_text.startswith("이번 측정에서는"))

    def test_plantar_analysis_has_safe_non_diagnostic_fallback(self) -> None:
        result = build_fallback_foot_type_text(
            FootTypeAnalysisContext(
                plantarFootprintAnalysisText=(
                    "왼발 뒤꿈치와 오른발 앞꿈치에 압력이 집중되어 있습니다."
                )
            )
        )

        self.assertEqual(result.evidence_id, "PLANTAR_PRESSURE_PATTERN")
        self.assertIn("발바닥의 압력이 부위별로", result.type_text)
        self.assertNotIn("아치가 낮", result.type_text)

    async def test_gpt_may_select_only_a_supplied_evidence_id(self) -> None:
        context = FootTypeAnalysisContext(
            archType="LOW",
            footWidthType="WIDE",
            leftPressurePercent=50.0,
            rightPressurePercent=50.0,
        )
        with (
            patch.object(settings, "openai_api_key", "test-key"),
            patch.object(settings, "openai_report_text_enabled", True),
            patch.object(settings, "openai_foot_type_text_enabled", True),
            patch(
                "app.services.report_text_generation._create_structured_response",
                new=AsyncMock(return_value={"selectedEvidenceId": "WIDTH_WIDE"}),
            ) as create_response,
        ):
            result = await generate_foot_type_text(context)

        self.assertEqual(result.source, "OPENAI")
        self.assertEqual(result.evidence_id, "WIDTH_WIDE")
        schema = create_response.await_args.kwargs["json_schema"]
        self.assertEqual(
            schema["properties"]["selectedEvidenceId"]["enum"],
            ["ARCH_LOW", "WIDTH_WIDE", "PRESSURE_BALANCED"],
        )

    async def test_unknown_gpt_selection_falls_back_to_arch_priority(self) -> None:
        context = FootTypeAnalysisContext(archType="LOW", footWidthType="WIDE")
        with (
            patch.object(settings, "openai_api_key", "test-key"),
            patch.object(settings, "openai_report_text_enabled", True),
            patch.object(settings, "openai_foot_type_text_enabled", True),
            patch(
                "app.services.report_text_generation._create_structured_response",
                new=AsyncMock(return_value={"selectedEvidenceId": "NOT_SUPPLIED"}),
            ),
        ):
            result = await generate_foot_type_text(context)

        self.assertEqual(result.source, "FALLBACK")
        self.assertEqual(result.evidence_id, "ARCH_LOW")

    async def test_openai_failure_falls_back_without_losing_result(self) -> None:
        context = FootTypeAnalysisContext(pressureBalanceType="RIGHT_DOMINANT")
        with (
            patch.object(settings, "openai_api_key", "test-key"),
            patch.object(settings, "openai_report_text_enabled", True),
            patch.object(settings, "openai_foot_type_text_enabled", True),
            patch(
                "app.services.report_text_generation._create_structured_response",
                new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
            ),
        ):
            result = await generate_foot_type_text(context)

        self.assertEqual(result.source, "FALLBACK")
        self.assertEqual(result.evidence_id, "PRESSURE_RIGHT_DOMINANT")
        self.assertFalse(result.type_text.startswith("이번 측정에서는"))

    async def test_openai_responses_request_disables_storage(self) -> None:
        class FakeAsyncClient:
            request_json: dict | None = None

            def __init__(self, *, timeout: float) -> None:
                self.timeout = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback) -> None:
                return None

            async def post(self, url: str, **kwargs) -> httpx.Response:
                type(self).request_json = kwargs["json"]
                return httpx.Response(
                    200,
                    json={"output_text": json.dumps({"selectedEvidenceId": "ARCH_LOW"})},
                    request=httpx.Request("POST", url),
                )

        with (
            patch.object(settings, "openai_api_key", "test-key"),
            patch(
                "app.services.report_text_generation.httpx.AsyncClient",
                FakeAsyncClient,
            ),
        ):
            result = await _create_structured_response(
                system_prompt="test",
                user_payload={"candidateEvidence": [{"evidenceId": "ARCH_LOW"}]},
                schema_name="test_schema",
                json_schema={
                    "type": "object",
                    "properties": {"selectedEvidenceId": {"type": "string"}},
                    "required": ["selectedEvidenceId"],
                    "additionalProperties": False,
                },
                max_output_tokens=20,
            )

        self.assertEqual(result, {"selectedEvidenceId": "ARCH_LOW"})
        self.assertIs(FakeAsyncClient.request_json["store"], False)


class FootTypeTextApiTests(unittest.TestCase):
    def test_route_is_registered_as_internal_post(self) -> None:
        operation = app.openapi()["paths"]["/api/reports/foot-type-text"]["post"]

        self.assertEqual(
            operation["responses"]["403"]["description"],
            "내부 API 키가 없거나 일치하지 않습니다.",
        )

    def test_authoritative_server_context_returns_generated_text(self) -> None:
        generated = FootTypeReportText(
            type_text=LOW_ARCH_TEXT,
            evidence_id="ARCH_LOW",
            source="OPENAI",
        )
        with (
            patch.object(settings, "feetfit_server_internal_api_key", "service-key"),
            patch(
                "app.api.routes.reports.generate_foot_type_text",
                new=AsyncMock(return_value=generated),
            ) as generator,
        ):
            response = TestClient(app).post(
                "/api/reports/foot-type-text",
                headers={
                    "Authorization": "Bearer server-minted-token",
                    "X-Internal-Api-Key": "service-key",
                },
                json={
                    "measurementSessionId": 30,
                    "measurementStatus": "COMPLETED",
                    "factsHash": FACTS_HASH,
                    "analysis": {
                        "leftPressurePercent": 46.0,
                        "rightPressurePercent": 54.0,
                        "plantarFootprintAnalysisText": "오른발 앞꿈치 압력 집중",
                    },
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {
                "measurementSessionId": 30,
                "factsHash": FACTS_HASH,
                "typeText": LOW_ARCH_TEXT,
                "evidenceId": "ARCH_LOW",
                "source": "OPENAI",
            },
        )
        generator.assert_awaited_once()

    def test_missing_internal_key_is_rejected_before_generation(self) -> None:
        with (
            patch.object(settings, "feetfit_server_internal_api_key", "service-key"),
            patch(
                "app.api.routes.reports.generate_foot_type_text",
                new=AsyncMock(),
            ) as generator,
        ):
            response = TestClient(app).post(
                "/api/reports/foot-type-text",
                headers={"Authorization": "Bearer token"},
                json={
                    "measurementSessionId": 30,
                    "measurementStatus": "COMPLETED",
                    "factsHash": FACTS_HASH,
                    "analysis": {
                        "leftPressurePercent": 46.0,
                        "rightPressurePercent": 54.0,
                    },
                },
            )

        self.assertEqual(response.status_code, 403)
        generator.assert_not_awaited()

    def test_non_completed_or_legacy_care_tips_payload_is_rejected(self) -> None:
        base = {
            "measurementSessionId": 30,
            "measurementStatus": "RUNNING",
            "factsHash": FACTS_HASH,
            "analysis": {
                "leftPressurePercent": 46.0,
                "rightPressurePercent": 54.0,
            },
        }
        with patch.object(settings, "feetfit_server_internal_api_key", "service-key"):
            running = TestClient(app).post(
                "/api/reports/foot-type-text",
                headers={
                    "Authorization": "Bearer token",
                    "X-Internal-Api-Key": "service-key",
                },
                json=base,
            )
            legacy = TestClient(app).post(
                "/api/reports/foot-type-text",
                headers={
                    "Authorization": "Bearer token",
                    "X-Internal-Api-Key": "service-key",
                },
                json={
                    **base,
                    "measurementStatus": "COMPLETED",
                    "careTips": ["1", "2", "3"],
                },
            )

        self.assertEqual(running.status_code, 422)
        self.assertEqual(legacy.status_code, 422)


if __name__ == "__main__":
    unittest.main()
