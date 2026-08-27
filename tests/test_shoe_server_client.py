from __future__ import annotations

from decimal import Decimal
import unittest
from unittest.mock import patch

import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.services.shoe.shoe_server_client import (
    ShoeServerClient,
    ShoeServerClientError,
    ShoeServerConfigurationError,
)


def _api_response(result: dict) -> dict:
    return {
        "isSuccess": True,
        "code": "COMMON200",
        "message": "성공입니다.",
        "result": result,
    }


class _TestPayload(BaseModel):
    marker: str


def _foot_state(*, balance_score: float = 82.0) -> dict:
    return {
        "dailyFootAnalysis": {
            "balanceScore": balance_score,
            "leftPressurePercent": 49.0,
            "rightPressurePercent": 51.0,
            "measuredLeftFootSizeMm": 268.2,
            "measuredRightFootSizeMm": 269.1,
            "leftFootWidthMm": 102.4,
            "rightFootWidthMm": 103.1,
            "avgTemperatureCelsius": 31.2,
            "avgHumidityPercent": 54.0,
            "typeText": "균형형",
        },
        "tinaPedisAnalysis": {
            "fungalSuspicionSafetyScore": 91,
            "skinReactionSafetyScore": 88,
        },
        "halluxValgusAnalysis": {
            "leftToeAngleDegree": 11.2,
            "rightToeAngleDegree": 12.4,
            "riskScore": 30.0,
        },
        "staticPressureAnalyses": [
            {
                "analysisId": 501,
                "footSide": "LEFT",
                "leftPressureRatio": 49.0,
                "rightPressureRatio": 51.0,
                "forefootPressureRatio": 57.0,
                "rearfootPressureRatio": 43.0,
                "centerOfPressureX": 0.48,
                "centerOfPressureY": 0.61,
                "balanceScore": 82.0,
                "balanceStatus": "VERY_GOOD",
                "analysisText": None,
            }
        ],
        "pressureSensorReadings": [
            {
                "readingId": 701,
                "footSide": "LEFT",
                "footRegion": "PRESSURE_0",
                "sensorIndex": 0,
                "pressureValue": 14.3,
                "pressureUnit": 1.0,
                "recordedAt": "2026-08-23T10:00:00",
            }
        ],
    }


def _shoe(shoe_id: int) -> dict:
    return {
        "shoeId": shoe_id,
        "brandName": "Example Brand",
        "shoeName": f"Shoe {shoe_id}",
        "modelCode": f"MODEL-{shoe_id}",
        "musinsaUrl": f"https://www.musinsa.com/products/{shoe_id}",
        "price": 129000,
        "imageUrl": None,
        "overallRating": 4.5,
        "reviewCount": 1,
        "reviews": [
            {
                "reviewId": shoe_id * 10,
                "rating": 5.0,
                "reviewText": "발볼이 편하고 뒤꿈치가 안정적이에요.",
                "source": "MUSINSA",
                "collectedAt": "2026-08-22T12:00:00",
            }
        ],
        "labMeasurements": [
            {
                "measurementId": shoe_id * 100,
                "source": "RUNREPEAT",
                "sourceUrl": f"https://runrepeat.com/shoe-{shoe_id}",
                "sourceBrandName": "Example Brand",
                "sourceShoeName": f"Shoe {shoe_id}",
                "sourceModelCode": f"MODEL-{shoe_id}",
                "testedSize": "US 9",
                "capturedAt": "2026-08-21T09:00:00",
                "parserVersion": "runrepeat-v1",
                "internalLengthMm": None,
                "widthMm": 98.0,
                "toeboxWidthMm": None,
                "toeboxHeightMm": None,
                "insoleThicknessMm": None,
                "heelStackMm": None,
                "forefootStackMm": None,
                "rawMetrics": [
                    {
                        "metricId": shoe_id * 1000,
                        "canonicalCharacteristic": "SHOCK_ABSORPTION",
                        "sourceMetricName": "Shock absorption",
                        "value": "21.30",
                        "averageValue": "19.80",
                        "sourceMinValue": None,
                        "sourceMaxValue": None,
                        "unit": "SA",
                        "testedSize": "US 9",
                        "methodName": "ASTM fixture",
                        "methodVersion": "2026",
                        "location": "heel",
                        "variant": None,
                        "comparisonSampleCount": 120,
                        "comparisonCohort": "road running shoes",
                        "rawValueText": "21.3 SA",
                    }
                ],
            }
        ],
    }


def _context_page(page: int, *, balance_score: float = 82.0) -> dict:
    return {
        "measurementSessionId": 30,
        "userId": 7,
        "measurementStatus": "COMPLETED",
        "footState": _foot_state(balance_score=balance_score),
        "shoes": [_shoe(101 + page)],
        "currentPage": page,
        "totalPages": 2,
        "totalElements": 2,
        "hasNext": page == 0,
    }


def _saved_context(*, user_id: int = 7, measurement_session_id: int = 30) -> dict:
    reasons = []
    for index, reason_type in enumerate(("FOREFOOT", "HEEL", "INSOLE"), start=1):
        reasons.append(
            {
                "reasonType": reason_type,
                "title": f"{reason_type} title",
                "riskLevel": "LOW",
                "reviewSummary": None,
                "reviews": [
                    {
                        "reviewId": 900 + index,
                        "reviewText": f"실제 리뷰 {index}",
                        "source": "MUSINSA",
                    }
                ],
            }
        )
    return {
        "measurementSessionId": measurement_session_id,
        "userId": user_id,
        "shoeId": 101,
        "brandName": "Example Brand",
        "shoeName": "Shoe 101",
        "fitScore": 80.0,
        "pointSummary": None,
        "analyzedAt": "2026-08-23T12:30:00",
        "reasons": reasons,
    }


class ShoeServerClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_runrepeat_characteristics_for_summary_grounding(self) -> None:
        seen_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(
                200,
                json=_api_response(
                    {
                        "shoeId": 101,
                        "summary": "쿠션감은 낮은 편입니다.",
                        "characteristics": [
                            {
                                "type": "CUSHION",
                                "level": "LOW",
                                "value": 40.0,
                                "averageValue": 30.0,
                                "minValue": 10.0,
                                "maxValue": 50.0,
                                "unit": "HA",
                                "testedSize": None,
                            }
                        ],
                    }
                ),
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            result = await client.fetch_shoe_characteristics(101)

        self.assertEqual(result.shoe_id, 101)
        self.assertEqual(result.characteristics[0].type, "CUSHION")
        self.assertEqual(result.characteristics[0].value, Decimal("40.0"))
        assert seen_request is not None
        self.assertEqual(seen_request.headers["Authorization"], "Bearer token")
        self.assertEqual(seen_request.headers["X-Internal-Api-Key"], "test-internal-key")

    async def test_fetches_every_page_and_preserves_raw_decimal_values(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            page = int(request.url.params["page"])
            return httpx.Response(200, json=_api_response(_context_page(page)), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer forwarded-token",
                internal_api_key="test-internal-key",
                http_client=http_client,
            )
            context = await client.fetch_recommendation_context(30)

        self.assertEqual([shoe.id for shoe in context.shoes], [101, 102])
        raw_value = context.shoes[0].lab_measurements[0].raw_metrics[0].value
        self.assertEqual(raw_value, Decimal("21.30"))
        self.assertEqual(len(requests), 2)
        self.assertEqual([request.url.params["page"] for request in requests], ["0", "1"])
        self.assertEqual([request.url.params["size"] for request in requests], ["100", "100"])
        self.assertTrue(all(request.headers["Authorization"] == "Bearer forwarded-token" for request in requests))
        self.assertTrue(
            all(request.headers["X-Internal-Api-Key"] == "test-internal-key" for request in requests)
        )

    async def test_rejects_metadata_that_changes_between_pages(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params["page"])
            result = _context_page(page, balance_score=82.0 if page == 0 else 75.0)
            return httpx.Response(200, json=_api_response(result), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            with self.assertRaisesRegex(ShoeServerClientError, "changed session metadata"):
                await client.fetch_recommendation_context(30)

    async def test_rejects_page_that_claims_to_end_before_total_pages(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            result = _context_page(0)
            result["hasNext"] = False
            result["totalElements"] = 1
            return httpx.Response(200, json=_api_response(result), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            with self.assertRaisesRegex(ShoeServerClientError, "inconsistent pagination"):
                await client.fetch_recommendation_context(30)

    async def test_rejects_non_musinsa_review_from_internal_contract(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            result = _context_page(0)
            result["totalPages"] = 1
            result["totalElements"] = 1
            result["hasNext"] = False
            result["shoes"][0]["reviews"][0]["source"] = "RUNREPEAT"
            return httpx.Response(200, json=_api_response(result), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            with self.assertRaisesRegex(ShoeServerClientError, "invalid ApiResponse contract"):
                await client.fetch_recommendation_context(30)

    async def test_recommendation_context_preserves_non_2xx_error_envelope(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "isSuccess": False,
                    "code": "MEASUREMENT4004",
                    "message": "완료된 측정 세션이 아닙니다.",
                    "result": None,
                },
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            with self.assertRaises(ShoeServerClientError) as raised:
                await client.fetch_recommendation_context(30)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("MEASUREMENT4004", str(raised.exception))
        self.assertIn("완료된 측정 세션이 아닙니다.", str(raised.exception))

    async def test_recommendation_context_uses_generic_non_2xx_error_for_invalid_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, content=b"not-json", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            with self.assertRaises(ShoeServerClientError) as raised:
                await client.fetch_recommendation_context(30)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(str(raised.exception), "Feetfit_Server returned HTTP 400.")

    async def test_summary_context_uses_jwt_identity_without_user_id_query(self) -> None:
        seen_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(200, json=_api_response(_saved_context()), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            saved = await client.fetch_saved_recommendation(
                measurement_session_id=30, shoe_id=101
            )

        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved.reasons[0].reviews[0].id, 901)
        assert seen_request is not None
        self.assertNotIn("userId", seen_request.url.params)
        self.assertEqual(seen_request.url.params["measurementSessionId"], "30")
        self.assertEqual(seen_request.headers["Authorization"], "Bearer token")
        self.assertEqual(seen_request.headers["X-Internal-Api-Key"], "test-internal-key")

    async def test_summary_context_rejects_measurement_session_mismatch(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_api_response(_saved_context(measurement_session_id=31)),
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            with self.assertRaisesRegex(ShoeServerClientError, "does not match"):
                await client.fetch_saved_recommendation(
                    measurement_session_id=30, shoe_id=101
                )

    async def test_missing_saved_recommendation_returns_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={
                    "isSuccess": False,
                    "code": "SHOE4005",
                    "message": "저장된 신발 추천을 찾을 수 없습니다.",
                    "result": None,
                },
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            self.assertIsNone(
                await client.fetch_saved_recommendation(
                    measurement_session_id=30, shoe_id=101
                )
            )

    async def test_other_404_from_summary_context_is_not_treated_as_missing_recommendation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={
                    "isSuccess": False,
                    "code": "SHOE4001",
                    "message": "신발을 찾을 수 없습니다.",
                    "result": None,
                },
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer token", internal_api_key="test-internal-key", http_client=http_client
            )
            with self.assertRaisesRegex(ShoeServerClientError, "SHOE4001") as raised:
                await client.fetch_saved_recommendation(
                    measurement_session_id=30, shoe_id=101
                )

        self.assertEqual(raised.exception.status_code, 404)

    async def test_write_requests_forward_bearer_and_internal_api_key(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=_api_response({}), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = ShoeServerClient(
                "Bearer forwarded-token",
                internal_api_key="test-internal-key",
                http_client=http_client,
            )
            payload = _TestPayload(marker="contract")
            await client.forward_recommendations(payload)
            await client.save_summary(101, payload)

        self.assertEqual([request.method for request in requests], ["POST", "POST"])
        self.assertTrue(all(request.headers["Authorization"] == "Bearer forwarded-token" for request in requests))
        self.assertTrue(
            all(request.headers["X-Internal-Api-Key"] == "test-internal-key" for request in requests)
        )

    def test_blank_internal_api_key_fails_before_any_request(self) -> None:
        with patch.object(settings, "feetfit_server_internal_api_key", "   "):
            with self.assertRaisesRegex(
                ShoeServerConfigurationError,
                "FEETFIT_SERVER_INTERNAL_API_KEY must be configured",
            ):
                ShoeServerClient("Bearer token")


if __name__ == "__main__":
    unittest.main()
