from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import settings
from app.api.routes import shoes as shoe_routes
from app.api.routes.shoes import _generate_and_save_summary
from app.main import app
from app.schemas.shoe_fit_comment import ShoeFitSummary
from app.schemas.shoe_server import ServerSavedRecommendation, ServerShoeCharacteristics
from app.schemas.shoes import (
    ShoeRecommendationForwardRequest,
    ShoeRecommendationItemPayload,
    ShoeRecommendationReasonPayload,
    ShoeSummaryForwardRequest,
    ShoeSummaryReasonPayload,
)
from app.services.shoe.shoe_recommendation import (
    ReasonFacts,
    ShoeFacts,
    ShoeRecommendationBatch,
    ShoeRecommendationBusyError,
)
from app.services.shoe.shoe_server_client import (
    ShoeServerClientError,
    ShoeServerConfigurationError,
)


def _recommendation_reasons() -> list[ShoeRecommendationReasonPayload]:
    return [
        ShoeRecommendationReasonPayload(
            reason_type=reason_type,
            title=f"{reason_type} title",
            risk_level="LOW",
            review_ids=[index],
        )
        for index, reason_type in enumerate(("FOREFOOT", "HEEL", "INSOLE"), start=1)
    ]


def _saved_recommendation() -> ServerSavedRecommendation:
    return ServerSavedRecommendation.model_validate(
        {
            "measurementSessionId": 30,
            "userId": 7,
            "shoeId": 101,
            "brandName": "Example Brand",
            "shoeName": "Example Shoe",
            "fitScore": 80.0,
            "pointSummary": None,
            "analyzedAt": "2026-08-23T12:30:00",
            "reasons": [
                {
                    "reasonType": reason_type,
                    "title": f"{reason_type} title",
                    "riskLevel": "LOW",
                    "reviewSummary": None,
                    "reviews": [
                        {
                            "reviewId": index,
                            "reviewText": f"서버에 저장된 실제 리뷰 {index}",
                            "source": "MUSINSA",
                        }
                    ],
                }
                for index, reason_type in enumerate(("FOREFOOT", "HEEL", "INSOLE"), start=1)
            ],
        }
    )


def _shoe_characteristics() -> ServerShoeCharacteristics:
    return ServerShoeCharacteristics.model_validate(
        {
            "shoeId": 101,
            "summary": "쿠션감은 낮은 편입니다.",
            "characteristics": [
                {
                    "type": "CUSHION",
                    "level": "LOW",
                    "value": "40.0",
                    "averageValue": "30.0",
                    "minValue": "10.0",
                    "maxValue": "50.0",
                    "unit": "HA",
                    "testedSize": None,
                }
            ],
        }
    )


class ShoePayloadContractTests(unittest.TestCase):
    def test_first_stage_payload_serializes_null_summaries_as_pending(self) -> None:
        payload = ShoeRecommendationForwardRequest(
            measurement_session_id=30,
            recommendations=[
                ShoeRecommendationItemPayload(
                    shoe_id=101,
                    fit_score=80.0,
                    reasons=_recommendation_reasons(),
                )
            ],
        )

        self.assertEqual(
            payload.model_dump(by_alias=True),
            {
                "measurementSessionId": 30,
                "recommendations": [
                    {
                        "shoeId": 101,
                        "fitScore": 80.0,
                        "pointSummary": None,
                        "reasons": [
                            {
                                "reasonType": "FOREFOOT",
                                "title": "FOREFOOT title",
                                "riskLevel": "LOW",
                                "reviewSummary": None,
                                "reviewIds": [1],
                            },
                            {
                                "reasonType": "HEEL",
                                "title": "HEEL title",
                                "riskLevel": "LOW",
                                "reviewSummary": None,
                                "reviewIds": [2],
                            },
                            {
                                "reasonType": "INSOLE",
                                "title": "INSOLE title",
                                "riskLevel": "LOW",
                                "reviewSummary": None,
                                "reviewIds": [3],
                            },
                        ],
                    }
                ],
            },
        )

    def test_first_stage_payload_requires_all_three_unique_reason_types(self) -> None:
        duplicate_reasons = _recommendation_reasons()
        duplicate_reasons[2] = ShoeRecommendationReasonPayload(
            reason_type="HEEL",
            title="duplicate",
            risk_level="MEDIUM",
            review_ids=[],
        )
        with self.assertRaises(ValidationError):
            ShoeRecommendationItemPayload(
                shoe_id=101,
                fit_score=80.0,
                reasons=duplicate_reasons,
            )

    def test_first_stage_payload_limits_evidence_to_three_unique_reviews(self) -> None:
        with self.assertRaises(ValidationError):
            ShoeRecommendationReasonPayload(
                reason_type="FOREFOOT",
                title="title",
                risk_level="LOW",
                review_ids=[1, 2, 3, 4],
            )
        with self.assertRaises(ValidationError):
            ShoeRecommendationReasonPayload(
                reason_type="FOREFOOT",
                title="title",
                risk_level="LOW",
                review_ids=[1, 1],
            )

    def test_first_stage_payload_rejects_an_empty_recommendation_batch(self) -> None:
        with self.assertRaises(ValidationError):
            ShoeRecommendationForwardRequest(
                measurement_session_id=30,
                recommendations=[],
            )

    def test_first_stage_payload_rejects_duplicate_shoe_ids(self) -> None:
        item = ShoeRecommendationItemPayload(
            shoe_id=101,
            fit_score=80.0,
            reasons=_recommendation_reasons(),
        )
        with self.assertRaisesRegex(ValidationError, "duplicate shoeId"):
            ShoeRecommendationForwardRequest(
                measurement_session_id=30,
                recommendations=[item, item],
            )

    def test_summary_callback_contract_contains_only_generated_text(self) -> None:
        payload = ShoeSummaryForwardRequest(
            measurement_session_id=30,
            point_summary="요약",
            reasons=[
                ShoeSummaryReasonPayload(
                    reason_type=reason_type,
                    review_summary=f"{reason_type} 요약",
                    review_ids=[index],
                )
                for index, reason_type in enumerate(
                    ("FOREFOOT", "HEEL", "INSOLE"), start=1
                )
            ],
        )
        self.assertEqual(
            payload.model_dump(by_alias=True),
            {
                "measurementSessionId": 30,
                "pointSummary": "요약",
                "reasons": [
                    {"reasonType": "FOREFOOT", "reviewSummary": "FOREFOOT 요약", "reviewIds": [1]},
                    {"reasonType": "HEEL", "reviewSummary": "HEEL 요약", "reviewIds": [2]},
                    {"reasonType": "INSOLE", "reviewSummary": "INSOLE 요약", "reviewIds": [3]},
                ],
            },
        )


class InternalTriggerSecurityTests(unittest.TestCase):
    def test_heavy_routes_reject_missing_or_wrong_internal_key_without_echo(self) -> None:
        routes = (
            ("/api/reports/shoe-recommendations", {"measurementSessionId": 30}),
            (
                "/api/shoes/summaries",
                {"shoeId": 101, "measurementSessionId": 30},
            ),
        )
        with patch.object(
            settings, "feetfit_server_internal_api_key", "super-secret-value"
        ):
            for route, body in routes:
                for supplied in (None, "wrong-key"):
                    headers = {"Authorization": "Bearer forwarded-token"}
                    if supplied is not None:
                        headers["X-Internal-Api-Key"] = supplied
                    with self.subTest(route=route, supplied=supplied):
                        response = TestClient(app).post(route, headers=headers, json=body)
                        self.assertEqual(response.status_code, 403)
                        self.assertNotIn("super-secret-value", response.text)
                        self.assertNotIn("wrong-key", response.text)


class ShoeRecommendationRouteContractTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(
            settings, "feetfit_server_internal_api_key", "test-internal-key"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_route_reads_server_context_then_forwards_pending_payload(self) -> None:
        server_client = MagicMock()
        server_client.fetch_recommendation_context = AsyncMock(return_value=object())
        upstream_request = httpx.Request("POST", "http://server/api/shoes/recommendations")
        server_client.forward_recommendations = AsyncMock(
            return_value=httpx.Response(200, json={"saved": True}, request=upstream_request)
        )
        batch = ShoeRecommendationBatch(
            user_id=7,
            measurement_session_id=30,
            items=[
                ShoeFacts(
                    shoe_id=101,
                    fit_score=80.0,
                    reasons=[
                        ReasonFacts(
                            reason_type=reason_type,
                            score=80.0,
                            title="title",
                            risk_level="LOW",
                            review_ids=[],
                        )
                        for reason_type in ("FOREFOOT", "HEEL", "INSOLE")
                    ],
                )
            ],
        )

        with (
            patch("app.api.routes.reports.ShoeServerClient", return_value=server_client) as client_type,
            patch("app.api.routes.reports.compute_shoe_recommendations", return_value=batch) as compute,
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 200)
        client_type.assert_called_once_with("Bearer forwarded-token")
        server_client.fetch_recommendation_context.assert_awaited_once_with(30)
        compute.assert_called_once()
        forwarded = server_client.forward_recommendations.await_args.args[0]
        self.assertIsNone(forwarded.recommendations[0].point_summary)
        self.assertTrue(all(reason.review_summary is None for reason in forwarded.recommendations[0].reasons))

    def test_internal_context_failure_does_not_fall_back_to_db_or_compute(self) -> None:
        server_client = MagicMock()
        server_client.fetch_recommendation_context = AsyncMock(
            side_effect=ShoeServerClientError("contract unavailable")
        )

        with (
            patch("app.api.routes.reports.ShoeServerClient", return_value=server_client),
            patch("app.api.routes.reports.compute_shoe_recommendations") as compute,
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 502)
        compute.assert_not_called()

    def test_local_server_configuration_failure_is_not_reported_as_bad_gateway(self) -> None:
        with (
            patch(
                "app.api.routes.reports.ShoeServerClient",
                side_effect=ShoeServerConfigurationError("internal API key is missing"),
            ),
            patch("app.api.routes.reports.compute_shoe_recommendations") as compute,
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "internal API key is missing")
        compute.assert_not_called()

    def test_internal_context_preserves_server_measurement_not_completed_400(self) -> None:
        server_client = MagicMock()
        server_client.fetch_recommendation_context = AsyncMock(
            side_effect=ShoeServerClientError(
                "measurement session is not completed",
                status_code=400,
            )
        )

        with (
            patch("app.api.routes.reports.ShoeServerClient", return_value=server_client),
            patch("app.api.routes.reports.compute_shoe_recommendations") as compute,
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "measurement session is not completed")
        compute.assert_not_called()

    def test_busy_recommendation_runtime_returns_conflict_not_generic_error(self) -> None:
        server_client = MagicMock()
        server_client.fetch_recommendation_context = AsyncMock(return_value=object())
        with (
            patch("app.api.routes.reports.ShoeServerClient", return_value=server_client),
            patch(
                "app.api.routes.reports.compute_shoe_recommendations",
                side_effect=ShoeRecommendationBusyError("BGE runtime is busy"),
            ),
        ):
            response = TestClient(app).post(
                "/api/reports/shoe-recommendations",
                headers={
                    "Authorization": "Bearer forwarded-token",
                    "X-Internal-Api-Key": "test-internal-key",
                },
                json={"measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("BGE runtime is busy", response.json()["detail"])


class ShoeSummaryRouteContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch.object(
            settings, "feetfit_server_internal_api_key", "test-internal-key"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        shoe_routes._summary_inflight.clear()
        self.addCleanup(shoe_routes._summary_inflight.clear)

    async def test_summary_trigger_requires_and_forwards_measurement_session(self) -> None:
        with patch(
            "app.api.routes.shoes._generate_and_save_summary",
            new=AsyncMock(return_value=None),
        ) as task:
            response = TestClient(app).post(
                "/api/shoes/summaries",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"shoeId": 101, "measurementSessionId": 30},
            )

        self.assertEqual(response.status_code, 202)
        task.assert_awaited_once_with(101, 30, "Bearer forwarded-token")

        missing_session = TestClient(app).post(
            "/api/shoes/summaries",
            headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
            json={"shoeId": 101},
        )
        self.assertEqual(missing_session.status_code, 422)

    async def test_duplicate_summary_trigger_is_deduplicated_by_session_and_shoe(self) -> None:
        with patch(
            "app.api.routes.shoes._generate_and_save_summary",
            new=AsyncMock(return_value=None),
        ) as task:
            first = TestClient(app).post(
                "/api/shoes/summaries",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"shoeId": 101, "measurementSessionId": 30},
            )
            second = TestClient(app).post(
                "/api/shoes/summaries",
                headers={"Authorization": "Bearer forwarded-token", "X-Internal-Api-Key": "test-internal-key"},
                json={"shoeId": 101, "measurementSessionId": 30},
            )

        self.assertEqual(first.json(), {"accepted": True, "deduplicated": False})
        self.assertEqual(second.json(), {"accepted": True, "deduplicated": True})
        task.assert_awaited_once()

    async def test_summary_uses_saved_server_context_and_callback_contract(self) -> None:
        server_client = MagicMock()
        server_client.fetch_saved_recommendation = AsyncMock(return_value=_saved_recommendation())
        server_client.fetch_shoe_characteristics = AsyncMock(return_value=_shoe_characteristics())
        server_client.save_summary = AsyncMock(return_value=None)
        generated = ShoeFitSummary(
            point_summary="종합 요약",
            forefoot_summary="발볼 요약",
            heel_summary="뒤꿈치 요약",
            insole_summary="깔창 요약",
            forefoot_review_ids=[1],
            heel_review_ids=[2],
            insole_review_ids=[3],
        )

        with (
            patch("app.api.routes.shoes.ShoeServerClient", return_value=server_client) as client_type,
            patch("app.api.routes.shoes.generate_shoe_summaries", new=AsyncMock(return_value=generated)) as llm,
        ):
            await _generate_and_save_summary(101, 30, "Bearer forwarded-token")

        client_type.assert_called_once_with("Bearer forwarded-token")
        server_client.fetch_saved_recommendation.assert_awaited_once_with(30, 101)
        server_client.fetch_shoe_characteristics.assert_awaited_once_with(101)
        llm.assert_awaited_once()
        self.assertEqual(llm.await_args.kwargs["characteristics"], _shoe_characteristics())
        llm_reasons = llm.await_args.kwargs["reasons"]
        self.assertEqual(llm_reasons[0].review_ids, [1])
        self.assertEqual(llm_reasons[0].review_texts, ["서버에 저장된 실제 리뷰 1"])
        saved_payload = server_client.save_summary.await_args.args[1]
        self.assertEqual(
            saved_payload.model_dump(by_alias=True),
            {
                "measurementSessionId": 30,
                "pointSummary": "종합 요약",
                "reasons": [
                    {"reasonType": "FOREFOOT", "reviewSummary": "발볼 요약", "reviewIds": [1]},
                    {"reasonType": "HEEL", "reviewSummary": "뒤꿈치 요약", "reviewIds": [2]},
                    {"reasonType": "INSOLE", "reviewSummary": "깔창 요약", "reviewIds": [3]},
                ],
            },
        )

    async def test_unexpected_background_failure_is_logged_and_dropped(self) -> None:
        server_client = MagicMock()
        server_client.fetch_saved_recommendation = AsyncMock(return_value=_saved_recommendation())
        server_client.fetch_shoe_characteristics = AsyncMock(return_value=_shoe_characteristics())
        server_client.save_summary = AsyncMock(return_value=None)

        with (
            patch("app.api.routes.shoes.ShoeServerClient", return_value=server_client),
            patch(
                "app.api.routes.shoes.generate_shoe_summaries",
                new=AsyncMock(side_effect=RuntimeError("unexpected Ollama failure")),
            ),
            patch("app.api.routes.shoes.logger.warning") as warning,
        ):
            await _generate_and_save_summary(101, 30, "Bearer forwarded-token")

        server_client.save_summary.assert_not_awaited()
        self.assertTrue(
            any(
                "예상하지 못한 오류" in call.args[0]
                for call in warning.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
