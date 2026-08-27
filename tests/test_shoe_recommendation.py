from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from app.schemas.shoe_server import ServerRecommendationContext
from app.core.config import settings
from app.services.shoe.shoe_fit_policy import (
    CLINICAL_VALIDATION_STATUS,
    POLICY_CLASSIFICATION,
    risk_level_from_score,
    score_shoe_fit,
)
from app.services.shoe.shoe_recommendation import (
    ShoeRecommendationBusyError,
    ShoeRecommendationError,
    compute_shoe_recommendations,
)
from app.services.shoe import shoe_recommendation
from app.services.shoe.shoe_feature_rules import (
    REASON_TYPES,
    build_need_query,
    derive_foot_need,
)


def _raw_metric(
    metric_id: int,
    characteristic: str,
    value: float,
    average: float,
    minimum: float,
    maximum: float,
    *,
    location: str | None = None,
    variant: str | None = None,
) -> dict:
    return {
        "metricId": metric_id,
        "canonicalCharacteristic": characteristic,
        "sourceMetricName": characteristic,
        "value": value,
        "averageValue": average,
        "sourceMinValue": minimum,
        "sourceMaxValue": maximum,
        "unit": "score",
        "testedSize": "US 9",
        "methodName": "test",
        "methodVersion": "new method",
        "location": location,
        "variant": variant,
        "comparisonSampleCount": 100,
        "comparisonCohort": "running shoes",
        "rawValueText": str(value),
    }


def _shoe(shoe_id: int, *, with_reviews: bool = True, width_mm: float = 108.0) -> dict:
    reviews = []
    if with_reviews:
        reviews = [
            {
                "reviewId": shoe_id * 10,
                "rating": 5.0,
                "reviewText": "발볼이 편하고 뒤꿈치가 안정적이에요. 쿠션과 통기성이 좋아요.",
                "source": "MUSINSA",
                "collectedAt": None,
            }
        ]
    return {
        "shoeId": shoe_id,
        "brandName": "Example Brand",
        "shoeName": f"Example Shoe {shoe_id}",
        "modelCode": f"EX-{shoe_id}",
        "musinsaUrl": f"https://example.com/shoes/{shoe_id}",
        "price": 100000,
        "imageUrl": None,
        "overallRating": 4.5,
        "reviewCount": len(reviews),
        "reviews": reviews,
        "labMeasurements": [
            {
                "measurementId": shoe_id * 100,
                "source": "RUNREPEAT",
                "sourceUrl": f"https://runrepeat.com/shoe-{shoe_id}",
                "sourceBrandName": "Example Brand",
                "sourceShoeName": f"Example Shoe {shoe_id}",
                "sourceModelCode": None,
                "testedSize": "US 9",
                "capturedAt": "2026-08-26T01:00:00",
                "parserVersion": "test-v1",
                "internalLengthMm": 270.0,
                "widthMm": width_mm,
                "toeboxWidthMm": 78.0,
                "toeboxHeightMm": 28.0,
                "insoleThicknessMm": 5.0,
                "heelStackMm": 35.0,
                "forefootStackMm": 27.0,
                "rawMetrics": [
                    _raw_metric(shoe_id * 1000 + 1, "WIDTH_SPACE", width_mm, 98, 88, 112),
                    _raw_metric(
                        shoe_id * 1000 + 2,
                        "TOEBOX_SPACE",
                        78,
                        74,
                        66,
                        84,
                        location="big toe",
                        variant="width",
                    ),
                    _raw_metric(
                        shoe_id * 1000 + 3,
                        "HEEL_HOLD",
                        4,
                        3,
                        1,
                        5,
                        location="HEEL",
                    ),
                    _raw_metric(
                        shoe_id * 1000 + 4,
                        "SHOCK_ABSORPTION",
                        150,
                        130,
                        60,
                        185,
                        location="HEEL",
                    ),
                    _raw_metric(
                        shoe_id * 1000 + 5,
                        "CUSHION",
                        25,
                        35,
                        20,
                        55,
                        variant="primary",
                    ),
                    _raw_metric(shoe_id * 1000 + 6, "BREATHABILITY", 5, 3, 1, 5),
                ],
            }
        ],
    }


def _context_dict(*, include_daily_analysis: bool = True) -> dict:
    return {
        "measurementSessionId": 30,
        "userId": 7,
        "measurementStatus": "COMPLETED",
        "footState": {
            "dailyFootAnalysis": (
                {
                    "balanceScore": 70.0,
                    "leftPressurePercent": 42.0,
                    "rightPressurePercent": 58.0,
                    "measuredLeftFootSizeMm": 268.0,
                    "measuredRightFootSizeMm": 270.0,
                    "leftFootWidthMm": 108.0,
                    "rightFootWidthMm": 110.0,
                    "avgTemperatureCelsius": 28.0,
                    "avgHumidityPercent": 70.0,
                    "typeText": "test",
                }
                if include_daily_analysis
                else None
            ),
            "tinaPedisAnalysis": None,
            "halluxValgusAnalysis": None,
            "staticPressureAnalyses": [],
            "pressureSensorReadings": [],
        },
        "shoes": [_shoe(101)],
    }


def _context(*, include_daily_analysis: bool = True) -> ServerRecommendationContext:
    return ServerRecommendationContext.model_validate(
        _context_dict(include_daily_analysis=include_daily_analysis)
    )


def _embedding_patches():
    def rank_in_insertion_order(_query_vector, embeddings, _top_k):
        return [(key, 0.9) for key in embeddings]

    return (
        patch(
            "app.services.shoe.shoe_recommendation.embed_texts",
            return_value=[object(), object(), object()],
        ),
        patch(
            "app.services.shoe.shoe_recommendation.get_or_embed_texts",
            side_effect=lambda texts: {key: object() for key in texts},
        ),
        patch(
            "app.services.shoe.shoe_recommendation.rank_by_similarity",
            side_effect=rank_in_insertion_order,
        ),
    )


class ShoeRecommendationContextTests(unittest.TestCase):
    def test_type_text_is_not_a_scoring_or_review_shortlist_input(self) -> None:
        before_raw = _context_dict()
        after_raw = copy.deepcopy(before_raw)
        after_raw["footState"]["dailyFootAnalysis"]["typeText"] = (
            "발의 아치가 낮아 발바닥이 넓게 닿는 편이에요."
        )
        before = ServerRecommendationContext.model_validate(before_raw)
        after = ServerRecommendationContext.model_validate(after_raw)

        before_areas = score_shoe_fit(before.shoes[0], before.foot_state)
        after_areas = score_shoe_fit(after.shoes[0], after.foot_state)

        self.assertEqual(before_areas, after_areas)
        self.assertEqual(
            derive_foot_need(before.foot_state),
            derive_foot_need(after.foot_state),
        )

    def test_quantitative_scoring_uses_runrepeat_and_keeps_three_areas(self) -> None:
        with _embedding_patches()[0], _embedding_patches()[1], _embedding_patches()[2]:
            batch = compute_shoe_recommendations(_context())

        self.assertEqual(batch.user_id, 7)
        self.assertEqual(batch.measurement_session_id, 30)
        self.assertEqual([item.shoe_id for item in batch.items], [101])
        self.assertGreaterEqual(batch.items[0].fit_score, 0)
        self.assertLessEqual(batch.items[0].fit_score, 100)
        self.assertEqual(
            [reason.reason_type for reason in batch.items[0].reasons],
            ["FOREFOOT", "HEEL", "INSOLE"],
        )
        for reason in batch.items[0].reasons:
            self.assertEqual(reason.risk_level, risk_level_from_score(reason.score))
            self.assertEqual(reason.review_ids, [1010])

    def test_reviews_do_not_change_quantitative_fit_or_risk(self) -> None:
        positive = _context()
        negative = _context()
        negative.shoes[0].reviews[0].review_text = "발볼이 좁고 뒤꿈치가 아프며 쿠션도 불편해요."
        patches = _embedding_patches()
        with patches[0], patches[1], patches[2]:
            positive_result = compute_shoe_recommendations(positive)
        patches = _embedding_patches()
        with patches[0], patches[1], patches[2]:
            negative_result = compute_shoe_recommendations(negative)

        self.assertEqual(positive_result.items[0].fit_score, negative_result.items[0].fit_score)
        self.assertEqual(
            [reason.risk_level for reason in positive_result.items[0].reasons],
            [reason.risk_level for reason in negative_result.items[0].reasons],
        )

    def test_reviewless_shoe_is_still_in_full_batch_with_empty_evidence(self) -> None:
        raw = _context_dict()
        raw["shoes"] = [_shoe(101, with_reviews=False), _shoe(102, with_reviews=False)]
        context = ServerRecommendationContext.model_validate(raw)
        with patch("app.services.shoe.shoe_recommendation.embed_texts") as embed_texts:
            batch = compute_shoe_recommendations(context)

        self.assertEqual([item.shoe_id for item in batch.items], [101, 102])
        self.assertTrue(
            all(not reason.review_ids for item in batch.items for reason in item.reasons)
        )
        embed_texts.assert_not_called()

    def test_full_338_shoe_contract_preserves_coverage_order_and_three_reasons(self) -> None:
        raw = _context_dict()
        raw["shoes"] = [
            _shoe(shoe_id, with_reviews=(shoe_id % 2 == 0))
            for shoe_id in range(1, 339)
        ]
        context = ServerRecommendationContext.model_validate(raw)
        patches = _embedding_patches()
        with patches[0], patches[1], patches[2]:
            batch = compute_shoe_recommendations(context)

        expected_ids = list(range(1, 339))
        actual_ids = [item.shoe_id for item in batch.items]
        self.assertEqual(actual_ids, expected_ids)
        self.assertEqual(len(actual_ids), len(set(actual_ids)))
        self.assertTrue(all(0 <= item.fit_score <= 100 for item in batch.items))
        self.assertTrue(
            all(
                [reason.reason_type for reason in item.reasons]
                == ["FOREFOOT", "HEEL", "INSOLE"]
                for item in batch.items
            )
        )
        reviewless = next(item for item in batch.items if item.shoe_id == 1)
        self.assertTrue(all(reason.review_ids == [] for reason in reviewless.reasons))

    def test_semantically_unrelated_reviews_are_not_force_filled(self) -> None:
        context = _context()
        with (
            patch(
                "app.services.shoe.shoe_recommendation.embed_texts",
                return_value=[object(), object(), object()],
            ),
            patch(
                "app.services.shoe.shoe_recommendation.get_or_embed_texts",
                side_effect=lambda texts: {key: object() for key in texts},
            ),
            patch(
                "app.services.shoe.shoe_recommendation.rank_by_similarity",
                side_effect=lambda _query, embeddings, _top_k: [
                    (key, 0.1) for key in embeddings
                ],
            ),
        ):
            batch = compute_shoe_recommendations(context)
        self.assertTrue(all(not reason.review_ids for reason in batch.items[0].reasons))

    def test_semantic_queries_cover_positive_and_negative_wear_experiences(self) -> None:
        context = _context()
        need = derive_foot_need(context.foot_state)
        queries = {
            reason_type: build_need_query(reason_type, need)
            for reason_type in REASON_TYPES
        }

        self.assertIn("넓고 편안", queries["FOREFOOT"])
        self.assertIn("좁아 아프", queries["FOREFOOT"])
        self.assertIn("안정적으로", queries["HEEL"])
        self.assertIn("벗겨지고", queries["HEEL"])
        self.assertIn("통풍이 좋", queries["INSOLE"])
        self.assertIn("답답하며", queries["INSOLE"])

    def test_relevant_negative_wear_review_can_be_selected_as_evidence(self) -> None:
        raw = _context_dict()
        raw["shoes"][0]["reviews"] = [
            {
                "reviewId": 5001,
                "rating": 5.0,
                "reviewText": "배송이 빠르고 포장이 깔끔합니다.",
                "source": "MUSINSA",
                "collectedAt": None,
            },
            {
                "reviewId": 5002,
                "rating": 2.0,
                "reviewText": "발볼이 좁아 아프고 뒤꿈치가 벗겨져 까지며 깔창은 답답하고 덥습니다.",
                "source": "MUSINSA",
                "collectedAt": None,
            },
        ]
        raw["shoes"][0]["reviewCount"] = 2
        context = ServerRecommendationContext.model_validate(raw)

        def rank_negative(_query, embeddings, _top_k):
            negative_key = next(key for key in embeddings if ":5002:" in key)
            return [(negative_key, 0.9)]

        with (
            patch(
                "app.services.shoe.shoe_recommendation.embed_texts",
                return_value=[object(), object(), object()],
            ),
            patch(
                "app.services.shoe.shoe_recommendation.get_or_embed_texts",
                side_effect=lambda texts: {key: object() for key in texts},
            ),
            patch(
                "app.services.shoe.shoe_recommendation.rank_by_similarity",
                side_effect=rank_negative,
            ),
        ):
            batch = compute_shoe_recommendations(context)

        self.assertTrue(
            all(reason.review_ids == [5002] for reason in batch.items[0].reasons)
        )

    def test_semantic_ranking_sees_relevant_review_after_legacy_leading_cutoff(self) -> None:
        raw = _context_dict()
        raw["shoes"][0]["reviews"] = [
            {
                "reviewId": 20_000 + index,
                "rating": 5.0,
                "reviewText": (
                    "일반적인 착화 후기입니다."
                    if index < 120
                    else "발볼과 뒤꿈치와 깔창 적합성을 자세히 설명한 관련 후기입니다."
                ),
                "source": "MUSINSA",
                "collectedAt": None,
            }
            for index in range(121)
        ]
        raw["shoes"][0]["reviewCount"] = 121
        context = ServerRecommendationContext.model_validate(raw)

        with (
            patch(
                "app.services.shoe.shoe_recommendation.embed_texts",
                return_value=[object(), object(), object()],
            ),
            patch(
                "app.services.shoe.shoe_recommendation.get_or_embed_texts",
                side_effect=lambda texts: {key: object() for key in texts},
            ),
            patch(
                "app.services.shoe.shoe_recommendation.rank_by_similarity",
                side_effect=lambda _query, embeddings, _top_k: [
                    (list(embeddings)[-1], 0.99)
                ],
            ),
        ):
            batch = compute_shoe_recommendations(context)

        self.assertTrue(
            all(reason.review_ids == [20_120] for reason in batch.items[0].reasons)
        )

    def test_duplicate_review_bodies_with_distinct_ids_are_selected_once(self) -> None:
        raw = _context_dict()
        raw["shoes"][0]["reviews"] = [
            {
                "reviewId": 5002,
                "rating": 5.0,
                "reviewText": "발볼이  편하고 뒤꿈치가 안정적이에요.\n쿠션이 좋아요.",
                "source": "MUSINSA",
                "collectedAt": None,
            },
            {
                "reviewId": 5001,
                "rating": 5.0,
                "reviewText": "\u200b발볼이 편하고 뒤꿈치가 안정적이에요. 쿠션이 좋아요.",
                "source": "MUSINSA",
                "collectedAt": None,
            },
            {
                "reviewId": 5003,
                "rating": 4.0,
                "reviewText": "앞코가 여유롭고 발목이 편해요. 깔창도 푹신해요.",
                "source": "MUSINSA",
                "collectedAt": None,
            },
        ]
        raw["shoes"][0]["reviewCount"] = 3
        context = ServerRecommendationContext.model_validate(raw)
        patches = _embedding_patches()

        with patches[0], patches[1], patches[2]:
            batch = compute_shoe_recommendations(context)

        self.assertTrue(
            all(reason.review_ids == [5001, 5003] for reason in batch.items[0].reasons)
        )

    def test_wider_lab_shoe_scores_better_for_wide_user(self) -> None:
        raw = _context_dict()
        raw["footState"]["dailyFootAnalysis"]["leftFootWidthMm"] = 114.0
        raw["footState"]["dailyFootAnalysis"]["rightFootWidthMm"] = 114.0
        raw["shoes"] = [
            _shoe(101, with_reviews=False, width_mm=98.0),
            _shoe(102, with_reviews=False, width_mm=120.0),
        ]
        batch = compute_shoe_recommendations(ServerRecommendationContext.model_validate(raw))
        narrow, wide = batch.items
        narrow_forefoot = next(r for r in narrow.reasons if r.reason_type == "FOREFOOT")
        wide_forefoot = next(r for r in wide.reasons if r.reason_type == "FOREFOOT")
        self.assertGreater(wide_forefoot.score, narrow_forefoot.score)

    def test_hallux_signal_changes_forefoot_only(self) -> None:
        healthy_raw = _context_dict()
        healthy_raw["footState"]["dailyFootAnalysis"]["leftPressurePercent"] = 50.0
        healthy_raw["footState"]["dailyFootAnalysis"]["rightPressurePercent"] = 50.0
        healthy_raw["footState"]["dailyFootAnalysis"]["balanceScore"] = 90.0
        healthy_raw["footState"]["dailyFootAnalysis"]["avgHumidityPercent"] = 40.0
        hallux_raw = copy.deepcopy(healthy_raw)
        hallux_raw["footState"]["halluxValgusAnalysis"] = {
            "leftToeAngleDegree": 32.0,
            "rightToeAngleDegree": 30.0,
            "riskScore": 90.0,
        }
        healthy = score_shoe_fit(
            ServerRecommendationContext.model_validate(healthy_raw).shoes[0],
            ServerRecommendationContext.model_validate(healthy_raw).foot_state,
        )
        hallux_context = ServerRecommendationContext.model_validate(hallux_raw)
        hallux = score_shoe_fit(hallux_context.shoes[0], hallux_context.foot_state)
        healthy_scores = {area.reason_type: area.score for area in healthy}
        hallux_scores = {area.reason_type: area.score for area in hallux}

        self.assertLess(hallux_scores["FOREFOOT"], healthy_scores["FOREFOOT"])
        self.assertEqual(hallux_scores["HEEL"], healthy_scores["HEEL"])
        self.assertEqual(hallux_scores["INSOLE"], healthy_scores["INSOLE"])

    def test_pressure_and_balance_signals_change_heel_only(self) -> None:
        healthy_raw = _context_dict()
        healthy_daily = healthy_raw["footState"]["dailyFootAnalysis"]
        healthy_daily["leftPressurePercent"] = 50.0
        healthy_daily["rightPressurePercent"] = 50.0
        healthy_daily["balanceScore"] = 90.0
        healthy_daily["avgHumidityPercent"] = 40.0
        stressed_raw = copy.deepcopy(healthy_raw)
        stressed_daily = stressed_raw["footState"]["dailyFootAnalysis"]
        stressed_daily["leftPressurePercent"] = 25.0
        stressed_daily["rightPressurePercent"] = 75.0
        stressed_daily["balanceScore"] = 35.0
        healthy_context = ServerRecommendationContext.model_validate(healthy_raw)
        stressed_context = ServerRecommendationContext.model_validate(stressed_raw)
        healthy_scores = {
            area.reason_type: area.score
            for area in score_shoe_fit(healthy_context.shoes[0], healthy_context.foot_state)
        }
        stressed_scores = {
            area.reason_type: area.score
            for area in score_shoe_fit(stressed_context.shoes[0], stressed_context.foot_state)
        }

        self.assertNotEqual(stressed_scores["HEEL"], healthy_scores["HEEL"])
        self.assertEqual(stressed_scores["FOREFOOT"], healthy_scores["FOREFOOT"])
        self.assertEqual(stressed_scores["INSOLE"], healthy_scores["INSOLE"])

    def test_humidity_and_fungal_signals_change_insole_only(self) -> None:
        healthy_raw = _context_dict()
        healthy_daily = healthy_raw["footState"]["dailyFootAnalysis"]
        healthy_daily["leftPressurePercent"] = 50.0
        healthy_daily["rightPressurePercent"] = 50.0
        healthy_daily["balanceScore"] = 90.0
        healthy_daily["avgHumidityPercent"] = 40.0
        healthy_raw["footState"]["tinaPedisAnalysis"] = {
            "fungalSuspicionSafetyScore": 90,
            "skinReactionSafetyScore": 90,
        }
        humid_raw = copy.deepcopy(healthy_raw)
        humid_raw["footState"]["dailyFootAnalysis"]["avgHumidityPercent"] = 80.0
        humid_raw["footState"]["tinaPedisAnalysis"]["fungalSuspicionSafetyScore"] = 20
        healthy_context = ServerRecommendationContext.model_validate(healthy_raw)
        humid_context = ServerRecommendationContext.model_validate(humid_raw)
        healthy_scores = {
            area.reason_type: area.score
            for area in score_shoe_fit(healthy_context.shoes[0], healthy_context.foot_state)
        }
        humid_scores = {
            area.reason_type: area.score
            for area in score_shoe_fit(humid_context.shoes[0], humid_context.foot_state)
        }

        self.assertGreater(humid_scores["INSOLE"], healthy_scores["INSOLE"])
        self.assertEqual(humid_scores["FOREFOOT"], healthy_scores["FOREFOOT"])
        self.assertEqual(humid_scores["HEEL"], healthy_scores["HEEL"])

    def test_high_risk_title_does_not_claim_a_specific_missing_component(self) -> None:
        context = _context()
        with (
            patch.object(settings, "shoe_risk_low_min_score", 90.0),
            patch.object(settings, "shoe_risk_medium_min_score", 80.0),
        ):
            areas = score_shoe_fit(context.shoes[0], context.foot_state)
        insole = next(area for area in areas if area.reason_type == "INSOLE")
        self.assertEqual(insole.risk_level, "HIGH")
        self.assertEqual(insole.title, "깔창 적합도 주의")
        self.assertNotIn("통기성 부족", insole.title)

    def test_missing_characteristic_is_not_synthesized_and_area_is_reweighted(self) -> None:
        raw = _context_dict()
        metrics = raw["shoes"][0]["labMeasurements"][0]["rawMetrics"]
        raw["shoes"][0]["labMeasurements"][0]["rawMetrics"] = [
            metric for metric in metrics if metric["canonicalCharacteristic"] != "BREATHABILITY"
        ]
        context = ServerRecommendationContext.model_validate(raw)
        patches = _embedding_patches()
        with patches[0], patches[1], patches[2]:
            batch = compute_shoe_recommendations(context)
        self.assertEqual(len(batch.items[0].reasons), 3)
        self.assertTrue(0 <= batch.items[0].fit_score <= 100)

    def test_missing_runrepeat_measurement_fails_closed(self) -> None:
        context = _context()
        context.shoes[0].lab_measurements.clear()
        patches = _embedding_patches()
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(ShoeRecommendationError, "no RunRepeat"):
                compute_shoe_recommendations(context)

    def test_area_without_any_real_component_fails_closed(self) -> None:
        raw = _context_dict()
        measurement = raw["shoes"][0]["labMeasurements"][0]
        measurement["widthMm"] = None
        measurement["internalLengthMm"] = None
        measurement["toeboxWidthMm"] = None
        measurement["rawMetrics"] = [
            metric
            for metric in measurement["rawMetrics"]
            if metric["canonicalCharacteristic"]
            not in {"WIDTH_SPACE", "TOEBOX_SPACE"}
        ]
        context = ServerRecommendationContext.model_validate(raw)
        patches = _embedding_patches()
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(
                ShoeRecommendationError, "refusing to synthesize"
            ):
                compute_shoe_recommendations(context)

    def test_duplicate_shoe_id_is_rejected_before_embedding(self) -> None:
        raw = _context_dict()
        raw["shoes"].append(copy.deepcopy(raw["shoes"][0]))
        context = ServerRecommendationContext.model_validate(raw)
        with patch("app.services.shoe.shoe_recommendation.embed_texts") as embed_texts:
            with self.assertRaisesRegex(ShoeRecommendationError, "duplicate shoeId"):
                compute_shoe_recommendations(context)
        embed_texts.assert_not_called()

    def test_rejects_context_without_supported_measurement_analysis(self) -> None:
        with self.assertRaisesRegex(ShoeRecommendationError, "measurement session 30"):
            compute_shoe_recommendations(_context(include_daily_analysis=False))

    def test_pressure_sensor_readings_alone_fail_closed_until_supported(self) -> None:
        raw = _context_dict(include_daily_analysis=False)
        raw["footState"]["pressureSensorReadings"] = [
            {
                "readingId": 1,
                "footSide": "LEFT",
                "footRegion": "PRESSURE_0",
                "sensorIndex": 0,
                "pressureValue": 12.3,
                "pressureUnit": 1.0,
                "recordedAt": None,
            }
        ]
        context = ServerRecommendationContext.model_validate(raw)
        with self.assertRaisesRegex(ShoeRecommendationError, "measurement session 30"):
            compute_shoe_recommendations(context)

    def test_policy_is_explicitly_non_clinical_temporary_heuristic(self) -> None:
        self.assertEqual(POLICY_CLASSIFICATION, "TEMPORARY_HEURISTIC")
        self.assertEqual(CLINICAL_VALIDATION_STATUS, "NOT_CLINICALLY_VALIDATED")

    def test_second_batch_fails_fast_while_shared_embedding_runtime_is_busy(self) -> None:
        self.assertTrue(shoe_recommendation._batch_lock.acquire(blocking=False))
        try:
            with patch.object(
                shoe_recommendation.settings,
                "shoe_recommendation_batch_lock_timeout_seconds",
                0,
            ):
                with self.assertRaises(ShoeRecommendationBusyError):
                    compute_shoe_recommendations(_context())
        finally:
            shoe_recommendation._batch_lock.release()


if __name__ == "__main__":
    unittest.main()
