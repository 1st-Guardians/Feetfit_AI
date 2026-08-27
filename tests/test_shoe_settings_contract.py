from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.core.config import Settings


class ShoeSettingsContractTests(unittest.TestCase):
    def test_accepts_server_and_payload_contract_boundaries(self) -> None:
        minimums = Settings(
            _env_file=None,
            shoe_recommendation_context_page_size=1,
            shoe_reviews_per_reason=1,
        )
        maximums = Settings(
            _env_file=None,
            shoe_recommendation_context_page_size=200,
            shoe_reviews_per_reason=3,
        )

        self.assertEqual(minimums.shoe_recommendation_context_page_size, 1)
        self.assertEqual(minimums.shoe_reviews_per_reason, 1)
        self.assertEqual(maximums.shoe_recommendation_context_page_size, 200)
        self.assertEqual(maximums.shoe_reviews_per_reason, 3)

    def test_rejects_context_page_size_outside_server_contract(self) -> None:
        for invalid_size in (0, 201):
            with self.subTest(invalid_size=invalid_size), self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    shoe_recommendation_context_page_size=invalid_size,
                )

    def test_rejects_review_count_outside_payload_contract(self) -> None:
        for invalid_count in (0, 4):
            with self.subTest(invalid_count=invalid_count), self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    shoe_reviews_per_reason=invalid_count,
                )

    def test_rejects_non_positive_component_weight_groups(self) -> None:
        invalid_groups = (
            {
                "shoe_forefoot_width_component_weight": 0,
                "shoe_forefoot_toebox_component_weight": 0,
            },
            {
                "shoe_heel_hold_component_weight": 0,
                "shoe_heel_shock_component_weight": 0,
                "shoe_heel_energy_component_weight": 0,
                "shoe_heel_cushion_component_weight": 0,
            },
            {
                "shoe_insole_breathability_component_weight": 0,
                "shoe_insole_cushion_component_weight": 0,
                "shoe_insole_shock_component_weight": 0,
            },
        )
        for overrides in invalid_groups:
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                Settings(_env_file=None, **overrides)

    def test_rejects_inverted_policy_thresholds_and_invalid_tolerance(self) -> None:
        invalid_overrides = (
            {"shoe_risk_low_min_score": 40, "shoe_risk_medium_min_score": 50},
            {
                "shoe_forefoot_width_ratio_neutral": 0.43,
                "shoe_forefoot_width_ratio_high": 0.40,
            },
            {
                "shoe_balance_neutral_score": 50,
                "shoe_balance_high_risk_score": 70,
            },
            {"shoe_metric_target_tolerance": 0},
            {"shoe_metric_target_tolerance": 1.1},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                Settings(_env_file=None, **overrides)

    def test_server_endpoint_accepts_field_override_and_preferred_env_alias(self) -> None:
        direct = Settings(
            _env_file=None,
            shoe_recommendation_endpoint="https://server.example/api/shoes/recommendations",
        )
        self.assertEqual(
            direct.shoe_recommendation_endpoint,
            "https://server.example/api/shoes/recommendations",
        )
        with patch.dict(
            "os.environ",
            {
                "FEETFIT_SERVER_RECOMMENDATION_ENDPOINT": (
                    "https://env.example/api/shoes/recommendations"
                )
            },
            clear=False,
        ):
            from_env = Settings(_env_file=None)
        self.assertEqual(
            from_env.shoe_recommendation_endpoint,
            "https://env.example/api/shoes/recommendations",
        )

    def test_internal_key_accepts_shared_alias_with_feetfit_name_precedence(self) -> None:
        with patch.dict(
            "os.environ",
            {"INTERNAL_API_KEY": "shared-key"},
            clear=True,
        ):
            shared = Settings(_env_file=None)
        self.assertEqual(shared.feetfit_server_internal_api_key, "shared-key")

        with patch.dict(
            "os.environ",
            {
                "INTERNAL_API_KEY": "shared-key",
                "FEETFIT_SERVER_INTERNAL_API_KEY": "ai-specific-key",
            },
            clear=True,
        ):
            preferred = Settings(_env_file=None)
        self.assertEqual(
            preferred.feetfit_server_internal_api_key,
            "ai-specific-key",
        )

    def test_default_shoe_server_endpoints_are_loopback_only(self) -> None:
        values = Settings(_env_file=None)
        for endpoint in (
            values.shoe_recommendation_endpoint,
            values.shoe_summary_save_endpoint_template,
            values.shoe_recommendation_context_endpoint,
            values.shoe_summary_context_endpoint_template,
        ):
            self.assertIn("127.0.0.1:8080", endpoint)
            self.assertNotIn("54.184.58.176", endpoint)

    def test_callback_timeout_cannot_drop_below_nine_hundred_seconds(self) -> None:
        self.assertEqual(
            Settings(_env_file=None).feetfit_server_callback_timeout_seconds,
            900.0,
        )
        with self.assertRaises(ValidationError):
            Settings(
                _env_file=None,
                feetfit_server_callback_timeout_seconds=899.9,
            )

    def test_semantic_threshold_is_bounded_as_temporary_policy(self) -> None:
        self.assertEqual(Settings(_env_file=None).shoe_review_semantic_min_score, 0.42)
        for invalid in (-1.01, 1.01):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                Settings(_env_file=None, shoe_review_semantic_min_score=invalid)

    def test_candidate_limit_is_positive_and_bounded(self) -> None:
        self.assertEqual(
            Settings(_env_file=None).shoe_max_candidate_reviews_per_reason,
            40,
        )
        for invalid in (0, -1, 201):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    shoe_max_candidate_reviews_per_reason=invalid,
                )

    def test_runtime_timeouts_must_be_positive_and_bounded(self) -> None:
        for field_name in (
            "report_proxy_timeout_seconds",
            "ollama_request_timeout_seconds",
        ):
            for invalid in (0, -0.1, 3600.1):
                with (
                    self.subTest(field_name=field_name, invalid=invalid),
                    self.assertRaises(ValidationError),
                ):
                    Settings(_env_file=None, **{field_name: invalid})


if __name__ == "__main__":
    unittest.main()
