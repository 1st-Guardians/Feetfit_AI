from __future__ import annotations

import unittest

from app.services.shoe.shoe_recommendation import (
    ReasonFacts,
    ShoeFacts,
    ShoeRecommendationBatch,
)
from app.services.shoe.shoe_recommendation_dry_run import build_dry_run_report
from tests.test_shoe_recommendation import _context


class ShoeRecommendationDryRunTests(unittest.TestCase):
    def test_pass_report_proves_full_coverage_and_review_ownership(self) -> None:
        context = _context()
        batch = ShoeRecommendationBatch(
            user_id=7,
            measurement_session_id=30,
            items=[
                ShoeFacts(
                    shoe_id=101,
                    fit_score=82.5,
                    reasons=[
                        ReasonFacts(
                            reason_type=reason_type,
                            score=82.5,
                            title="title",
                            risk_level="LOW",
                            review_ids=[1010],
                        )
                        for reason_type in ("FOREFOOT", "HEEL", "INSOLE")
                    ],
                )
            ],
        )
        report = build_dry_run_report(context, batch)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["coverage"]["inputShoeCount"], 1)
        self.assertFalse(report["safety"]["serverMutationRequested"])
        self.assertFalse(report["safety"]["ollamaCalled"])

    def test_foreign_review_and_duplicate_output_fail(self) -> None:
        context = _context()
        invalid = ShoeFacts(
            shoe_id=101,
            fit_score=50.0,
            reasons=[
                ReasonFacts(
                    reason_type=reason_type,
                    score=50.0,
                    title="title",
                    risk_level="MEDIUM",
                    review_ids=[9999],
                )
                for reason_type in ("FOREFOOT", "HEEL", "INSOLE")
            ],
        )
        report = build_dry_run_report(
            context,
            ShoeRecommendationBatch(
                user_id=7,
                measurement_session_id=30,
                items=[invalid, invalid],
            ),
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("DUPLICATE_OUTPUT_SHOE_ID", report["issues"])
        self.assertTrue(
            any(issue.startswith("REVIEW_OWNERSHIP_CONFLICT") for issue in report["issues"])
        )


if __name__ == "__main__":
    unittest.main()
