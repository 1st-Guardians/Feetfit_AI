from __future__ import annotations

import unittest

from app.services.shoe.shoe_feature_rules import classify_point_evidence


def _signals(text: str) -> set[tuple[str, str, str, str]]:
    return {
        (signal.category, signal.subject, signal.value, signal.stance)
        for signal in classify_point_evidence(text)
    }


class PointEvidenceAdversarialClassificationTests(unittest.TestCase):
    def test_full_up_with_explicit_discomfort_is_not_positive(self) -> None:
        signals = _signals("1업했지만 작고 불편해요.")

        self.assertNotIn(("SIZE_FIT", "SIZE_OPTION", "FULL_UP", "POSITIVE"), signals)
        self.assertIn(("SIZE_FIT", "SIZE_OPTION", "FULL_UP", "NEGATIVE"), signals)

    def test_negated_heaviness_is_not_heavy(self) -> None:
        signals = _signals("생각보다 무겁지 않아요.")

        self.assertFalse(
            any(subject == "WEIGHT" and value == "HEAVY" for _, subject, value, _ in signals)
        )

    def test_negated_long_wear_pain_is_comfortable_not_discomfort(self) -> None:
        signals = _signals("오래 신어도 안 아프고 편해요.")

        self.assertNotIn(
            ("LONG_WEAR", "LONG_WEAR_COMFORT", "DISCOMFORT", "NEGATIVE"),
            signals,
        )
        self.assertIn(
            ("LONG_WEAR", "LONG_WEAR_COMFORT", "COMFORTABLE", "POSITIVE"),
            signals,
        )

    def test_negated_outfit_pairing_is_not_positive(self) -> None:
        signals = _signals("통큰 바지에 안 어울려요.")

        self.assertNotIn(("STYLING", "OUTFIT", "WIDE_PANTS", "POSITIVE"), signals)
        self.assertIn(("STYLING", "OUTFIT", "WIDE_PANTS", "NEGATIVE"), signals)

    def test_wide_wearer_is_not_roomy_shoe(self) -> None:
        signals = _signals("발볼이 넓은 사람이라 한 사이즈 업했어요.")

        self.assertNotIn(("WIDTH_FIT", "WIDTH_SPACE", "ROOMY", "POSITIVE"), signals)
        self.assertIn(
            ("WIDTH_FIT", "FOOT_CONDITION", "WIDE_OR_HIGH_INSTEP", "NEUTRAL"),
            signals,
        )

    def test_size_options_keep_their_own_polarity_in_one_sentence(self) -> None:
        signals = _signals("반업은 편하고 정사이즈는 좁아요.")

        self.assertIn(("SIZE_FIT", "SIZE_OPTION", "HALF_UP", "POSITIVE"), signals)
        self.assertIn(("SIZE_FIT", "SIZE_OPTION", "TRUE_SIZE", "NEGATIVE"), signals)
        self.assertNotIn(("SIZE_FIT", "SIZE_OPTION", "HALF_UP", "NEGATIVE"), signals)
        self.assertNotIn(("SIZE_FIT", "SIZE_OPTION", "TRUE_SIZE", "POSITIVE"), signals)


if __name__ == "__main__":
    unittest.main()
