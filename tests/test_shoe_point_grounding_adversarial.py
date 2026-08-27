from __future__ import annotations

import unittest

from app.schemas.shoe_fit_comment import ShoeFitSummary, ShoePointSummaryClaim
from app.services.shoe.shoe_feature_rules import PointSignal
from app.services.shoe.shoe_fit_comment_service import (
    _contains_mechanical_area_result_summary,
    _point_claims_are_traceable,
)
from app.services.shoe.shoe_point_summary import PointEvidenceFact


def _fact(
    evidence_id: str,
    *,
    category: str,
    subject: str,
    value: str,
    stance: str,
    source: str = "MUSINSA",
    mode: str = "SINGLE",
    review_ids: tuple[int, ...] = (101,),
    evidence_text: str,
) -> PointEvidenceFact:
    return PointEvidenceFact(
        evidence_id=evidence_id,
        signal=PointSignal(category, subject, value, stance),
        source=source,
        mode=mode,
        review_ids=review_ids if source == "MUSINSA" else (),
        evidence_texts=(evidence_text,),
    )


def _summary(
    *claims: tuple[str, tuple[str, ...]],
    point_summary: str | None = None,
) -> ShoeFitSummary:
    claim_models = [
        ShoePointSummaryClaim(text=text, evidence_ids=list(evidence_ids))
        for text, evidence_ids in claims
    ]
    if point_summary is None:
        point_summary = " ".join(
            f"{claim.text.rstrip('.')}." for claim in claim_models
        )
    return ShoeFitSummary(
        point_summary=point_summary,
        point_summary_claims=claim_models,
        forefoot_summary="발볼",
        heel_summary="뒤꿈치",
        insole_summary="깔창",
        forefoot_review_ids=[],
        heel_review_ids=[],
        insole_review_ids=[],
    )


class PointClaimGroundingAdversarialTests(unittest.TestCase):
    def test_assertion_without_any_evidence_is_rejected(self) -> None:
        summary = _summary(point_summary="이 신발은 내구성이 뛰어납니다.")

        self.assertFalse(
            _point_claims_are_traceable(summary, shoe_name="테스트", facts=[])
        )

    def test_single_fact_scope_is_required_for_each_claim(self) -> None:
        full_up = _fact(
            "MUSINSA:SIZE:FULL_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="FULL_UP",
            stance="POSITIVE",
            evidence_text="발볼이 있어 1업했습니다.",
        )
        unsafe = _summary(
            ("한 착화 사례에서는 한 사이즈 업을 선택했습니다", (full_up.evidence_id,)),
            ("한 사이즈 업이 잘 맞습니다", (full_up.evidence_id,)),
        )
        safe = _summary(
            ("한 착화 사례에서는 한 사이즈 업을 선택했습니다", (full_up.evidence_id,)),
        )

        self.assertFalse(
            _point_claims_are_traceable(unsafe, shoe_name="테스트", facts=[full_up])
        )
        self.assertTrue(
            _point_claims_are_traceable(safe, shoe_name="테스트", facts=[full_up])
        )

    def test_single_fact_cannot_use_universal_quantifiers(self) -> None:
        full_up = _fact(
            "MUSINSA:SIZE:FULL_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="FULL_UP",
            stance="POSITIVE",
            evidence_text="1업했습니다.",
        )
        summary = _summary(
            ("누구나 한 사이즈 업을 선택했습니다", (full_up.evidence_id,)),
        )

        self.assertFalse(
            _point_claims_are_traceable(summary, shoe_name="테스트", facts=[full_up])
        )

    def test_mixed_fact_requires_explicit_variation_scope(self) -> None:
        tight = _fact(
            "MUSINSA:WIDTH:TIGHT",
            category="WIDTH_FIT",
            subject="WIDTH_SPACE",
            value="TIGHT",
            stance="NEGATIVE",
            mode="MIXED",
            evidence_text="발볼이 좁게 느껴집니다.",
        )
        unsafe = _summary(("발볼은 좁은 편입니다", (tight.evidence_id,)))
        safe = _summary(
            ("발볼 착화감에는 개인차가 있어 좁게 느껴질 수 있습니다", (tight.evidence_id,))
        )

        self.assertFalse(
            _point_claims_are_traceable(unsafe, shoe_name="테스트", facts=[tight])
        )
        self.assertTrue(
            _point_claims_are_traceable(safe, shoe_name="테스트", facts=[tight])
        )

    def test_lab_fact_rejects_reversal_exaggeration_and_unrelated_property(self) -> None:
        cushion = _fact(
            "RUNREPEAT:CUSHION:HIGH",
            category="CUSHION_FEEL",
            subject="CUSHION_SOFTNESS",
            value="SOFT",
            stance="OBJECTIVE",
            source="RUNREPEAT",
            mode="LAB",
            review_ids=(),
            evidence_text="RunRepeat 비교 특성에서 쿠션감은 높은 편입니다.",
        )
        accurate = _summary(
            ("RunRepeat 실측에서 쿠션감은 높은 편입니다", (cushion.evidence_id,))
        )
        reversed_claim = _summary(
            ("RunRepeat 실측에서 쿠션감은 낮은 편입니다", (cushion.evidence_id,))
        )
        exaggerated = _summary(
            ("쿠션은 구름 위를 걷듯 최고 수준입니다", (cushion.evidence_id,))
        )
        unrelated = _summary(
            ("이 신발은 내구성이 뛰어납니다", (cushion.evidence_id,))
        )

        self.assertTrue(
            _point_claims_are_traceable(accurate, shoe_name="테스트", facts=[cushion])
        )
        for unsafe in (reversed_claim, exaggerated, unrelated):
            self.assertFalse(
                _point_claims_are_traceable(
                    unsafe, shoe_name="테스트", facts=[cushion]
                )
            )

    def test_full_up_may_not_be_generalized_to_unspecified_size_up(self) -> None:
        full_up = _fact(
            "MUSINSA:SIZE:FULL_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="FULL_UP",
            stance="POSITIVE",
            evidence_text="1업했더니 편했습니다.",
        )
        unsafe = _summary(
            ("일부 착화에서는 사이즈 업을 고려할 만합니다", (full_up.evidence_id,))
        )
        safe = _summary(
            ("일부 착화에서는 한 사이즈 업을 고려할 만합니다", (full_up.evidence_id,))
        )

        self.assertFalse(
            _point_claims_are_traceable(unsafe, shoe_name="테스트", facts=[full_up])
        )
        self.assertTrue(
            _point_claims_are_traceable(safe, shoe_name="테스트", facts=[full_up])
        )

    def test_numeric_size_relation_must_preserve_source_order_and_direction(self) -> None:
        generic_up = _fact(
            "MUSINSA:SIZE:GENERIC_UP",
            category="SIZE_FIT",
            subject="SIZE_OPTION",
            value="GENERIC_UP",
            stance="POSITIVE",
            evidence_text="평소 250을 신다가 260으로 사이즈 업했습니다.",
        )
        correct = _summary(
            ("일부 착화에서는 250에서 260으로 사이즈 업했습니다", (generic_up.evidence_id,))
        )
        reversed_claim = _summary(
            ("일부 착화에서는 260에서 250으로 사이즈 다운했습니다", (generic_up.evidence_id,))
        )

        self.assertTrue(
            _point_claims_are_traceable(correct, shoe_name="테스트", facts=[generic_up])
        )
        self.assertFalse(
            _point_claims_are_traceable(
                reversed_claim, shoe_name="테스트", facts=[generic_up]
            )
        )

    def test_distributed_area_list_and_report_style_variants_are_rejected(self) -> None:
        distributed = (
            "발볼 적합도는 주의입니다. "
            "뒤꿈치 적합도는 좋습니다. "
            "깔창 적합도는 보통입니다."
        )
        parallel = "발볼은 좁습니다. 뒤꿈치는 안정적입니다. 깔창은 단단합니다."

        self.assertTrue(_contains_mechanical_area_result_summary(distributed))
        self.assertTrue(_contains_mechanical_area_result_summary(parallel))
        self.assertTrue(
            _contains_mechanical_area_result_summary(
                "구매평에 따르면 쿠션이 좋고 착화평도 우수합니다."
            )
        )


if __name__ == "__main__":
    unittest.main()
