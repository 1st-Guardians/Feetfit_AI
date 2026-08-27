from __future__ import annotations

import unittest

from app.schemas.shoe_fit_comment import ShoeFitSummary, ShoePointSummaryClaim
from app.schemas.shoe_server import ServerShoeCharacteristics
from app.services.shoe.shoe_fit_comment_service import (
    ReasonFactsForPrompt,
    ShoeFitCommentError,
    _point_claims_are_traceable,
    prepare_point_summary_evidence,
)
from app.services.shoe.shoe_point_summary import (
    join_point_summary_claims,
    render_product_point_summary,
)


def _empty_reason(reason_type: str) -> ReasonFactsForPrompt:
    return ReasonFactsForPrompt(
        reason_type=reason_type,
        title=f"{reason_type} title",
        risk_level="LOW",
        review_ids=[],
        review_texts=[],
    )


def _reasons(
    *pairs: tuple[str, list[int], list[str]],
) -> list[ReasonFactsForPrompt]:
    supplied = {reason_type: (ids, texts) for reason_type, ids, texts in pairs}
    return [
        ReasonFactsForPrompt(
            reason_type=reason_type,
            title=f"{reason_type} title",
            risk_level="LOW",
            review_ids=supplied.get(reason_type, ([], []))[0],
            review_texts=supplied.get(reason_type, ([], []))[1],
        )
        for reason_type in ("FOREFOOT", "HEEL", "INSOLE")
    ]


def _characteristics(*rows: tuple[str, str]) -> ServerShoeCharacteristics:
    return ServerShoeCharacteristics.model_validate(
        {
            "shoeId": 1,
            "summary": "RunRepeat 특성",
            "characteristics": [
                {
                    "type": characteristic,
                    "level": level,
                    "value": 1,
                    "averageValue": 1,
                    "minValue": 0,
                    "maxValue": 2,
                    "unit": "score",
                    "testedSize": None,
                }
                for characteristic, level in rows
            ],
        }
    )


class PointEvidenceAggregationTests(unittest.TestCase):
    def test_categories_and_review_ids_remain_traceable(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101],
                    ["발볼과 발등이 있어서 1업했고 정사이즈는 발가락이 까졌습니다."],
                ),
                ("HEEL", [201], ["발목을 안정적으로 잡아줍니다."]),
                (
                    "INSOLE",
                    [301],
                    ["바닥이 얇아서 오래 서 있으면 발바닥이 아프고 무게감도 있습니다."],
                ),
            )
        )
        by_id = {fact.evidence_id: fact for fact in facts}
        self.assertEqual(
            by_id["MUSINSA:SIZE_FIT:SIZE_OPTION:FULL_UP:POSITIVE"].review_ids,
            (101,),
        )
        self.assertIn("MUSINSA:HEEL_FEEL:HEEL_HOLD:STABLE:POSITIVE", by_id)
        self.assertIn(
            "MUSINSA:LONG_WEAR:LONG_WEAR_COMFORT:DISCOMFORT:NEGATIVE",
            by_id,
        )
        self.assertIn("MUSINSA:WEIGHT_FEEL:WEIGHT:HEAVY:NEGATIVE", by_id)

    def test_same_review_across_reason_slots_counts_once(self) -> None:
        body = "발목을 안정적으로 잡아줍니다."
        facts = prepare_point_summary_evidence(
            _reasons(
                ("FOREFOOT", [101], [body]),
                ("HEEL", [101], [body]),
                ("INSOLE", [101], [body]),
            )
        )
        stable = next(fact for fact in facts if fact.signal.value == "STABLE")
        self.assertEqual(stable.review_ids, (101,))
        self.assertEqual(stable.mode, "SINGLE")

    def test_duplicate_bodies_do_not_create_false_consensus(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [109, 101],
                    ["발목이 안정적이에요.", "\u200b발목이  안정적이에요.\n"],
                )
            )
        )
        stable = next(fact for fact in facts if fact.signal.value == "STABLE")
        self.assertEqual(stable.review_ids, (101,))
        self.assertEqual(stable.mode, "SINGLE")

    def test_same_review_id_with_different_body_fails_closed(self) -> None:
        with self.assertRaisesRegex(ShoeFitCommentError, "more than one review body"):
            prepare_point_summary_evidence(
                _reasons(
                    ("FOREFOOT", [101], ["발볼이 편해요."]),
                    ("HEEL", [101], ["발목이 안정적이에요."]),
                )
            )

    def test_distinct_reviews_same_signal_become_consensus(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    ["발목을 안정적으로 잡아줘요.", "뒤꿈치를 안정적으로 받쳐줍니다."],
                )
            )
        )
        stable = next(fact for fact in facts if fact.signal.value == "STABLE")
        self.assertEqual(stable.mode, "CONSENSUS")
        self.assertEqual(stable.review_ids, (101, 102))

    def test_opposite_width_signals_are_mixed(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    ["발볼이 좁아서 불편해요.", "발볼이 여유롭고 편해요."],
                )
            )
        )
        width = [fact for fact in facts if fact.signal.subject == "WIDTH_SPACE"]
        self.assertEqual({fact.mode for fact in width}, {"MIXED"})

    def test_long_wear_is_not_inferred_without_duration_marker(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("INSOLE", [101], ["바닥이 얇고 발바닥이 아파요."]))
        )
        self.assertFalse(any(fact.signal.category == "LONG_WEAR" for fact in facts))

    def test_soft_cushion_and_thin_sole_are_separate_not_conflicting(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("INSOLE", [101], ["쿠션은 푹신하지만 바닥은 얇아요."]))
        )
        selected = {
            fact.signal.subject: fact.mode
            for fact in facts
            if fact.signal.subject in {"CUSHION_SOFTNESS", "SOLE_THICKNESS"}
        }
        self.assertEqual(selected, {"CUSHION_SOFTNESS": "SINGLE", "SOLE_THICKNESS": "SINGLE"})

    def test_runrepeat_and_review_same_concept_conflict_is_explicit(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("INSOLE", [101], ["쿠션이 딱딱하고 푹신하지 않아요."])),
            _characteristics(("CUSHION", "HIGH")),
        )
        cushion = [fact for fact in facts if fact.signal.subject == "CUSHION_SOFTNESS"]
        self.assertTrue(cushion)
        self.assertTrue(all(fact.mode == "CONFLICT" for fact in cushion))


class PointSummaryRendererTests(unittest.TestCase):
    def test_product_copy_uses_consensus_and_scopes_single_evidence(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "발볼과 발등이 있어서 1업했고 정사이즈는 발가락이 까졌어요.",
                        "발볼러는 힘들어서 사이즈를 여유 있게 골라야 합니다.",
                    ],
                ),
                (
                    "HEEL",
                    [201, 202],
                    ["발목을 안정적으로 잡아줘요.", "뒤꿈치를 안정적으로 받쳐줍니다."],
                ),
                (
                    "INSOLE",
                    [301, 302],
                    [
                        "바닥이 얇아서 오래 서 있으면 발바닥이 아파요.",
                        "쿠션 때문인지 신발에 무게감이 있습니다.",
                    ],
                ),
            ),
            _characteristics(("CUSHION", "HIGH")),
        )
        summary = join_point_summary_claims(
            render_product_point_summary("반스 하프캡 - 블랙", facts)
        )
        self.assertIn("한 사이즈 업이 더 맞을 수 있", summary)
        self.assertIn("발목은 비교적 안정적으로 잡아주는 편", summary)
        self.assertIn("실측상 쿠션은 부드러운 편", summary)
        self.assertIn("오래 서 있을 때 발바닥 부담", summary)
        self.assertIn("일부 착화에서는 다소 무게감", summary)
        for report_style in ("후기를 참고", "확인하는 것이 좋", "한편", "부위별 결과"):
            self.assertNotIn(report_style, summary)

    def test_single_styling_fact_is_not_generalized(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("HEEL", [101], ["통큰 바지에 잘 어울려요."]))
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )
        self.assertIn("일부 착화에서는 통큰 바지", summary)
        self.assertNotIn("통큰 바지 코디에 잘 어울리는 신발", summary)

    def test_negative_styling_fact_is_never_rendered_as_positive(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("HEEL", [101], ["통큰 바지에는 안 어울려요."]))
        )
        summary = join_point_summary_claims(
            render_product_point_summary("테스트 신발", facts)
        )
        self.assertNotIn("자연스럽게 어울", summary)

    def test_full_up_is_never_rendered_as_half_up(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101],
                    ["발볼과 발등이 있는 편이라 1업했고 정사이즈는 발가락이 까졌어요."],
                )
            )
        )
        summary = join_point_summary_claims(render_product_point_summary("테스트", facts))
        self.assertIn("한 사이즈 업", summary)
        self.assertNotIn("반 사이즈 업", summary)

    def test_black_color_copy_requires_review_evidence_and_cites_it(self) -> None:
        facts_without_color = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102],
                    ["디자인이 예뻐요.", "외관이 마음에 듭니다."],
                )
            )
        )
        claims_without_color = render_product_point_summary(
            "테스트 신발 - 블랙", facts_without_color
        )
        self.assertNotIn("검정 색감", join_point_summary_claims(claims_without_color))

        facts_with_color = prepare_point_summary_evidence(
            _reasons(
                (
                    "HEEL",
                    [101, 102, 103],
                    [
                        "디자인이 예뻐요.",
                        "외관이 마음에 듭니다.",
                        "검은색 디자인이 예뻐요.",
                    ],
                )
            )
        )
        color_claim = next(
            claim
            for claim in render_product_point_summary(
                "테스트 신발 - 블랙", facts_with_color
            )
            if "검정 색감" in claim.text
        )
        color_evidence_id = next(
            fact.evidence_id
            for fact in facts_with_color
            if any("검은색" in text for text in fact.evidence_texts)
        )
        self.assertIn(color_evidence_id, color_claim.evidence_ids)

    def test_single_size_fact_does_not_invent_a_foot_condition(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(("FOREFOOT", [101], ["한 사이즈 업했더니 편하게 맞아요."]))
        )
        summary = join_point_summary_claims(render_product_point_summary("테스트", facts))
        self.assertIn("일부 착화에서는 한 사이즈 업이 더 맞을 수 있", summary)
        self.assertNotIn("발볼이나 발등", summary)

    def test_consensus_half_up_outweighs_single_full_up(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102, 103],
                    [
                        "반업했더니 편하게 잘 맞아요.",
                        "반 사이즈 업하니 편안합니다.",
                        "한 사이즈 업했더니 편하게 맞아요.",
                    ],
                )
            )
        )
        summary = join_point_summary_claims(render_product_point_summary("테스트", facts))
        self.assertIn("반 사이즈 업을 고려", summary)
        self.assertNotIn("한 사이즈 업을 고려", summary)

    def test_mixed_weight_is_not_rendered_as_heavy_consensus(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "INSOLE",
                    [101, 102],
                    ["신발에 무게감이 있어요.", "신어보니 가볍고 편합니다."],
                )
            )
        )
        summary = join_point_summary_claims(render_product_point_summary("테스트", facts))
        self.assertIn("착화에 따라 무겁거나 가볍게", summary)
        self.assertNotIn("다소 무게감이 느껴지는 편", summary)

    def test_deterministic_product_claims_pass_the_same_grounding_gate(self) -> None:
        facts = prepare_point_summary_evidence(
            _reasons(
                (
                    "FOREFOOT",
                    [101, 102],
                    [
                        "발볼과 발등이 있어서 1업했고 정사이즈는 발가락이 까졌어요.",
                        "발볼러는 힘들어요.",
                    ],
                ),
                (
                    "HEEL",
                    [201, 202],
                    ["발목을 안정적으로 잡아줘요.", "뒤꿈치를 안정적으로 받쳐줍니다."],
                ),
                (
                    "INSOLE",
                    [301],
                    ["바닥이 얇아서 오래 서 있으면 발바닥이 아파요."],
                ),
            ),
            _characteristics(("CUSHION", "HIGH")),
        )
        claims = render_product_point_summary("테스트 신발", facts)
        point_summary = join_point_summary_claims(claims)
        result = ShoeFitSummary(
            point_summary=point_summary,
            point_summary_claims=[
                ShoePointSummaryClaim(
                    text=claim.text,
                    evidence_ids=list(claim.evidence_ids),
                )
                for claim in claims
            ],
            forefoot_summary="발볼 요약",
            heel_summary="뒤꿈치 요약",
            insole_summary="깔창 요약",
            forefoot_review_ids=[],
            heel_review_ids=[],
            insole_review_ids=[],
        )

        self.assertTrue(
            _point_claims_are_traceable(
                result,
                shoe_name="테스트 신발",
                facts=facts,
            )
        )


if __name__ == "__main__":
    unittest.main()
