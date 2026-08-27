from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Literal

from app.schemas.shoe_server import ServerShoeCharacteristics
from app.services.shoe.shoe_feature_rules import (
    PointSignal,
    classify_point_evidence,
    extract_point_summary_evidence,
    normalize_review_text,
)


EvidenceMode = Literal["SINGLE", "CONSENSUS", "MIXED", "LAB", "CONFLICT"]
EvidenceSource = Literal["MUSINSA", "RUNREPEAT"]


@dataclass(frozen=True)
class PointEvidenceFact:
    evidence_id: str
    signal: PointSignal
    source: EvidenceSource
    mode: EvidenceMode
    review_ids: tuple[int, ...]
    evidence_texts: tuple[str, ...]

    @property
    def support_count(self) -> int:
        return len(self.review_ids) if self.source == "MUSINSA" else 1


@dataclass(frozen=True)
class PointSummaryClaim:
    text: str
    evidence_ids: tuple[str, ...]


_OPPOSITE_VALUES = {
    "WIDTH_SPACE": {frozenset(("TIGHT", "ROOMY"))},
    "TOEBOX_SPACE": {frozenset(("TIGHT", "ROOMY"))},
    "HEEL_HOLD": {frozenset(("STABLE", "LOOSE"))},
    "HEEL_COMFORT": {frozenset(("COMFORTABLE", "RUBBING"))},
    "CUSHION_SOFTNESS": {frozenset(("SOFT", "FIRM"))},
    "SOLE_THICKNESS": {frozenset(("THIN", "THICK"))},
    "LONG_WEAR_COMFORT": {frozenset(("COMFORTABLE", "DISCOMFORT"))},
    "WEIGHT": {frozenset(("LIGHT", "HEAVY"))},
    "BREATHABILITY": {frozenset(("HIGH", "LOW"))},
    "SHOCK_ABSORPTION": {frozenset(("HIGH", "LOW"))},
}


def _signals_conflict(left: PointSignal, right: PointSignal) -> bool:
    if left.category != right.category or left.subject != right.subject:
        return False
    if left.value == right.value:
        return {left.stance, right.stance} == {"POSITIVE", "NEGATIVE"}
    return frozenset((left.value, right.value)) in _OPPOSITE_VALUES.get(
        left.subject, set()
    )


def _canonical_review_pairs(
    review_pairs: Iterable[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Deduplicate repeated slots and legacy duplicate bodies without fake consensus."""

    text_by_id: dict[int, str] = {}
    rank_by_body: dict[str, int] = {}
    ids_by_body: dict[str, list[int]] = defaultdict(list)
    raw_text_by_body: dict[str, str] = {}
    for rank, (review_id, review_text) in enumerate(review_pairs):
        body = normalize_review_text(review_text)
        if not body:
            continue
        previous = text_by_id.get(review_id)
        if previous is not None and previous != body:
            raise ValueError(
                f"reviewId={review_id} is associated with more than one review body"
            )
        text_by_id[review_id] = body
        if review_id not in ids_by_body[body]:
            ids_by_body[body].append(review_id)
        rank_by_body.setdefault(body, rank)
        raw_text_by_body.setdefault(body, review_text)

    return [
        (min(ids_by_body[body]), raw_text_by_body[body])
        for body in sorted(rank_by_body, key=rank_by_body.__getitem__)
    ]


def _aggregate_review_facts(
    review_pairs: Iterable[tuple[int, str]],
) -> list[PointEvidenceFact]:
    evidence_by_signal: dict[PointSignal, dict[int, str]] = defaultdict(dict)
    for review_id, review_text in _canonical_review_pairs(review_pairs):
        seen_for_review: set[PointSignal] = set()
        for clause in extract_point_summary_evidence(review_text):
            for signal in classify_point_evidence(clause):
                if signal in seen_for_review:
                    continue
                seen_for_review.add(signal)
                evidence_by_signal[signal][review_id] = clause

    facts: list[PointEvidenceFact] = []
    for signal, evidence_by_review in sorted(
        evidence_by_signal.items(), key=lambda item: item[0]
    ):
        review_ids = tuple(evidence_by_review)
        mode: EvidenceMode = "CONSENSUS" if len(review_ids) >= 2 else "SINGLE"
        facts.append(
            PointEvidenceFact(
                evidence_id=(
                    f"MUSINSA:{signal.category}:{signal.subject}:"
                    f"{signal.value}:{signal.stance}"
                ),
                signal=signal,
                source="MUSINSA",
                mode=mode,
                review_ids=review_ids,
                evidence_texts=tuple(evidence_by_review.values()),
            )
        )

    return [
        replace(
            fact,
            mode="MIXED"
            if any(
                other is not fact
                and other.source == "MUSINSA"
                and _signals_conflict(fact.signal, other.signal)
                for other in facts
            )
            else fact.mode,
        )
        for fact in facts
    ]


def _runrepeat_signal(characteristic_type: str, level: str) -> PointSignal:
    level_value = level.upper()
    mappings: dict[str, tuple[str, str, str]] = {
        "WIDTH_SPACE": (
            "WIDTH_FIT",
            "LAB_WIDTH_SPACE",
            {"LOW": "TIGHT", "MEDIUM": "BALANCED", "HIGH": "ROOMY"}[level_value],
        ),
        "TOEBOX_SPACE": (
            "WIDTH_FIT",
            "LAB_TOEBOX_SPACE",
            {"LOW": "TIGHT", "MEDIUM": "BALANCED", "HIGH": "ROOMY"}[level_value],
        ),
        "CUSHION": (
            "CUSHION_FEEL",
            "CUSHION_SOFTNESS",
            {"LOW": "FIRM", "MEDIUM": "BALANCED", "HIGH": "SOFT"}[level_value],
        ),
        "HEEL_HOLD": (
            "HEEL_FEEL",
            "HEEL_STRUCTURE",
            {"LOW": "FLEXIBLE", "MEDIUM": "BALANCED", "HIGH": "FIRM"}[level_value],
        ),
        "BREATHABILITY": ("OTHER", "BREATHABILITY", level_value),
        "SHOCK_ABSORPTION": ("CUSHION_FEEL", "SHOCK_ABSORPTION", level_value),
        "ENERGY_RETURN": ("OTHER", "ENERGY_RETURN", level_value),
    }
    category, subject, value = mappings[characteristic_type]
    return PointSignal(category, subject, value, "OBJECTIVE")


def _runrepeat_text(characteristic_type: str, level: str) -> str:
    level_text = {"LOW": "낮은 편", "MEDIUM": "보통 수준", "HIGH": "높은 편"}[level]
    names = {
        "CUSHION": "쿠션감",
        "SHOCK_ABSORPTION": "충격 완화",
        "ENERGY_RETURN": "반발력",
        "WIDTH_SPACE": "발볼 공간",
        "TOEBOX_SPACE": "앞코 공간",
        "HEEL_HOLD": "뒤꿈치 구조 강성",
        "BREATHABILITY": "통기성",
    }
    return f"RunRepeat 비교 특성에서 {names[characteristic_type]}은 {level_text}입니다."


def _runrepeat_facts(
    characteristics: ServerShoeCharacteristics | None,
) -> list[PointEvidenceFact]:
    if characteristics is None:
        return []
    facts: list[PointEvidenceFact] = []
    for item in characteristics.characteristics:
        if item.level is None:
            continue
        signal = _runrepeat_signal(item.type, item.level)
        facts.append(
            PointEvidenceFact(
                evidence_id=f"RUNREPEAT:{item.type}:{item.level}",
                signal=signal,
                source="RUNREPEAT",
                mode="LAB",
                review_ids=(),
                evidence_texts=(_runrepeat_text(item.type, item.level),),
            )
        )
    return facts


def prepare_point_evidence_facts(
    review_pairs: Iterable[tuple[int, str]],
    characteristics: ServerShoeCharacteristics | None = None,
) -> list[PointEvidenceFact]:
    review_facts = _aggregate_review_facts(review_pairs)
    lab_facts = _runrepeat_facts(characteristics)
    all_facts = [*review_facts, *lab_facts]
    return [
        replace(
            fact,
            mode="CONFLICT"
            if any(
                other.source != fact.source
                and _signals_conflict(fact.signal, other.signal)
                for other in all_facts
            )
            else fact.mode,
        )
        for fact in all_facts
    ]


def format_point_evidence_facts(facts: list[PointEvidenceFact]) -> str:
    if not facts:
        return "- 근거 없음"
    lines: list[str] = []
    for fact in facts:
        review_ids = ",".join(str(value) for value in fact.review_ids) or "-"
        expression = {
            "SINGLE": "단일 근거: 일부 착화/한 사례로만 제한",
            "CONSENSUS": "서로 다른 근거 리뷰 2개 이상: 제품 특성형 표현 가능",
            "MIXED": "상반된 리뷰 근거: 개인차로 표현하거나 생략",
            "LAB": "RunRepeat 객관 특성: 해당 특성명과 level 의미만 사용",
            "CONFLICT": "리뷰와 RunRepeat 충돌: pointSummary 주장에서는 생략",
        }[fact.mode]
        texts = " | ".join(f'"{text}"' for text in fact.evidence_texts[:3])
        lines.append(
            f"- evidenceId={fact.evidence_id}; category={fact.signal.category}; "
            f"signal={fact.signal.subject}:{fact.signal.value}:{fact.signal.stance}; "
            f"source={fact.source}; mode={fact.mode}; reviewIds={review_ids}; "
            f"rule={expression}; evidence={texts}"
        )
    return "\n".join(lines)


def _fact(
    facts: list[PointEvidenceFact],
    *,
    subject: str,
    value: str | None = None,
    source: EvidenceSource | None = None,
) -> PointEvidenceFact | None:
    candidates = [
        fact
        for fact in facts
        if fact.signal.subject == subject
        and (value is None or fact.signal.value == value)
        and (source is None or fact.source == source)
        and fact.mode != "CONFLICT"
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.mode == "CONSENSUS",
            item.mode == "LAB",
            item.support_count,
        ),
    )


def render_product_point_summary(
    shoe_name: str,
    facts: list[PointEvidenceFact],
) -> list[PointSummaryClaim]:
    """Deterministic product-copy fallback using only typed evidence facts."""

    claims: list[PointSummaryClaim] = []
    width_tight = _fact(facts, subject="WIDTH_SPACE", value="TIGHT", source="MUSINSA")
    width_roomy = _fact(facts, subject="WIDTH_SPACE", value="ROOMY", source="MUSINSA")
    full_up = _fact(facts, subject="SIZE_OPTION", value="FULL_UP", source="MUSINSA")
    half_up = _fact(facts, subject="SIZE_OPTION", value="HALF_UP", source="MUSINSA")
    generic_up = _fact(facts, subject="SIZE_OPTION", value="GENERIC_UP", source="MUSINSA")
    true_size = _fact(facts, subject="SIZE_OPTION", value="TRUE_SIZE", source="MUSINSA")
    foot_condition = _fact(
        facts,
        subject="FOOT_CONDITION",
        value="WIDE_OR_HIGH_INSTEP",
        source="MUSINSA",
    )
    heel_stable = _fact(facts, subject="HEEL_HOLD", value="STABLE", source="MUSINSA")

    fit_parts: list[str] = []
    fit_evidence: list[str] = []
    width_is_mixed = bool(
        width_tight
        and width_roomy
        and (width_tight.mode in ("MIXED", "CONFLICT") or width_roomy.mode in ("MIXED", "CONFLICT"))
    )
    size_wording = {
        "FULL_UP": "한 사이즈 업",
        "HALF_UP": "반 사이즈 업",
        "GENERIC_UP": "사이즈 업",
    }
    exact_size_fact = max(
        (
            fact
            for fact in (full_up, half_up, generic_up)
            if fact is not None and fact.signal.stance == "POSITIVE"
        ),
        key=lambda fact: (
            fact.mode == "CONSENSUS",
            fact.support_count,
            fact.signal.value != "GENERIC_UP",
        ),
        default=None,
    )
    exact_size_wording = (
        size_wording[exact_size_fact.signal.value] if exact_size_fact else None
    )
    condition_ids = set(foot_condition.review_ids) if foot_condition else set()
    exact_condition_ids = (
        set(exact_size_fact.review_ids) & condition_ids if exact_size_fact else set()
    )
    tight_condition_ids = (
        set(width_tight.review_ids) & condition_ids if width_tight else set()
    )

    if exact_size_fact and exact_size_fact.mode == "CONSENSUS":
        condition_prefix = (
            "발볼이나 발등이 있는 편이라면 "
            if len(exact_condition_ids) >= 2
            else ""
        )
        fit_parts.append(
            f"{condition_prefix}{exact_size_wording}을 고려할 만합니다"
        )
        fit_evidence.append(exact_size_fact.evidence_id)
        if condition_prefix and foot_condition:
            fit_evidence.append(foot_condition.evidence_id)
    elif (
        width_tight is not None
        and width_tight.support_count >= 2
    ):
        prefix = "발볼 착화감에는 개인차가 있지만 " if width_is_mixed else ""
        if exact_size_fact and exact_size_wording:
            exact_prefix = (
                "발볼이나 발등이 있는 편이라면 "
                if exact_condition_ids
                else "일부 착화에서는 "
            )
            fit_parts.append(
                f"{exact_prefix}{exact_size_wording}이 더 맞을 수 있습니다"
            )
            fit_evidence.append(exact_size_fact.evidence_id)
            if exact_condition_ids and foot_condition:
                fit_evidence.append(foot_condition.evidence_id)
        if len(tight_condition_ids) >= 2:
            if exact_size_fact:
                fit_parts.append(
                    f"{prefix}비교적 밀착되는 핏에 가깝습니다"
                )
                fit_evidence.append(width_tight.evidence_id)
            else:
                fit_parts.append(
                    f"{prefix}발볼이나 발등이 있는 편에서는 비교적 밀착될 수 있습니다"
                )
                fit_evidence.extend((width_tight.evidence_id, foot_condition.evidence_id))
        else:
            fit_parts.append(
                f"{prefix}발볼은 비교적 밀착되게 느껴질 수 있습니다"
            )
            fit_evidence.append(width_tight.evidence_id)
    elif exact_size_fact:
        condition_prefix = (
            "발볼이나 발등이 있는 편이라면 "
            if exact_condition_ids
            else "일부 착화에서는 "
        )
        fit_parts.append(
            f"{condition_prefix}{exact_size_wording}이 더 맞을 수 있습니다"
        )
        fit_evidence.append(exact_size_fact.evidence_id)
        if exact_condition_ids and foot_condition:
            fit_evidence.append(foot_condition.evidence_id)
    elif width_tight and width_tight.mode == "CONSENSUS":
        fit_parts.append("발볼은 비교적 밀착되는 핏에 가깝습니다")
        fit_evidence.append(width_tight.evidence_id)

    if (
        true_size
        and true_size.mode == "CONSENSUS"
        and true_size.signal.stance == "POSITIVE"
    ):
        fit_parts.append("정사이즈는 전체적으로 딱 맞는 핏입니다")
        fit_evidence.append(true_size.evidence_id)
    if heel_stable and heel_stable.mode == "CONSENSUS":
        fit_parts.append("발목은 비교적 안정적으로 잡아주는 편입니다")
        fit_evidence.append(heel_stable.evidence_id)
    if fit_parts:
        for index in range(len(fit_parts) - 1):
            if fit_parts[index].endswith("고려할 만합니다"):
                fit_parts[index] = (
                    fit_parts[index].removesuffix("고려할 만합니다")
                    + "고려할 만하고"
                )
            elif fit_parts[index].endswith("입니다"):
                fit_parts[index] = fit_parts[index].removesuffix("입니다") + "이며"
            elif fit_parts[index].endswith("습니다"):
                fit_parts[index] = fit_parts[index].removesuffix("습니다") + "고"
        claims.append(
            PointSummaryClaim(
                text=", ".join(fit_parts),
                evidence_ids=tuple(dict.fromkeys(fit_evidence)),
            )
        )

    lab_cushion = _fact(facts, subject="CUSHION_SOFTNESS", source="RUNREPEAT")
    review_soft = _fact(facts, subject="CUSHION_SOFTNESS", value="SOFT", source="MUSINSA")
    review_firm = _fact(facts, subject="CUSHION_SOFTNESS", value="FIRM", source="MUSINSA")
    sole_thin = _fact(facts, subject="SOLE_THICKNESS", value="THIN", source="MUSINSA")
    long_wear = _fact(facts, subject="LONG_WEAR_COMFORT", value="DISCOMFORT", source="MUSINSA")
    comfort_parts: list[str] = []
    comfort_evidence: list[str] = []
    if lab_cushion:
        cushion_phrase = {
            "SOFT": "실측상 쿠션은 부드러운 편",
            "BALANCED": "실측상 쿠션 부드러움은 보통 수준",
            "FIRM": "실측상 쿠션은 단단한 편",
        }[lab_cushion.signal.value]
        comfort_parts.append(cushion_phrase)
        comfort_evidence.append(lab_cushion.evidence_id)
    elif review_soft and review_soft.mode == "CONSENSUS":
        comfort_parts.append("쿠션은 비교적 부드럽게 느껴지는 편")
        comfort_evidence.append(review_soft.evidence_id)
    elif review_firm and review_firm.mode == "CONSENSUS":
        comfort_parts.append("쿠션은 푹신한 타입보다 바닥감이 느껴지는 편")
        comfort_evidence.append(review_firm.evidence_id)

    if sole_thin and long_wear and set(sole_thin.review_ids) & set(long_wear.review_ids):
        thin_text = (
            "일부 착화에서는 바닥이 얇게 느껴져 오래 서 있을 때 "
            "발바닥 부담이 생길 수 있습니다"
        )
        if comfort_parts:
            comfort_parts[-1] = f"{comfort_parts[-1]}이지만, {thin_text}"
        else:
            comfort_parts.append(thin_text)
        comfort_evidence.extend((sole_thin.evidence_id, long_wear.evidence_id))
    elif sole_thin:
        thin_text = "일부 착화에서는 바닥이 얇게 느껴질 수 있습니다"
        if comfort_parts:
            comfort_parts[-1] = f"{comfort_parts[-1]}이지만, {thin_text}"
        else:
            comfort_parts.append(thin_text)
        comfort_evidence.append(sole_thin.evidence_id)
    if comfort_parts:
        if not comfort_parts[-1].endswith(("습니다", "입니다")):
            comfort_parts[-1] += "입니다"
        claims.append(
            PointSummaryClaim(
                text=", ".join(comfort_parts),
                evidence_ids=tuple(dict.fromkeys(comfort_evidence)),
            )
        )

    heavy = _fact(facts, subject="WEIGHT", value="HEAVY", source="MUSINSA")
    light = _fact(facts, subject="WEIGHT", value="LIGHT", source="MUSINSA")
    design = _fact(facts, subject="APPEARANCE", value="POSITIVE", source="MUSINSA")
    black_color_design = next(
        (
            fact
            for fact in facts
            if fact.source == "MUSINSA"
            and fact.signal.subject == "APPEARANCE"
            and fact.signal.value == "POSITIVE"
            and any(
                marker in normalize_review_text(" ".join(fact.evidence_texts))
                for marker in ("검정", "검은색", "블랙")
            )
        ),
        None,
    )
    weight_text: str | None = None
    weight_evidence: list[str] = []
    weight_is_mixed = bool(
        heavy
        and light
        and (heavy.mode in ("MIXED", "CONFLICT") or light.mode in ("MIXED", "CONFLICT"))
    )
    if weight_is_mixed:
        weight_text = "착화에 따라 무겁거나 가볍게 느껴지는 정도가 다를 수 있습니다"
        weight_evidence.extend((heavy.evidence_id, light.evidence_id))
    elif heavy:
        if heavy.mode == "CONSENSUS":
            weight_text = "착화 시 다소 무게감이 느껴지는 편입니다"
        else:
            weight_text = "일부 착화에서는 다소 무게감이 느껴질 수 있습니다"
        weight_evidence.append(heavy.evidence_id)
    elif light:
        if light.mode == "CONSENSUS":
            weight_text = "가볍게 느껴지는 편입니다"
        else:
            weight_text = "일부 착화에서는 가볍게 느껴질 수 있습니다"
        weight_evidence.append(light.evidence_id)

    design_text: str | None = None
    design_evidence: list[str] = []
    if design and design.mode == "CONSENSUS":
        design_evidence.append(design.evidence_id)
        if black_color_design and black_color_design.evidence_id != design.evidence_id:
            if black_color_design.mode == "CONSENSUS":
                design_text = "검정 색감과 디자인에 대한 만족도는 높은 편입니다"
            else:
                design_text = (
                    "검정 색감은 만족스럽게 느껴질 수 있고, "
                    "디자인에 대한 만족도는 높은 편입니다"
                )
            design_evidence.append(black_color_design.evidence_id)
        elif black_color_design:
            design_text = "검정 색감과 디자인에 대한 만족도는 높은 편입니다"
        else:
            design_text = "디자인에 대한 만족도는 높은 편입니다"

    if weight_text and len(claims) < 4:
        claims.append(
            PointSummaryClaim(
                text=weight_text,
                evidence_ids=tuple(dict.fromkeys(weight_evidence)),
            )
        )
    if design_text and len(claims) < 4:
        claims.append(
            PointSummaryClaim(
                text=design_text,
                evidence_ids=tuple(dict.fromkeys(design_evidence)),
            )
        )

    outfit = next(
        (
            fact
            for fact in facts
            if fact.source == "MUSINSA"
            and fact.signal.subject == "OUTFIT"
            and fact.signal.value in {"JEANS", "SLACKS", "SKIRT", "WIDE_PANTS"}
            and fact.signal.stance == "POSITIVE"
            and fact.mode != "CONFLICT"
        ),
        None,
    )
    if outfit and len(claims) < 4:
        outfit_name = {
            "JEANS": "청바지",
            "SLACKS": "슬랙스",
            "SKIRT": "스커트",
            "WIDE_PANTS": "통큰 바지",
        }[outfit.signal.value]
        outfit_text = (
            f"{outfit_name} 코디에 자연스럽게 어울리는 편입니다"
            if outfit.mode == "CONSENSUS"
            else f"일부 착화에서는 {outfit_name} 코디와 자연스럽게 어울렸습니다"
        )
        claims.append(
            PointSummaryClaim(
                text=outfit_text,
                evidence_ids=(outfit.evidence_id,),
            )
        )

    if not claims:
        claims.append(
            PointSummaryClaim(
                text="착화감에 관한 일관된 근거가 아직 부족합니다",
                evidence_ids=(),
            )
        )
    return claims[:4]


def join_point_summary_claims(claims: list[PointSummaryClaim]) -> str:
    return " ".join(f"{claim.text.rstrip('.')}." for claim in claims)
