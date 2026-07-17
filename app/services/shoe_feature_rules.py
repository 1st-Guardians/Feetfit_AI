from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.shoe_db import UserFootState

REASON_TYPES = ("FOREFOOT", "HEEL", "INSOLE")

# 발볼폭 대비 발길이 비율 기준
FOREFOOT_WIDTH_LENGTH_RATIO_THRESHOLD = 0.4
# 무지외반 각도(도) 기준
FOREFOOT_HALLUX_ANGLE_THRESHOLD_DEGREE = 15.0
# 좌우 압력 차이(%p) 기준
HEEL_PRESSURE_IMBALANCE_THRESHOLD_PERCENT = 10.0
# 자세 균형 점수 기준
HEEL_BALANCE_SCORE_THRESHOLD = 70.0
# 피부 자극 안전 점수 기준
HEEL_SKIN_REACTION_SAFETY_SCORE_THRESHOLD = 70.0
# 평균 습도(%) 기준
INSOLE_HUMIDITY_THRESHOLD_PERCENT = 60.0
# 무좀 의심 안전 점수 기준
INSOLE_FUNGAL_SAFETY_SCORE_THRESHOLD = 70.0

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=[다요])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    return [part.strip() for part in parts if part.strip()]


_KEYWORDS_BY_REASON: dict[str, list[str]] = {
    "FOREFOOT": ["발볼", "앞볼", "앞꿈치", "앞코", "엄지", "발가락", "폭이", "볼이"],
    "HEEL": ["뒤꿈치", "뒷굽", "힐", "까짐", "까졌", "미끄러", "헐렁", "뒤축"],
    "INSOLE": ["깔창", "쿠션", "푹신", "딱딱", "통풍", "통기성", "바닥", "발바닥"],
}

_POSITIVE_KEYWORDS = ["편하", "여유", "좋았", "좋아요", "좋습니다", "시원하", "부드럽", "안정적", "쾌적", "조아", "조하", "굳", "굿"]
_NEGATIVE_KEYWORDS = ["좁아", "좁다", "아팠", "아파요", "아파", "까졌", "까짐", "딱딱하", "답답하", "불편", "미끄러"]


def bucket_review_by_reason(review_text: str) -> list[str]:
    matched = []
    for reason_type, keywords in _KEYWORDS_BY_REASON.items():
        if any(keyword in review_text for keyword in keywords):
            matched.append(reason_type)
    return matched


def polarity_score(review_text: str) -> int:
    score = 0
    score += sum(1 for keyword in _POSITIVE_KEYWORDS if keyword in review_text)
    score -= sum(1 for keyword in _NEGATIVE_KEYWORDS if keyword in review_text)
    return score


@dataclass(frozen=True)
class FootNeed:
    wide_forefoot_needed: bool
    forefoot_note: str
    heel_instability_risk: bool
    heel_note: str
    breathability_needed: bool
    insole_note: str


def _safe(value: float | None, default: float = 0.0) -> float:
    return default if value is None else value


def derive_foot_need(state: UserFootState) -> FootNeed:
    daily = state.daily_foot_analysis
    tinea = state.tina_pedis_analysis
    hallux = state.hallux_valgus_analysis

    left_width = _safe(daily.left_foot_width_mm) if daily else 0.0
    right_width = _safe(daily.right_foot_width_mm) if daily else 0.0
    left_length = _safe(daily.measured_left_foot_size_mm) if daily else 0.0
    right_length = _safe(daily.measured_right_foot_size_mm) if daily else 0.0
    width_length_ratio = 0.0
    if left_length and right_length:
        width_length_ratio = ((left_width / left_length) + (right_width / right_length)) / 2

    left_angle = _safe(hallux.left_toe_angle_degree) if hallux else 0.0
    right_angle = _safe(hallux.right_toe_angle_degree) if hallux else 0.0
    max_hallux_angle = max(left_angle, right_angle)

    wide_forefoot_needed = (
        width_length_ratio >= FOREFOOT_WIDTH_LENGTH_RATIO_THRESHOLD
        or max_hallux_angle >= FOREFOOT_HALLUX_ANGLE_THRESHOLD_DEGREE
    )
    forefoot_note_parts = []
    if width_length_ratio >= FOREFOOT_WIDTH_LENGTH_RATIO_THRESHOLD:
        forefoot_note_parts.append("발볼이 넓은 편")
    if max_hallux_angle >= FOREFOOT_HALLUX_ANGLE_THRESHOLD_DEGREE:
        forefoot_note_parts.append(f"무지외반 각도가 {max_hallux_angle:.1f}도로 앞코 압박에 주의가 필요")
    forefoot_note = ", ".join(forefoot_note_parts) or "발볼과 앞코 압박이 크지 않은 평이한 발 상태"

    left_pressure = _safe(daily.left_pressure_percent, 50.0) if daily else 50.0
    right_pressure = _safe(daily.right_pressure_percent, 50.0) if daily else 50.0
    pressure_imbalance = abs(left_pressure - right_pressure)
    balance_score = _safe(daily.balance_score, 100.0) if daily else 100.0
    skin_reaction_score = _safe(tinea.skin_reaction_safety_score, 100.0) if tinea else 100.0

    heel_instability_risk = (
        pressure_imbalance >= HEEL_PRESSURE_IMBALANCE_THRESHOLD_PERCENT
        or balance_score < HEEL_BALANCE_SCORE_THRESHOLD
        or skin_reaction_score < HEEL_SKIN_REACTION_SAFETY_SCORE_THRESHOLD
    )
    heel_note_parts = []
    if pressure_imbalance >= HEEL_PRESSURE_IMBALANCE_THRESHOLD_PERCENT:
        heel_note_parts.append(f"좌우 압력 차이가 {pressure_imbalance:.1f}%p")
    if balance_score < HEEL_BALANCE_SCORE_THRESHOLD:
        heel_note_parts.append(f"자세 균형 점수가 {balance_score:.0f}점으로 낮은 편")
    if skin_reaction_score < HEEL_SKIN_REACTION_SAFETY_SCORE_THRESHOLD:
        heel_note_parts.append("피부 자극/염증 반응이 있어 마찰에 민감")
    heel_note = ", ".join(heel_note_parts) or "좌우 압력과 자세 균형이 안정적인 편"

    humidity = _safe(daily.avg_humidity_percent) if daily else 0.0
    fungal_score = _safe(tinea.fungal_suspicion_safety_score, 100.0) if tinea else 100.0

    breathability_needed = (
        humidity >= INSOLE_HUMIDITY_THRESHOLD_PERCENT or fungal_score < INSOLE_FUNGAL_SAFETY_SCORE_THRESHOLD
    )
    insole_note_parts = []
    if humidity >= INSOLE_HUMIDITY_THRESHOLD_PERCENT:
        insole_note_parts.append(f"평균 습도가 {humidity:.0f}%로 높은 편")
    if fungal_score < INSOLE_FUNGAL_SAFETY_SCORE_THRESHOLD:
        insole_note_parts.append("무좀 의심 신호가 있어 통기성 관리가 필요")
    insole_note = ", ".join(insole_note_parts) or "발 환경(습도)이 양호한 편"

    return FootNeed(
        wide_forefoot_needed=wide_forefoot_needed,
        forefoot_note=forefoot_note,
        heel_instability_risk=heel_instability_risk,
        heel_note=heel_note,
        breathability_needed=breathability_needed,
        insole_note=insole_note,
    )


def build_need_query(reason_type: str, need: FootNeed) -> str:
    if reason_type == "FOREFOOT":
        if need.wide_forefoot_needed:
            return f"{need.forefoot_note}. 발볼과 앞코에 여유가 있고 압박이 적은 신발이 필요하다."
        return f"{need.forefoot_note}. 발볼이 넉넉하지 않아도 되는 발 상태다."
    if reason_type == "HEEL":
        if need.heel_instability_risk:
            return f"{need.heel_note}. 뒤꿈치를 안정적으로 잡아주고 마찰이 적은 신발이 필요하다."
        return f"{need.heel_note}. 뒤꿈치 안정성에 큰 제약은 없는 발 상태다."
    if reason_type == "INSOLE":
        if need.breathability_needed:
            return f"{need.insole_note}. 통기성이 좋고 쿠션감 있는 깔창이 필요하다."
        return f"{need.insole_note}. 깔창 통기성에 큰 제약은 없는 발 상태다."
    raise ValueError(f"Unknown reason type: {reason_type}")


def default_title(reason_type: str, risk_level: str) -> str:
    titles = {
        ("FOREFOOT", "LOW"): "발볼 여유로움",
        ("FOREFOOT", "MEDIUM"): "발볼 보통",
        ("FOREFOOT", "HIGH"): "발볼 좁음 주의",
        ("HEEL", "LOW"): "뒤꿈치 안정적",
        ("HEEL", "MEDIUM"): "뒤꿈치 보통",
        ("HEEL", "HIGH"): "뒤꿈치 까짐 주의",
        ("INSOLE", "LOW"): "쿠션/통기성 우수",
        ("INSOLE", "MEDIUM"): "쿠션/통기성 보통",
        ("INSOLE", "HIGH"): "깔창 통기성 부족",
    }
    return titles[(reason_type, risk_level)]
