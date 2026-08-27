from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.core.config import settings
from app.schemas.shoe_server import ServerUserFootState

REASON_TYPES = ("FOREFOOT", "HEEL", "INSOLE")

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=[다요])\s+|\n+")
_INVISIBLE_TEXT_PATTERN = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_review_text(text: str) -> str:
    """Return a conservative fingerprint for duplicate review text.

    Punctuation and words are intentionally preserved so two genuinely different
    reviews are not merged. Unicode presentation differences, invisible
    characters, and whitespace-only differences are ignored.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _INVISIBLE_TEXT_PATTERN.sub("", normalized)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip().casefold()


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    return [part.strip() for part in parts if part.strip()]


_KEYWORDS_BY_REASON: dict[str, list[str]] = {
    "FOREFOOT": ["발볼", "앞볼", "앞꿈치", "앞코", "엄지", "발가락", "폭이", "볼이"],
    "HEEL": ["뒤꿈치", "뒷굽", "힐", "까짐", "까졌", "미끄러", "헐렁", "뒤축", "발목"],
    "INSOLE": ["깔창", "쿠션", "푹신", "딱딱", "통풍", "통기성", "바닥", "발바닥"],
}

_NON_FIT_TOPIC_KEYWORDS = (
    "바지",
    "밑단",
    "코디",
    "패션",
    "예뻐",
    "예쁘",
    "이뻐",
    "디자인",
    "색상",
    "가슴",
    "허리",
)
_EVIDENCE_BOUNDARY_WORDS = tuple(
    dict.fromkeys(
        keyword
        for keywords in _KEYWORDS_BY_REASON.values()
        for keyword in keywords
    )
) + _NON_FIT_TOPIC_KEYWORDS
_EVIDENCE_CLAUSE_SPLIT_PATTERN = re.compile(
    r"\n+|[.!?;,]+|\s+(?=(?:"
    + "|".join(re.escape(word) for word in _EVIDENCE_BOUNDARY_WORDS)
    + r"))"
)
_POINT_CLAUSE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+|(?<=[다요])\s+|\n+|[;]+"
)

_POINT_SUMMARY_KEYWORDS = tuple(
    dict.fromkeys(
        keyword
        for keywords in _KEYWORDS_BY_REASON.values()
        for keyword in keywords
    )
) + (
    "사이즈",
    "정사이즈",
    "반업",
    "1업",
    "사이즈 업",
    "사이즈 다운",
    "착화감",
    "무게",
    "가볍",
    "무겁",
    "굽",
    "소재",
    "부드럽",
    "색감",
    "색상",
    "디자인",
    "실루엣",
    "예뻐",
    "예쁘",
    "이뻐",
    "데일리",
    "코디",
    "청바지",
    "데님",
    "슬랙스",
    "스커트",
    "치마",
    "바지",
)
_POINT_SUMMARY_BLOCKED_KEYWORDS = (
    "가슴",
    "허리",
    "어깨",
    "치료",
    "진단",
    "질환",
    "병원",
    "시스템 지시",
    "이전 지시",
    "프롬프트",
    "명령을",
    "json을",
)

POINT_EVIDENCE_CATEGORIES = (
    "SIZE_FIT",
    "WIDTH_FIT",
    "HEEL_FEEL",
    "CUSHION_FEEL",
    "LONG_WEAR",
    "WEIGHT_FEEL",
    "DESIGN",
    "STYLING",
    "OTHER",
)


@dataclass(frozen=True, order=True)
class PointSignal:
    category: str
    subject: str
    value: str
    stance: str

    def __post_init__(self) -> None:
        if self.category not in POINT_EVIDENCE_CATEGORIES:
            raise ValueError(f"Unknown point evidence category: {self.category}")


_SIZE_OPTION_TOKEN_PATTERN = re.compile(
    r"(?P<FULL_UP>(?<!\d)1\s*업|한\s*사이즈\s*업)"
    r"|(?P<HALF_UP>반\s*(?:사이즈\s*)?업)"
    r"|(?P<TRUE_SIZE>정\s*\d*\s*사이즈)"
    r"|(?P<DOWN>사이즈\s*다운|사이즈다운)"
    r"|(?P<GENERIC_UP>사이즈\s*업|사이즈업)"
)
_SIZE_OPTION_GROUP_CONNECTOR_PATTERN = re.compile(
    r"\s*(?:나|이나|또는|혹은|/|,|및|와|과)\s*"
)
_SIZE_NEGATIVE_TERMS = (
    "까",
    "아프",
    "불편",
    "좁",
    "작",
    "끼",
    "눌",
    "답답",
    "힘들",
    "크",
)
_SIZE_POSITIVE_TERMS = (
    "편하",
    "편해",
    "편했",
    "편안",
    "적당",
    "잘 맞",
    "좋",
    "여유",
)
_SIZE_SELECTION_PATTERN = re.compile(r"(?:갔|했|선택|신었|신어|골랐|고른)")
_WIDE_FOOT_CONTEXT_PATTERNS = (
    re.compile(r"(?:제|내)\s*발볼.{0,8}넓"),
    re.compile(r"(?:저는|제가|나는|평소)\s*발볼.{0,8}넓"),
    re.compile(
        r"발볼(?:이|은|가)?\s*(?:좀|많이|매우)?\s*넓(?:은)?\s*"
        r"(?:사람|분|편)(?:이|이라|인데|이고|입니다|이에요|이라서)?"
    ),
)
_EXPLICIT_SHOE_WIDTH_PATTERN = re.compile(
    r"(?:이\s*신발|신발|제품|모델).{0,12}발볼"
)


def _term_occurrence_is_negated(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 8) : start]
    after = text[end : end + 12]
    return bool(
        re.search(r"(?:안|못|별로|전혀)\s*$", before)
        or re.match(r"(?:하)?지(?:는|도)?\s*(?:않|못)", after)
    )


def _has_unnegated_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        for match in re.finditer(re.escape(term), text):
            if not _term_occurrence_is_negated(text, match.start(), match.end()):
                return True
    return False


def _has_negated_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        for match in re.finditer(re.escape(term), text):
            if _term_occurrence_is_negated(text, match.start(), match.end()):
                return True
    return False


def _size_option_contexts(text: str) -> list[tuple[str, str]]:
    """Keep each size option tied to its own predicate.

    Coordinated options such as ``정사이즈나 반업은 좁다`` share the predicate
    after the final option, while contrasting clauses such as
    ``반업은 편하고 정사이즈는 좁다`` remain separate.
    """

    matches = list(_SIZE_OPTION_TOKEN_PATTERN.finditer(text))
    contexts: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        last_grouped = index
        while last_grouped + 1 < len(matches):
            gap = text[
                matches[last_grouped].end() : matches[last_grouped + 1].start()
            ]
            if not _SIZE_OPTION_GROUP_CONNECTOR_PATTERN.fullmatch(gap):
                break
            last_grouped += 1
        context_end = (
            matches[last_grouped + 1].start()
            if last_grouped + 1 < len(matches)
            else len(text)
        )
        contexts.append((match.lastgroup or "", text[match.start() : context_end]))
    return contexts


def _size_option_stance(option: str, context: str) -> str:
    if _has_unnegated_term(context, _SIZE_NEGATIVE_TERMS):
        return "NEGATIVE"
    if _has_unnegated_term(context, _SIZE_POSITIVE_TERMS):
        return "POSITIVE"
    if option != "TRUE_SIZE" and _SIZE_SELECTION_PATTERN.search(context):
        return "POSITIVE"
    return "NEUTRAL"


def _describes_wearer_wide_foot(text: str) -> bool:
    if _EXPLICIT_SHOE_WIDTH_PATTERN.search(text):
        return False
    return any(pattern.search(text) for pattern in _WIDE_FOOT_CONTEXT_PATTERNS)

_POSITIVE_KEYWORDS = ["편하", "여유", "좋았", "좋아요", "좋습니다", "시원하", "부드럽", "안정적", "쾌적", "조아", "조하", "굳", "굿"]
_NEGATIVE_KEYWORDS = ["좁아", "좁다", "아팠", "아파요", "아파", "까졌", "까짐", "딱딱하", "답답하", "불편", "미끄러"]


def bucket_review_by_reason(review_text: str) -> list[str]:
    matched = []
    for reason_type, keywords in _KEYWORDS_BY_REASON.items():
        if any(keyword in review_text for keyword in keywords):
            matched.append(reason_type)
    return matched


def extract_reason_evidence(reason_type: str, review_text: str) -> str | None:
    """Keep only clauses that explicitly support one fit reason.

    The semantic shortlist operates on individual sentences, but only review IDs
    are persisted by Feetfit_Server. When a later summary request returns the full
    review body, this function prevents unrelated fashion, appearance, or body
    anecdotes from being exposed to Ollama as fit evidence.
    """

    if reason_type not in _KEYWORDS_BY_REASON:
        raise ValueError(f"Unknown reason type: {reason_type}")

    normalized_source = unicodedata.normalize("NFKC", review_text)
    normalized_source = _INVISIBLE_TEXT_PATTERN.sub("", normalized_source)
    clauses = _EVIDENCE_CLAUSE_SPLIT_PATTERN.split(normalized_source)
    selected: list[str] = []
    seen: set[str] = set()
    for raw_clause in clauses:
        clause = _WHITESPACE_PATTERN.sub(" ", raw_clause).strip()
        if not clause or reason_type not in bucket_review_by_reason(clause):
            continue
        if any(keyword in clause for keyword in _NON_FIT_TOPIC_KEYWORDS):
            continue
        fingerprint = normalize_review_text(clause)
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(clause)
        if len(selected) >= 3:
            break
    return " ".join(selected) or None


def extract_point_summary_evidence(review_text: str) -> list[str]:
    """Extract safe fit/product/style clauses for the overall shoe description.

    Unlike area summaries, pointSummary may describe sizing, cushioning, weight,
    appearance, and outfit compatibility. The returned clauses are still
    verbatim evidence: emotional body metaphors, medical language, and prompt
    injection-like instructions are excluded before Ollama sees them.
    """

    normalized_source = unicodedata.normalize("NFKC", review_text)
    normalized_source = _INVISIBLE_TEXT_PATTERN.sub("", normalized_source)
    clauses = _POINT_CLAUSE_SPLIT_PATTERN.split(normalized_source)
    selected: list[str] = []
    seen: set[str] = set()
    for raw_clause in clauses:
        clause = _WHITESPACE_PATTERN.sub(" ", raw_clause).strip()
        normalized_clause = normalize_review_text(clause)
        if not normalized_clause:
            continue
        if any(
            keyword in normalized_clause
            for keyword in _POINT_SUMMARY_BLOCKED_KEYWORDS
        ):
            continue
        if not any(keyword in normalized_clause for keyword in _POINT_SUMMARY_KEYWORDS):
            continue
        if normalized_clause in seen:
            continue
        seen.add(normalized_clause)
        selected.append(clause)
    return selected


def classify_point_evidence(clause: str) -> tuple[PointSignal, ...]:
    """Map one safe review clause to traceable, direction-aware product facts.

    Classification is deliberately multi-label.  A sentence can support size,
    width, heel and design facts, but later aggregation still keeps every fact
    tied to this exact review instead of combining a global bag of words.
    """

    text = normalize_review_text(clause)
    signals: set[PointSignal] = set()

    def add(category: str, subject: str, value: str, stance: str) -> None:
        signals.add(PointSignal(category, subject, value, stance))

    negative = _has_unnegated_term(
        text,
        ("까", "아프", "아파", "불편", "좁", "작", "끼", "눌", "답답", "힘들"),
    )
    positive = _has_unnegated_term(
        text,
        ("편하", "편해", "편했", "편안", "적당", "잘 맞", "좋", "여유", "안정", "만족"),
    )

    size_value_by_option = {
        "FULL_UP": "FULL_UP",
        "HALF_UP": "HALF_UP",
        "TRUE_SIZE": "TRUE_SIZE",
        "GENERIC_UP": "GENERIC_UP",
        "DOWN": "DOWN",
    }
    for option, context in _size_option_contexts(text):
        add(
            "SIZE_FIT",
            "SIZE_OPTION",
            size_value_by_option[option],
            _size_option_stance(option, context),
        )

    width_subject = any(
        marker in text for marker in ("발볼", "앞볼", "앞코", "발등", "발볼러", "발가락")
    )
    if width_subject:
        wide_foot_context = _describes_wearer_wide_foot(text)
        if any(marker in text for marker in ("좁", "끼", "눌", "까", "힘들", "작")):
            add("WIDTH_FIT", "WIDTH_SPACE", "TIGHT", "NEGATIVE")
        if (
            any(marker in text for marker in ("여유", "넉넉", "발볼도 편", "발볼이 편"))
            or ("넓" in text and not wide_foot_context)
        ):
            add("WIDTH_FIT", "WIDTH_SPACE", "ROOMY", "POSITIVE")
        if wide_foot_context or any(
            marker in text
            for marker in (
                "발볼러",
                "발볼 발등",
                "발볼과 발등",
                "발볼이나 발등",
                "발등있는",
            )
        ):
            add("WIDTH_FIT", "FOOT_CONDITION", "WIDE_OR_HIGH_INSTEP", "NEUTRAL")

    heel_subject = any(marker in text for marker in ("뒤꿈치", "뒷꿈치", "뒤축", "발목"))
    if heel_subject:
        if any(marker in text for marker in ("안정", "잡아", "받쳐", "지지")):
            add("HEEL_FEEL", "HEEL_HOLD", "STABLE", "POSITIVE")
        if any(marker in text for marker in ("헐렁", "들뜨", "벗겨", "미끄러")):
            add("HEEL_FEEL", "HEEL_HOLD", "LOOSE", "NEGATIVE")
        if any(marker in text for marker in ("까", "쓸", "마찰", "불편")):
            add("HEEL_FEEL", "HEEL_COMFORT", "RUBBING", "NEGATIVE")
        if any(marker in text for marker in ("편하", "폭신", "부드럽")):
            add("HEEL_FEEL", "HEEL_COMFORT", "COMFORTABLE", "POSITIVE")

    cushion_subject = any(marker in text for marker in ("쿠션", "깔창", "바닥", "발바닥"))
    if cushion_subject:
        soft_negated = bool(
            re.search(r"푹신하?지\s*(?:않|못)", text)
            or "안 푹신" in text
        )
        if not soft_negated and any(marker in text for marker in ("푹신", "폭신", "부드럽")):
            add("CUSHION_FEEL", "CUSHION_SOFTNESS", "SOFT", "POSITIVE")
        if soft_negated or any(marker in text for marker in ("딱딱", "쿠션감 없", "쿠션이 없")):
            add("CUSHION_FEEL", "CUSHION_SOFTNESS", "FIRM", "NEGATIVE")
        if "바닥" in text and "얇" in text:
            add("CUSHION_FEEL", "SOLE_THICKNESS", "THIN", "NEGATIVE")
        if "바닥" in text and "두껍" in text:
            add("CUSHION_FEEL", "SOLE_THICKNESS", "THICK", "POSITIVE")
        if "충격" in text:
            add("CUSHION_FEEL", "SHOCK_FEEL", "HIGH" if negative else "LOW", "NEGATIVE" if negative else "POSITIVE")

    duration = any(marker in text for marker in ("오래", "장시간", "종일", "하루종일", "오래 서"))
    if duration and _has_unnegated_term(
        text, ("아프", "아파", "피로", "불편", "부담", "힘들")
    ):
        add("LONG_WEAR", "LONG_WEAR_COMFORT", "DISCOMFORT", "NEGATIVE")
    elif duration and _has_unnegated_term(
        text, ("편하", "편해", "편했", "괜찮", "좋")
    ):
        add("LONG_WEAR", "LONG_WEAR_COMFORT", "COMFORTABLE", "POSITIVE")

    heavy_is_negated = _has_negated_term(text, ("무게감", "무겁")) or bool(
        re.search(r"무게감(?:이|은|도)?\s*(?:없|적)", text)
    )
    if not heavy_is_negated and _has_unnegated_term(text, ("무게감", "무겁")):
        add("WEIGHT_FEEL", "WEIGHT", "HEAVY", "NEGATIVE")
    if _has_unnegated_term(text, ("가볍",)):
        add("WEIGHT_FEEL", "WEIGHT", "LIGHT", "POSITIVE")

    appearance = any(marker in text for marker in ("디자인", "색감", "색상", "예쁘", "예뻐", "이뻐"))
    if appearance:
        add("DESIGN", "APPEARANCE", "POSITIVE" if positive or any(marker in text for marker in ("예쁘", "예뻐", "이뻐", "마음에")) else "MENTIONED", "POSITIVE" if positive else "NEUTRAL")
    if "둥글" in text:
        add("DESIGN", "SILHOUETTE", "ROUNDED", "NEUTRAL")
    if "날렵" in text:
        add("DESIGN", "SILHOUETTE", "SLIM", "NEUTRAL")
    if any(marker in text for marker in ("로우", "낮은 굽", "굽은 낮")):
        add("DESIGN", "PROFILE", "LOW", "NEUTRAL")

    styling_values = {
        "JEANS": ("청바지", "데님"),
        "SLACKS": ("슬랙스",),
        "SKIRT": ("스커트", "치마"),
        "WIDE_PANTS": ("통큰 바지", "와이드 팬츠"),
        "DAILY": ("데일리", "일상용"),
    }
    negative_pairing = _has_negated_term(text, ("어울",))
    positive_pairing = _has_unnegated_term(text, ("잘 어울", "잘어울", "어울"))
    for value, markers in styling_values.items():
        if any(marker in text for marker in markers):
            outfit_stance = (
                "NEGATIVE"
                if negative_pairing
                else ("POSITIVE" if positive_pairing or positive else "NEUTRAL")
            )
            add("STYLING", "OUTFIT", value, outfit_stance)
    if (positive_pairing or "스타일내기" in text) and not negative_pairing:
        add("STYLING", "OUTFIT", "GENERAL", "POSITIVE")

    if not signals and any(marker in text for marker in ("소재", "굽", "착화감")):
        add("OTHER", "PRODUCT_DETAIL", "MENTIONED", "NEUTRAL")
    return tuple(sorted(signals))


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


def derive_foot_need(state: ServerUserFootState) -> FootNeed:
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
        width_length_ratio >= settings.shoe_forefoot_width_ratio_neutral
        or max_hallux_angle >= settings.shoe_hallux_angle_neutral_degree
    )
    forefoot_note_parts = []
    if width_length_ratio >= settings.shoe_forefoot_width_ratio_neutral:
        forefoot_note_parts.append("발볼이 넓은 편")
    if max_hallux_angle >= settings.shoe_hallux_angle_neutral_degree:
        forefoot_note_parts.append(f"무지외반 각도가 {max_hallux_angle:.1f}도로 앞코 압박에 주의가 필요")
    forefoot_note = ", ".join(forefoot_note_parts) or "발볼과 앞코 압박이 크지 않은 평이한 발 상태"

    left_pressure = _safe(daily.left_pressure_percent, 50.0) if daily else 50.0
    right_pressure = _safe(daily.right_pressure_percent, 50.0) if daily else 50.0
    pressure_imbalance = abs(left_pressure - right_pressure)
    balance_score = _safe(daily.balance_score, 100.0) if daily else 100.0
    skin_reaction_score = _safe(tinea.skin_reaction_safety_score, 100.0) if tinea else 100.0

    heel_instability_risk = (
        pressure_imbalance >= settings.shoe_pressure_imbalance_neutral_percent
        or balance_score < settings.shoe_balance_neutral_score
        or skin_reaction_score < settings.shoe_skin_safety_neutral_score
    )
    heel_note_parts = []
    if pressure_imbalance >= settings.shoe_pressure_imbalance_neutral_percent:
        heel_note_parts.append(f"좌우 압력 차이가 {pressure_imbalance:.1f}%p")
    if balance_score < settings.shoe_balance_neutral_score:
        heel_note_parts.append(f"자세 균형 점수가 {balance_score:.0f}점으로 낮은 편")
    if skin_reaction_score < settings.shoe_skin_safety_neutral_score:
        heel_note_parts.append("피부 자극/염증 반응이 있어 마찰에 민감")
    heel_note = ", ".join(heel_note_parts) or "좌우 압력과 자세 균형이 안정적인 편"

    humidity = _safe(daily.avg_humidity_percent) if daily else 0.0
    fungal_score = _safe(tinea.fungal_suspicion_safety_score, 100.0) if tinea else 100.0

    breathability_needed = (
        humidity >= settings.shoe_humidity_neutral_percent
        or fungal_score < settings.shoe_fungal_safety_neutral_score
    )
    insole_note_parts = []
    if humidity >= settings.shoe_humidity_neutral_percent:
        insole_note_parts.append(f"평균 습도가 {humidity:.0f}%로 높은 편")
    if fungal_score < settings.shoe_fungal_safety_neutral_score:
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
        requirement = (
            "발볼과 앞코에 여유가 있고 압박이 적은 신발이 필요하다"
            if need.wide_forefoot_needed
            else "발볼이 넉넉하지 않아도 되는 발 상태다"
        )
        return (
            f"{need.forefoot_note}. {requirement}. 무신사 실제 착화 리뷰에서 발볼/앞코가 "
            "넓고 편안하거나, 반대로 좁아 아프고 발가락이 눌리는 경험을 모두 찾는다."
        )
    if reason_type == "HEEL":
        requirement = (
            "뒤꿈치를 안정적으로 잡아주고 마찰이 적은 신발이 필요하다"
            if need.heel_instability_risk
            else "뒤꿈치 안정성에 큰 제약은 없는 발 상태다"
        )
        return (
            f"{need.heel_note}. {requirement}. 무신사 실제 착화 리뷰에서 뒤꿈치가 안정적으로 "
            "잡히거나, 반대로 들뜨고 벗겨지고 쓸려 까지는 경험을 모두 찾는다."
        )
    if reason_type == "INSOLE":
        requirement = (
            "통기성이 좋고 쿠션감 있는 깔창이 필요하다"
            if need.breathability_needed
            else "깔창 통기성에 큰 제약은 없는 발 상태다"
        )
        return (
            f"{need.insole_note}. {requirement}. 무신사 실제 착화 리뷰에서 깔창이 푹신하고 "
            "통풍이 좋거나, 반대로 딱딱하고 충격이 크고 답답하며 열감이나 땀이 나는 경험을 모두 찾는다."
        )
    raise ValueError(f"Unknown reason type: {reason_type}")


def default_title(reason_type: str, risk_level: str) -> str:
    # Keep titles deterministic and area-level. A low aggregate score can be
    # caused by any of the available RunRepeat components (for example shock
    # absorption rather than breathability), so a component-specific title
    # would be capable of contradicting the quantitative evidence.
    titles = {
        ("FOREFOOT", "LOW"): "발볼 적합도 좋음",
        ("FOREFOOT", "MEDIUM"): "발볼 적합도 보통",
        ("FOREFOOT", "HIGH"): "발볼 적합도 주의",
        ("HEEL", "LOW"): "뒤꿈치 적합도 좋음",
        ("HEEL", "MEDIUM"): "뒤꿈치 적합도 보통",
        ("HEEL", "HIGH"): "뒤꿈치 적합도 주의",
        ("INSOLE", "LOW"): "깔창 적합도 좋음",
        ("INSOLE", "MEDIUM"): "깔창 적합도 보통",
        ("INSOLE", "HIGH"): "깔창 적합도 주의",
    }
    return titles[(reason_type, risk_level)]
