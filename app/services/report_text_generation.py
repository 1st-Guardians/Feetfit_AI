from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.schemas.reports import FootTypeAnalysisContext


logger = logging.getLogger(__name__)
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass(frozen=True)
class TineaReportText:
    fungal_suspicion_safety_description: str
    skin_reaction_safety_description: str
    total_score_description: str


@dataclass(frozen=True)
class FootTypeEvidence:
    evidence_id: str
    category: str
    canonical_fact: str
    type_text: str


@dataclass(frozen=True)
class FootTypeReportText:
    type_text: str
    evidence_id: str
    source: str


def _compact_text(value: Any, fallback: str, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        text = fallback
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _openai_enabled() -> bool:
    return bool(settings.openai_report_text_enabled and (settings.openai_api_key or "").strip())


def _png_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_response_text(data: dict) -> str:
    direct_text = data.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                raise RuntimeError(f"OpenAI refused report text generation: {content.get('refusal')}")
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
    raise RuntimeError("OpenAI response did not contain output text.")


async def _create_structured_response(
    *,
    system_prompt: str,
    user_payload: dict,
    schema_name: str,
    json_schema: dict,
    images: list[tuple[str, bytes]] | None = None,
    max_output_tokens: int = 600,
) -> dict:
    content: list[dict] = [
        {
            "type": "input_text",
            "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
        }
    ]
    if settings.openai_report_include_images:
        for label, image_bytes in images or []:
            if not image_bytes:
                continue
            content.append({"type": "input_text", "text": f"분석 이미지: {label}"})
            content.append({"type": "input_image", "image_url": _png_data_url(image_bytes), "detail": "low"})

    request_body = {
        "model": settings.openai_report_model,
        # Foot-analysis inputs may be sensitive.  Responses used for report
        # copy are not retained by OpenAI when this request-level flag is set.
        "store": False,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            }
        },
        "max_output_tokens": max_output_tokens,
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=settings.openai_report_timeout_seconds) as client:
        response = await client.post(OPENAI_RESPONSES_URL, headers=headers, json=request_body)

    if response.status_code >= 400:
        body = response.text.replace("\n", "\\n")
        raise RuntimeError(f"OpenAI Responses API returned {response.status_code}: {body[:600]}")

    data = response.json()
    if data.get("status") == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
        raise RuntimeError(f"OpenAI response was incomplete: {reason}")

    return json.loads(_extract_response_text(data))


_ARCH_EVIDENCE = {
    "LOW": FootTypeEvidence(
        evidence_id="ARCH_LOW",
        category="ARCH",
        canonical_fact="분석 결과에서 발 아치가 낮은 유형으로 분류됨",
        type_text=(
            "발의 아치가 낮아 발바닥이 넓게 닿는 편이에요. "
            "오래 걷거나 서 있으면 피로가 커질 수 있어 아치를 잘 받쳐주는 신발이 더 편안할 수 있어요."
        ),
    ),
    "NORMAL": FootTypeEvidence(
        evidence_id="ARCH_NORMAL",
        category="ARCH",
        canonical_fact="분석 결과에서 발 아치가 보통 유형으로 분류됨",
        type_text=(
            "발의 아치와 발바닥 접촉이 비교적 균형적인 편이에요. "
            "발을 고르게 받쳐주고 전체적으로 편안하게 맞는 신발이 잘 맞을 수 있어요."
        ),
    ),
    "HIGH": FootTypeEvidence(
        evidence_id="ARCH_HIGH",
        category="ARCH",
        canonical_fact="분석 결과에서 발 아치가 높은 유형으로 분류됨",
        type_text=(
            "발의 아치가 높은 편이라 발바닥의 일부 부위에 압력이 집중될 수 있어요. "
            "충격을 부드럽게 분산하고 쿠션이 안정적인 신발이 더 편안할 수 있어요."
        ),
    ),
}

_WIDTH_EVIDENCE = {
    "NARROW": FootTypeEvidence(
        evidence_id="WIDTH_NARROW",
        category="WIDTH",
        canonical_fact="분석 결과에서 발볼이 좁은 유형으로 분류됨",
        type_text=(
            "발볼이 슬림한 편이라 너비가 넉넉한 신발에서는 발이 안에서 움직일 수 있어요. "
            "발볼을 안정적으로 잡아주는 신발이 더 잘 맞을 수 있어요."
        ),
    ),
    "NORMAL": FootTypeEvidence(
        evidence_id="WIDTH_NORMAL",
        category="WIDTH",
        canonical_fact="분석 결과에서 발볼이 보통 유형으로 분류됨",
        type_text=(
            "발볼 너비가 비교적 균형적인 편이에요. "
            "발길이와 발볼이 자연스럽게 맞고 발을 안정적으로 받쳐주는 신발이 편안할 수 있어요."
        ),
    ),
    "WIDE": FootTypeEvidence(
        evidence_id="WIDTH_WIDE",
        category="WIDTH",
        canonical_fact="분석 결과에서 발볼이 넓은 유형으로 분류됨",
        type_text=(
            "발볼이 넓은 편이라 앞쪽이 타이트한 신발에서는 압박을 느낄 수 있어요. "
            "발볼 공간이 여유로운 신발을 고르면 더 편안할 수 있어요."
        ),
    ),
}

_PRESSURE_EVIDENCE = {
    "LEFT_DOMINANT": FootTypeEvidence(
        evidence_id="PRESSURE_LEFT_DOMINANT",
        category="PRESSURE_BALANCE",
        canonical_fact="분석 결과에서 왼발 압력 비중이 더 높은 유형으로 분류됨",
        type_text=(
            "왼발에 압력이 조금 더 실리는 편이에요. "
            "양쪽 발을 고르게 받쳐주고 뒤꿈치가 안정적인 신발이 편안할 수 있어요."
        ),
    ),
    "BALANCED": FootTypeEvidence(
        evidence_id="PRESSURE_BALANCED",
        category="PRESSURE_BALANCE",
        canonical_fact="분석 결과에서 양발 압력 분포가 균형 유형으로 분류됨",
        type_text=(
            "양쪽 발의 압력 분포가 비교적 균형적인 편이에요. "
            "발바닥을 고르게 받쳐주는 신발이 편안함을 유지하는 데 도움이 될 수 있어요."
        ),
    ),
    "RIGHT_DOMINANT": FootTypeEvidence(
        evidence_id="PRESSURE_RIGHT_DOMINANT",
        category="PRESSURE_BALANCE",
        canonical_fact="분석 결과에서 오른발 압력 비중이 더 높은 유형으로 분류됨",
        type_text=(
            "오른발에 압력이 조금 더 실리는 편이에요. "
            "양쪽 발을 고르게 받쳐주고 뒤꿈치가 안정적인 신발이 편안할 수 있어요."
        ),
    ),
}

_PLANTAR_PRESSURE_EVIDENCE = FootTypeEvidence(
    evidence_id="PLANTAR_PRESSURE_PATTERN",
    category="PLANTAR_PRESSURE",
    canonical_fact="발바닥 분석 결과에서 부위별 압력 분포 차이가 확인됨",
    type_text=(
        "발바닥의 압력이 부위별로 다르게 분포하는 편이에요. "
        "발바닥을 고르게 받쳐주고 압력을 분산해 주는 신발이 더 편안할 수 있어요."
    ),
)


def _pressure_balance_type(context: FootTypeAnalysisContext) -> str:
    if context.pressure_balance_type != "UNKNOWN":
        return context.pressure_balance_type
    if (
        context.left_pressure_percent is None
        or context.right_pressure_percent is None
    ):
        return "UNKNOWN"

    difference = context.left_pressure_percent - context.right_pressure_percent
    tolerance = settings.foot_type_pressure_balance_tolerance_percent
    if abs(difference) <= tolerance:
        return "BALANCED"
    return "LEFT_DOMINANT" if difference > 0 else "RIGHT_DOMINANT"


def foot_type_evidence(context: FootTypeAnalysisContext) -> list[FootTypeEvidence]:
    """Return only evidence explicitly classified by the upstream analysis.

    Raw foot dimensions are intentionally not interpreted here.  In
    particular, width/length measurements must never be used to infer a low
    arch, flat foot, or a categorical width without a validated classifier.
    """

    evidence: list[FootTypeEvidence] = []
    if context.arch_type != "UNKNOWN":
        evidence.append(_ARCH_EVIDENCE[context.arch_type])
    if context.foot_width_type != "UNKNOWN":
        evidence.append(_WIDTH_EVIDENCE[context.foot_width_type])
    pressure_balance_type = _pressure_balance_type(context)
    if pressure_balance_type != "UNKNOWN":
        evidence.append(_PRESSURE_EVIDENCE[pressure_balance_type])
    if context.plantar_footprint_analysis_text is not None:
        evidence.append(_PLANTAR_PRESSURE_EVIDENCE)
    return evidence


def build_fallback_foot_type_text(
    context: FootTypeAnalysisContext,
) -> FootTypeReportText:
    candidates = foot_type_evidence(context)
    if not candidates:  # The request schema normally rejects this first.
        raise ValueError("Foot type text requires at least one classified analysis result.")
    selected = candidates[0]
    return FootTypeReportText(
        type_text=selected.type_text,
        evidence_id=selected.evidence_id,
        source="FALLBACK",
    )


async def generate_foot_type_text(
    context: FootTypeAnalysisContext,
) -> FootTypeReportText:
    """Select one grounded shoe-list message and render audited Korean copy.

    GPT only chooses among evidence IDs supplied by the completed analysis.
    The final sentence is rendered from a vetted mapping so an API response
    cannot introduce a diagnosis or an unsupported foot characteristic.
    """

    fallback = build_fallback_foot_type_text(context)
    candidates = foot_type_evidence(context)
    if not (
        _openai_enabled()
        and settings.openai_foot_type_text_enabled
    ):
        return fallback

    candidate_ids = [candidate.evidence_id for candidate in candidates]
    schema = {
        "type": "object",
        "properties": {
            "selectedEvidenceId": {
                "type": "string",
                "enum": candidate_ids,
            }
        },
        "required": ["selectedEvidenceId"],
        "additionalProperties": False,
    }
    payload = {
        "task": "신발 목록 상단에 표시할 가장 유용한 발 분석 근거 한 개 선택",
        "rules": [
            "후보에 없는 발 특성이나 의학적 진단을 추론하지 않는다.",
            "신발 선택에 가장 직접적으로 도움이 되는 근거를 우선한다.",
            "ARCH, WIDTH, PRESSURE_BALANCE가 모두 있다면 ARCH를 우선 검토한다.",
            "문구가 '이번 측정에서는'으로 시작되지 않도록 한다.",
            "반드시 candidateEvidence의 evidenceId 중 하나만 선택한다.",
        ],
        "candidateEvidence": [
            {
                "evidenceId": candidate.evidence_id,
                "category": candidate.category,
                "fact": candidate.canonical_fact,
            }
            for candidate in candidates
        ],
    }

    try:
        data = await _create_structured_response(
            system_prompt=(
                "너는 FeetFit의 신발 선택 보조 문구를 위한 근거 선택기다. "
                "제공된 구조화 분석 근거 밖의 사실은 만들지 말고 JSON 스키마만 반환한다."
            ),
            user_payload=payload,
            schema_name="foot_type_text_evidence_selection",
            json_schema=schema,
            images=None,
            max_output_tokens=80,
        )
        selected_id = data.get("selectedEvidenceId")
        selected = next(
            candidate
            for candidate in candidates
            if candidate.evidence_id == selected_id
        )
    except Exception as exc:
        logger.warning(
            "GPT foot type evidence selection failed; using fallback. error=%s",
            exc,
        )
        return fallback

    return FootTypeReportText(
        type_text=selected.type_text,
        evidence_id=selected.evidence_id,
        source="OPENAI",
    )


def _metrics(result: Any) -> dict:
    metrics = getattr(result, "metrics", None)
    return metrics if isinstance(metrics, dict) else {}


def _side_payload(result: Any) -> dict:
    metrics = _metrics(result)
    return {
        "fungal_safety_score": getattr(result, "fungal_safety_score", None),
        "skin_reaction_safety_score": getattr(result, "skin_reaction_safety_score", None),
        "fungal_score_label": metrics.get("fungal_score_label"),
        "skin_reaction_score_label": metrics.get("inflammation_score_label"),
        "overall_health_label": metrics.get("overall_health_label"),
        "fungal_pixel_ratio": metrics.get("fungal_pixel_ratio"),
        "inflammation_pixel_ratio": metrics.get("inflammation_pixel_ratio"),
        "max_fungal_prob": metrics.get("max_fungal_prob"),
        "max_inflammation_prob": metrics.get("max_inflammation_prob"),
        "fungal_regions": metrics.get("fungal_regions", []),
        "inflammation_regions": metrics.get("inflammation_regions", []),
        "score_reliability": metrics.get("score_reliability"),
        "foot_outline_found": metrics.get("foot_outline_found"),
    }


def _region_locations(result: Any, key: str) -> list[str]:
    locations = []
    for region in _metrics(result).get(key, [])[:3]:
        if not isinstance(region, dict):
            continue
        location = str(region.get("location") or "").strip()
        if location and location not in locations:
            locations.append(location)
    return locations


def _location_text(left_result: Any, right_result: Any, key: str) -> str:
    parts = []
    for side_label, result in (("왼발", left_result), ("오른발", right_result)):
        locations = _region_locations(result, key)
        if locations:
            parts.append(f"{side_label} {', '.join(locations)}")
    return "; ".join(parts)


def _min_score(left_result: Any, right_result: Any, attr_name: str) -> int:
    values = []
    for result in (left_result, right_result):
        value = getattr(result, attr_name, None)
        if isinstance(value, (int, float)):
            values.append(int(round(float(value))))
    return min(values) if values else 100


def _score_intensity(score: int) -> str:
    if score >= 90:
        return "낮은 수준으로"
    if score >= 70:
        return "약하게"
    if score >= 40:
        return "중간 수준으로"
    return "뚜렷하게"


def build_fallback_tinea_report_text(left_result: Any, right_result: Any) -> TineaReportText:
    fungal_score = _min_score(left_result, right_result, "fungal_safety_score")
    skin_score = _min_score(left_result, right_result, "skin_reaction_safety_score")
    fungal_locations = _location_text(left_result, right_result, "fungal_regions")
    skin_locations = _location_text(left_result, right_result, "inflammation_regions")
    fungal_signal = bool(fungal_locations) or fungal_score < 95
    skin_signal = bool(skin_locations) or skin_score < 95

    if fungal_signal:
        location = fungal_locations or "발 일부 영역"
        fungal_text = (
            f"{location}에서 무좀 의심 신호가 {_score_intensity(fungal_score)} 관찰됩니다. "
            "각질, 변색 또는 표면 손상 변화가 이어지는지 관찰과 관리가 필요합니다."
        )
    else:
        fungal_text = "분석 이미지 기준으로 양발에서 뚜렷한 무좀 의심 영역은 크게 관찰되지 않습니다."

    if skin_signal:
        location = skin_locations or "발 일부 피부"
        skin_text = (
            f"{location}에서 피부 발적 또는 자극 반응이 {_score_intensity(skin_score)} 확인됩니다. "
            "심한 부기나 짓무름은 함께 관찰해 주세요."
        )
    else:
        skin_text = "피부 발적과 자극 반응은 뚜렷하지 않아 전반적으로 안정적인 편입니다."

    if fungal_signal and skin_signal:
        total_text = "무좀 의심 신호와 피부 자극 신호가 함께 보여 청결, 건조, 마찰 관리가 필요합니다."
    elif fungal_signal:
        total_text = "무좀 의심 신호가 관찰되어 해당 부위의 변색, 각질, 손상 변화를 꾸준히 확인해 주세요."
    elif skin_signal:
        total_text = "무좀 의심은 크지 않지만 피부 자극 가능성이 있어 건조와 마찰 관리를 권장합니다."
    else:
        total_text = "전반적으로 큰 위험 신호는 두드러지지 않지만 발을 건조하고 청결하게 유지해 주세요."

    return TineaReportText(
        fungal_suspicion_safety_description=_compact_text(fungal_text, fungal_text),
        skin_reaction_safety_description=_compact_text(skin_text, skin_text),
        total_score_description=_compact_text(total_text, total_text, max_chars=160),
    )


async def generate_tinea_report_text(
    left_result: Any,
    right_result: Any,
    *,
    suspicion_map_png: bytes | None = None,
    photo_overlay_png: bytes | None = None,
) -> TineaReportText:
    fallback = build_fallback_tinea_report_text(left_result, right_result)
    if not _openai_enabled():
        return fallback

    schema = {
        "type": "object",
        "properties": {
            "fungal_suspicion_safety_description": {"type": "string"},
            "skin_reaction_safety_description": {"type": "string"},
            "total_score_description": {"type": "string"},
        },
        "required": [
            "fungal_suspicion_safety_description",
            "skin_reaction_safety_description",
            "total_score_description",
        ],
        "additionalProperties": False,
    }
    payload = {
        "task": "무좀/피부반응 리포트 문구 생성",
        "guidelines": [
            "한국어로 작성한다.",
            "진단처럼 단정하지 말고 '의심', '가능성', '관찰됩니다' 표현을 사용한다.",
            "무좀 의심과 피부 발적/자극/염증 반응을 구분한다.",
            "위치가 불명확하면 특정 발가락을 억지로 쓰지 말고 '발가락 주변', '발 중앙부'처럼 완화한다.",
            "각 필드는 1~2문장으로 짧게 작성한다.",
        ],
        "color_legend": {
            "photo_overlay": "파란색 계열은 무좀 의심, 빨간색 계열은 피부 자극/염증 의심",
            "suspicion_map": "하늘색 계열은 무좀 의심, 분홍색 계열은 피부 자극/염증 의심",
        },
        "left_foot": _side_payload(left_result),
        "right_foot": _side_payload(right_result),
        "fallback_text": {
            "fungal_suspicion_safety_description": fallback.fungal_suspicion_safety_description,
            "skin_reaction_safety_description": fallback.skin_reaction_safety_description,
            "total_score_description": fallback.total_score_description,
        },
    }
    images = [
        ("왼쪽이 왼발, 오른쪽이 오른발인 의심 영역 지도", suspicion_map_png or b""),
        ("왼쪽이 왼발, 오른쪽이 오른발인 원본 오버레이", photo_overlay_png or b""),
    ]

    try:
        data = await _create_structured_response(
            system_prompt=(
                "너는 발 이미지 AI 분석 결과를 사용자에게 설명하는 한국어 리포트 작성자다. "
                "제공된 점수, 위치 요약, 분석 이미지만 근거로 짧고 구체적인 문구를 작성한다."
            ),
            user_payload=payload,
            schema_name="tinea_report_text",
            json_schema=schema,
            images=images,
            max_output_tokens=700,
        )
    except Exception as exc:
        logger.warning("GPT tinea report text generation failed; using fallback. error=%s", exc)
        return fallback

    return TineaReportText(
        fungal_suspicion_safety_description=_compact_text(
            data.get("fungal_suspicion_safety_description"),
            fallback.fungal_suspicion_safety_description,
        ),
        skin_reaction_safety_description=_compact_text(
            data.get("skin_reaction_safety_description"),
            fallback.skin_reaction_safety_description,
        ),
        total_score_description=_compact_text(
            data.get("total_score_description"),
            fallback.total_score_description,
            max_chars=180,
        ),
    )


async def generate_hallux_score_analysis_text(left_angle: float, right_angle: float, fallback_text: str) -> str:
    fallback = _compact_text(fallback_text, fallback_text, max_chars=160)
    if not _openai_enabled():
        return fallback

    schema = {
        "type": "object",
        "properties": {"score_analysis_text": {"type": "string"}},
        "required": ["score_analysis_text"],
        "additionalProperties": False,
    }
    payload = {
        "task": "무지외반 HVA 각도 리포트 한 줄 문구 생성",
        "left_toe_angle_degree": round(float(left_angle), 1),
        "right_toe_angle_degree": round(float(right_angle), 1),
        "angle_rule": {
            "normal": "15도 미만",
            "borderline": "15도 이상 20도 미만",
            "progressed": "20도 이상 40도 미만",
            "severe": "40도 이상",
        },
        "style": [
            "한국어 한 문장만 작성한다.",
            "더 높은 쪽과 각도를 포함한다.",
            "양쪽 모두 15도 이상이면 양발 모두를 언급한다.",
            "진단 확정이 아니라 관리/주의 필요성으로 표현한다.",
        ],
        "fallback_text": fallback,
    }

    try:
        data = await _create_structured_response(
            system_prompt=(
                "너는 무지외반 AI 분석 결과를 사용자에게 설명하는 한국어 리포트 작성자다. "
                "HVA 각도만 근거로 간단한 scoreAnalysisText 한 문장을 작성한다."
            ),
            user_payload=payload,
            schema_name="hallux_report_text",
            json_schema=schema,
            images=None,
            max_output_tokens=160,
        )
    except Exception as exc:
        logger.warning("GPT hallux report text generation failed; using fallback. error=%s", exc)
        return fallback

    return _compact_text(data.get("score_analysis_text"), fallback, max_chars=160)
