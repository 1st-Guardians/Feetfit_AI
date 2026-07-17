from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import logging
from typing import Any

import httpx

from app.core.config import settings


logger = logging.getLogger(__name__)
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass(frozen=True)
class TineaReportText:
    fungal_suspicion_safety_description: str
    skin_reaction_safety_description: str
    total_score_description: str


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
