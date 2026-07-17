from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError

from app.core.config import settings
from app.prompts.shoe_fit_comment_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.schemas.shoe_fit_comment import ShoeFitSummary

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_NO_REVIEW_NOTE = "없음 (판단만 근거로 사용하고, 구체적인 리뷰 내용을 지어내지 말 것)"

_REASON_LABEL = {"FOREFOOT": "발볼/앞코", "HEEL": "뒤꿈치", "INSOLE": "깔창/통기성"}

_llm: ChatOllama | None = None
_llm_lock = threading.Lock()


class ShoeFitCommentError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReasonFactsForPrompt:
    reason_type: str  # "FOREFOOT" | "HEEL" | "INSOLE"
    title: str
    risk_level: str
    review_texts: list[str]


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = ChatOllama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=settings.ollama_temperature,
                    format="json",
                    client_kwargs={"timeout": settings.ollama_request_timeout_seconds},
                )
    return _llm


def _format_reason_block(reason: ReasonFactsForPrompt) -> str:
    if reason.review_texts:
        review_lines = "\n".join(f'  {i + 1}. "{text}"' for i, text in enumerate(reason.review_texts))
    else:
        review_lines = f"  {_NO_REVIEW_NOTE}"

    label = _REASON_LABEL[reason.reason_type]
    return (
        f"- 항목: {label} ({reason.reason_type})\n"
        f"- 제목: {reason.title}\n"
        f"- 위험도: {reason.risk_level}\n"
        f"- 참고 리뷰 문장:\n{review_lines}"
    )


def _build_user_prompt(
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
) -> str:
    reason_by_type = {reason.reason_type: reason for reason in reasons}
    return USER_PROMPT_TEMPLATE.format(
        shoe_name=shoe_name,
        fit_score=fit_score,
        overall_risk_level=overall_risk_level,
        forefoot_block=_format_reason_block(reason_by_type["FOREFOOT"]),
        heel_block=_format_reason_block(reason_by_type["HEEL"]),
        insole_block=_format_reason_block(reason_by_type["INSOLE"]),
    )


def _parse_llm_json(raw_text: str) -> dict:
    """정상적인 JSON이면 그대로 파싱하고, 실패하면 텍스트 안에서 JSON 블록만 추출해 재파싱한다."""
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_PATTERN.search(text)
    if not match:
        raise ShoeFitCommentError(f"Ollama 응답에서 JSON을 찾을 수 없습니다: {text[:200]}")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ShoeFitCommentError(f"Ollama 응답 JSON 파싱에 실패했습니다: {text[:200]}") from exc


async def generate_shoe_summaries(
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
) -> ShoeFitSummary:
    """신발 하나의 fitScore/riskLevel/부위별 판단 결과를 바탕으로, Ollama로
    pointSummary + 부위별 reviewSummary(발볼/뒤꿈치/깔창)를 한 번의 호출로 생성한다."""
    user_prompt = _build_user_prompt(shoe_name, fit_score, overall_risk_level, reasons)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    try:
        response = await _get_llm().ainvoke(messages)
    except (httpx.HTTPError, ResponseError) as exc:
        raise ShoeFitCommentError(f"Ollama 요청에 실패했습니다: {exc}") from exc

    parsed = _parse_llm_json(response.content)
    return ShoeFitSummary.model_validate(parsed)
