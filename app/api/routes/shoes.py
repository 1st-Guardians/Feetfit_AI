from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, Security, status
from fastapi.security import HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import bearer_scheme
from app.schemas.shoes import ShoeSummaryForwardRequest, ShoeSummaryReasonPayload, ShoeSummaryTriggerRequest
from app.services.shoe_db import ShoeDbError, connect_shoe_db, fetch_saved_shoe_recommendation
from app.services.shoe_fit_comment_service import ReasonFactsForPrompt, ShoeFitCommentError, generate_shoe_summaries
from app.services.shoe_recommendation import risk_level_from_score

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/summaries",
    status_code=status.HTTP_202_ACCEPTED,
    summary="신발 상세 조회 시 착용 코멘트 생성 트리거",
    description=(
        "Feetfit_Server가 신발 상세 조회 응답의 pointSummary가 비어 있을 때 호출합니다 (fire-and-forget). "
        "이미 배치(/reports/shoe-recommendations)로 계산/저장되어 있는 fitScore·riskLevel·근거 리뷰를 "
        "다시 계산하지 않고 그대로 읽어와서, Ollama로 pointSummary와 부위별 reviewSummary 문장만 생성합니다. "
        "즉시 202를 반환하고, 실제 생성/저장은 백그라운드에서 진행한 뒤 완료되면 "
        "Feetfit_Server의 POST /api/shoes/{shoeId}/summaries로 결과를 저장 요청합니다."
    ),
)
async def trigger_shoe_summary_generation(
    request: ShoeSummaryTriggerRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    authorization_header = f"{credentials.scheme} {credentials.credentials}"
    background_tasks.add_task(
        _generate_and_save_summary, request.shoe_id, request.user_id, authorization_header
    )
    return {"accepted": True}


def _fetch_saved_recommendation(user_id: int, shoe_id: int):
    connection = connect_shoe_db()
    try:
        return fetch_saved_shoe_recommendation(connection, user_id, shoe_id)
    finally:
        connection.close()


async def _generate_and_save_summary(shoe_id: int, user_id: int, authorization_header: str) -> None:
    try:
        saved = await run_in_threadpool(_fetch_saved_recommendation, user_id, shoe_id)
    except ShoeDbError:
        logger.warning("shoe_id=%s user_id=%s: DB 조회 실패", shoe_id, user_id, exc_info=True)
        return

    if saved is None:
        logger.warning(
            "shoe_id=%s user_id=%s: 저장된 신발 적합도 데이터가 없습니다 (배치가 먼저 실행돼야 합니다).",
            shoe_id, user_id,
        )
        return

    try:
        summary = await generate_shoe_summaries(
            shoe_name=saved.shoe_name,
            fit_score=saved.fit_score,
            overall_risk_level=risk_level_from_score(saved.fit_score),
            reasons=[
                ReasonFactsForPrompt(
                    reason_type=reason.reason_type,
                    title=reason.title,
                    risk_level=reason.risk_level,
                    review_texts=reason.review_texts,
                )
                for reason in saved.reasons
            ],
        )
    except ShoeFitCommentError:
        logger.warning("shoe_id=%s user_id=%s: Ollama 코멘트 생성 실패", shoe_id, user_id, exc_info=True)
        return

    forward_request = ShoeSummaryForwardRequest(
        point_summary=summary.point_summary,
        reasons=[
            ShoeSummaryReasonPayload(reason_type="FOREFOOT", review_summary=summary.forefoot_summary),
            ShoeSummaryReasonPayload(reason_type="HEEL", review_summary=summary.heel_summary),
            ShoeSummaryReasonPayload(reason_type="INSOLE", review_summary=summary.insole_summary),
        ],
    )
    headers = {"accept": "*/*", "Authorization": authorization_header}
    save_url = settings.shoe_summary_save_endpoint_template.format(shoe_id=shoe_id)

    try:
        async with httpx.AsyncClient(timeout=settings.report_proxy_timeout_seconds) as client:
            response = await client.post(
                save_url,
                headers=headers,
                json=forward_request.model_dump(by_alias=True),
            )
            response.raise_for_status()
    except httpx.HTTPError:
        logger.warning("shoe_id=%s user_id=%s: 요약 저장 콜백 실패 (url=%s)", shoe_id, user_id, save_url, exc_info=True)
