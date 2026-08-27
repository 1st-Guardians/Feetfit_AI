from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, BackgroundTasks, Security, status
from fastapi.security import HTTPAuthorizationCredentials

from app.core.security import bearer_scheme, require_internal_api_key
from app.schemas.shoes import ShoeSummaryForwardRequest, ShoeSummaryReasonPayload, ShoeSummaryTriggerRequest
from app.services.shoe.shoe_fit_comment_service import ReasonFactsForPrompt, ShoeFitCommentError, generate_shoe_summaries
from app.services.shoe.shoe_recommendation import risk_level_from_score
from app.services.shoe.shoe_server_client import ShoeServerClient, ShoeServerClientError

logger = logging.getLogger(__name__)

router = APIRouter()
_summary_inflight: set[tuple[int, int]] = set()
_summary_inflight_lock = threading.Lock()


@router.post(
    "/summaries",
    status_code=status.HTTP_202_ACCEPTED,
    summary="신발 상세 조회 시 착용 코멘트 생성 트리거",
    description=(
        "Feetfit_Server가 신발 상세 조회 응답의 pointSummary가 비어 있을 때 호출합니다 (fire-and-forget). "
        "이미 배치(/reports/shoe-recommendations)로 계산/저장되어 있는 fitScore·riskLevel·근거 리뷰를 "
        "JWT 범위의 Server 내부 API에서 다시 계산하지 않고 읽어와서, Ollama로 pointSummary와 부위별 "
        "reviewSummary 문장만 생성합니다. "
        "즉시 202를 반환하고, 실제 생성/저장은 백그라운드에서 진행한 뒤 완료되면 "
        "Feetfit_Server의 POST /api/shoes/{shoeId}/summaries로 결과를 저장 요청합니다."
    ),
)
async def trigger_shoe_summary_generation(
    request: ShoeSummaryTriggerRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    _internal_api_key: None = Security(require_internal_api_key),
) -> dict:
    authorization_header = f"{credentials.scheme} {credentials.credentials}"
    key = (request.measurement_session_id, request.shoe_id)
    with _summary_inflight_lock:
        if key in _summary_inflight:
            return {"accepted": True, "deduplicated": True}
        _summary_inflight.add(key)
    background_tasks.add_task(
        _generate_and_save_summary,
        request.shoe_id,
        request.measurement_session_id,
        authorization_header,
    )
    return {"accepted": True, "deduplicated": False}


async def _generate_and_save_summary(
    shoe_id: int, measurement_session_id: int, authorization_header: str
) -> None:
    """BackgroundTasks 경계에서는 모든 예외를 흡수한다.

    이 엔드포인트는 호출자에게 이미 202를 반환한 뒤 실행되므로, 예상하지 못한
    Ollama/계약 오류도 ASGI background-task 예외로 다시 전파하지 않는다.
    """
    try:
        await _generate_and_save_summary_inner(
            shoe_id, measurement_session_id, authorization_header
        )
    except Exception:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: 요약 백그라운드 작업에서 예상하지 못한 오류",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
    finally:
        with _summary_inflight_lock:
            _summary_inflight.discard((measurement_session_id, shoe_id))


async def _generate_and_save_summary_inner(
    shoe_id: int, measurement_session_id: int, authorization_header: str
) -> None:
    try:
        server_client = ShoeServerClient(authorization_header)
        saved = await server_client.fetch_saved_recommendation(
            measurement_session_id, shoe_id
        )
        characteristics = await server_client.fetch_shoe_characteristics(shoe_id)
    except ShoeServerClientError:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: Server 요약 context 조회 실패",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
        return

    if saved is None:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: 저장된 신발 적합도 데이터가 없습니다 (배치가 먼저 실행돼야 합니다).",
            shoe_id,
            measurement_session_id,
        )
        return

    try:
        summary = await generate_shoe_summaries(
            shoe_name=f"{saved.brand_name} {saved.shoe_name}",
            fit_score=saved.fit_score,
            overall_risk_level=risk_level_from_score(saved.fit_score),
            reasons=[
                ReasonFactsForPrompt(
                    reason_type=reason.reason_type,
                    title=reason.title,
                    risk_level=reason.risk_level,
                    review_ids=[review.id for review in reason.reviews],
                    review_texts=[review.review_text for review in reason.reviews],
                )
                for reason in saved.reasons
            ],
            characteristics=characteristics,
        )
    except ShoeFitCommentError:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: Ollama 코멘트 생성 실패",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
        return

    forward_request = ShoeSummaryForwardRequest(
        measurement_session_id=measurement_session_id,
        point_summary=summary.point_summary,
        reasons=[
            ShoeSummaryReasonPayload(
                reason_type="FOREFOOT",
                review_summary=summary.forefoot_summary,
                review_ids=summary.forefoot_review_ids,
            ),
            ShoeSummaryReasonPayload(
                reason_type="HEEL",
                review_summary=summary.heel_summary,
                review_ids=summary.heel_review_ids,
            ),
            ShoeSummaryReasonPayload(
                reason_type="INSOLE",
                review_summary=summary.insole_summary,
                review_ids=summary.insole_review_ids,
            ),
        ],
    )
    try:
        await server_client.save_summary(shoe_id, forward_request)
    except ShoeServerClientError:
        logger.warning(
            "shoe_id=%s measurement_session_id=%s: 요약 저장 콜백 실패",
            shoe_id,
            measurement_session_id,
            exc_info=True,
        )
