import json

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Response, Security, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import bearer_scheme
from app.schemas.reports import TineaPedisReportRequest
from app.services.tinea_analysis import AnalysisError, analyze_foot_image, combine_jpeg_images_side_by_side


router = APIRouter()


@router.post(
    "/tina-pedis",
    summary="왼발/오른발 사진 분석 후 무좀 리포트 저장 요청",
    description=(
        "왼발 사진과 오른발 사진을 각각 1장씩 업로드하면 서버 내부 AI가 무좀 의심 영역과 염증/자극 영역을 분석합니다. "
        "분석 후 양발의 원형 의심 지도를 좌우로 합친 이미지와, 양발의 사진 오버레이를 좌우로 합친 이미지를 생성해 "
        "외부 리포트 서버로 multipart 요청을 전송합니다. "
        "상단 Authorize 버튼에서 Bearer 토큰을 먼저 입력하세요."
    ),
    responses={
        200: {"description": "리포트 서버에서 반환한 응답입니다."},
        201: {"description": "리포트 서버에서 저장 성공 후 반환한 응답입니다."},
        400: {"description": "업로드한 이미지가 비어 있습니다."},
        413: {"description": "업로드한 이미지 용량이 너무 큽니다."},
        422: {"description": "요청 폼 데이터가 올바르지 않습니다."},
        500: {"description": "AI 분석 중 오류가 발생했습니다."},
        502: {"description": "리포트 서버 요청에 실패했습니다."},
    },
)
async def create_tinea_pedis_report(
    measurementSessionId: int = Form(1, ge=1, description="측정 세션 ID입니다. 기본값은 1입니다."),
    leftFootImage: UploadFile = File(..., description="분석할 왼발 원본 사진 1장을 업로드하세요."),
    rightFootImage: UploadFile = File(..., description="분석할 오른발 원본 사진 1장을 업로드하세요."),
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> Response:
    left_image_bytes = await read_upload_image(leftFootImage, "leftFootImage")
    right_image_bytes = await read_upload_image(rightFootImage, "rightFootImage")

    try:
        left_analysis = await run_in_threadpool(
            analyze_foot_image,
            left_image_bytes,
            leftFootImage.filename,
        )
        right_analysis = await run_in_threadpool(
            analyze_foot_image,
            right_image_bytes,
            rightFootImage.filename,
        )
        combined_suspicion_map = await run_in_threadpool(
            combine_jpeg_images_side_by_side,
            left_analysis.suspicion_map_jpeg,
            right_analysis.suspicion_map_jpeg,
        )
        combined_photo_overlay = await run_in_threadpool(
            combine_jpeg_images_side_by_side,
            left_analysis.photo_overlay_jpeg,
            right_analysis.photo_overlay_jpeg,
        )
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    report_request = TineaPedisReportRequest(
        measurement_session_id=measurementSessionId,
        fungal_suspicion_safety_score=min(left_analysis.fungal_safety_score, right_analysis.fungal_safety_score),
        skin_reaction_safety_score=min(
            left_analysis.skin_reaction_safety_score,
            right_analysis.skin_reaction_safety_score,
        ),
    )
    forwarded_request = json.dumps(report_request.model_dump(by_alias=True), ensure_ascii=False)

    files = {
        "suspiciousAreaMapImage": (
            "left_right_suspicion_map.jpg",
            combined_suspicion_map,
            "image/jpeg",
        ),
        "originalFootImage": (
            "left_right_photo_overlay.jpg",
            combined_photo_overlay,
            "image/jpeg",
        ),
    }
    headers = {
        "accept": "*/*",
        "Authorization": f"{credentials.scheme} {credentials.credentials}",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.report_proxy_timeout_seconds) as client:
            upstream_response = await client.post(
                settings.tinea_report_endpoint,
                headers=headers,
                data={"request": forwarded_request},
                files=files,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Report server request failed: {exc}",
        ) from exc

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type"),
    )


async def read_upload_image(upload: UploadFile, field_name: str) -> bytes:
    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} 이미지가 비어 있습니다.",
        )
    if len(image_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{field_name} 이미지 용량이 {settings.max_upload_size_bytes} bytes를 초과했습니다.",
        )
    return image_bytes
