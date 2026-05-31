from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "좌우 발 사진을 업로드하면 AI가 무좀 의심 영역과 염증/각질 관련 징후를 분석하고, "
        "의심 영역 지도와 원본 오버레이 이미지를 생성해 리포트 서버로 전송합니다."
    ),
)

app.include_router(api_router, prefix="/api")


@app.get("/health", tags=["상태 확인"], summary="서버 상태 확인")
def health_check():
    return {"status": "ok"}
