from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "발 사진 1장을 업로드하면 AI가 무좀 의심 영역과 염증/자극 영역을 분석하고, "
        "의심 지도 이미지와 사진 오버레이 이미지를 생성해 리포트 서버로 전송합니다."
    ),
)

app.include_router(api_router, prefix="/api")


@app.get("/health", tags=["상태 확인"], summary="서버 상태 확인")
def health_check():
    return {"status": "ok"}
