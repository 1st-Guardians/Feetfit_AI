import logging

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.services.shoe.shoe_embedding import embedding_runtime_status
from app.services.shoe.shoe_fit_comment_service import ollama_runtime_status


logger = logging.getLogger(__name__)
_runtime_preflight_logged = False


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "좌우 발 사진을 업로드하면 AI가 무좀 의심 영역과 염증/각질 관련 징후를 분석하고, "
        "의심 영역 지도와 원본 오버레이 이미지를 생성해 리포트 서버로 전송합니다."
    ),
)

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
async def log_ai_runtime_preflight() -> None:
    global _runtime_preflight_logged
    if _runtime_preflight_logged:
        return
    status = embedding_runtime_status()
    ollama = await ollama_runtime_status()
    logger.info(
        "Phase D runtime: CUDA available=%s GPU name=%s PyTorch CUDA version=%s "
        "BGE-M3 device=%s Ollama reachable=%s model=%s GPU in use=%s VRAM bytes=%s",
        status["cudaAvailable"],
        status["gpuName"],
        status["pytorchCudaVersion"],
        status["bgeM3ResolvedDevice"],
        ollama["reachable"],
        ollama["model"],
        ollama["gpuInUse"],
        ollama.get("sizeVram"),
    )
    _runtime_preflight_logged = True


@app.get("/health", tags=["상태 확인"], summary="서버 상태 확인")
def health_check():
    return {"status": "ok"}
