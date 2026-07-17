from fastapi import APIRouter

from app.api.routes import reports, shoes


api_router = APIRouter()
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(shoes.router, prefix="/shoes", tags=["shoes"])
