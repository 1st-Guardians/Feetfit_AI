import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status
from fastapi.security import HTTPBearer

from app.core.config import settings


bearer_scheme = HTTPBearer(
    scheme_name="Bearer 토큰 인증",
    bearerFormat="JWT",
    description="리포트 서버로 전달할 JWT Bearer 토큰을 입력하세요.",
)


def require_internal_api_key(
    provided: Annotated[str | None, Header(alias="X-Internal-Api-Key")] = None,
) -> None:
    expected = settings.feetfit_server_internal_api_key.strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feetfit AI internal API authentication is not configured.",
        )
    if provided is None or not hmac.compare_digest(provided, expected):
        # Missing and wrong keys deliberately use the same response and are never logged.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden internal API request.",
        )
