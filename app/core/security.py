from fastapi.security import HTTPBearer


bearer_scheme = HTTPBearer(
    scheme_name="Bearer 토큰 인증",
    bearerFormat="JWT",
    description="리포트 서버로 전달할 JWT Bearer 토큰을 입력하세요.",
)
