from __future__ import annotations

from decimal import Decimal
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.schemas.shoe_server import (
    ServerApiResponse,
    ServerRecommendationContext,
    ServerRecommendationContextPage,
    ServerSavedRecommendation,
    ServerShoeCharacteristics,
)


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class ShoeServerClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ShoeServerConfigurationError(ShoeServerClientError):
    """Feetfit_AI의 로컬 Server 연동 설정이 유효하지 않을 때 발생한다."""


class ShoeServerClient:
    """Feetfit_Server의 AI용 계약을 호출한다.

    Authorization은 사용자의 원래 Bearer 값을 그대로 전달한다. 내부 조회가 실패해도
    shared DB 조회로 되돌아가지 않는다.
    """

    def __init__(
        self,
        authorization_header: str,
        *,
        internal_api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_internal_api_key = (
            settings.feetfit_server_internal_api_key
            if internal_api_key is None
            else internal_api_key
        ).strip()
        if not resolved_internal_api_key:
            raise ShoeServerConfigurationError(
                "FEETFIT_SERVER_INTERNAL_API_KEY must be configured before calling Feetfit_Server."
            )
        self._headers = {
            "accept": "application/json",
            "Authorization": authorization_header,
            "X-Internal-Api-Key": resolved_internal_api_key,
        }
        self._http_client = http_client

    async def fetch_recommendation_context(
        self, measurement_session_id: int
    ) -> ServerRecommendationContext:
        first_page: ServerRecommendationContextPage | None = None
        shoes = []
        seen_shoe_ids: set[int] = set()
        page = 0

        while True:
            response = await self._request(
                "GET",
                settings.shoe_recommendation_context_endpoint,
                params={
                    "measurementSessionId": measurement_session_id,
                    "page": page,
                    "size": settings.shoe_recommendation_context_page_size,
                },
            )
            result = self._parse_result(response, ServerRecommendationContextPage)
            self._validate_context_page(result, measurement_session_id, page, first_page)

            for shoe in result.shoes:
                if shoe.id in seen_shoe_ids:
                    raise ShoeServerClientError(
                        f"Feetfit_Server returned duplicate shoeId={shoe.id} across context pages."
                    )
                seen_shoe_ids.add(shoe.id)
                shoes.append(shoe)

            if first_page is None:
                first_page = result
            if not result.has_next:
                break
            page += 1

        if first_page is None:  # pragma: no cover - the loop always performs one request
            raise ShoeServerClientError("Feetfit_Server recommendation context is empty.")
        if len(shoes) != first_page.total_elements:
            raise ShoeServerClientError(
                "Feetfit_Server pagination totalElements does not match the collected shoes "
                f"(expected={first_page.total_elements}, collected={len(shoes)})."
            )

        return ServerRecommendationContext(
            measurement_session_id=first_page.measurement_session_id,
            user_id=first_page.user_id,
            measurement_status=first_page.measurement_status,
            foot_state=first_page.foot_state,
            shoes=shoes,
        )

    async def fetch_saved_recommendation(
        self, measurement_session_id: int, shoe_id: int
    ) -> ServerSavedRecommendation | None:
        url = settings.shoe_summary_context_endpoint_template.format(shoe_id=shoe_id)
        response = await self._request(
            "GET", url, params={"measurementSessionId": measurement_session_id}
        )
        if response.status_code == 404:
            error_code, error_message = self._parse_error(response)
            if error_code == "SHOE4005":
                return None
            detail = f" ({error_code}: {error_message})" if error_code else ""
            raise ShoeServerClientError(
                f"Feetfit_Server returned HTTP 404{detail}.",
                status_code=404,
            )
        result = self._parse_result(response, ServerSavedRecommendation)
        if (
            result.measurement_session_id != measurement_session_id
            or result.shoe_id != shoe_id
        ):
            raise ShoeServerClientError(
                "Feetfit_Server summary context does not match the requested "
                "measurementSessionId/shoeId."
            )
        return result

    async def fetch_shoe_characteristics(
        self, shoe_id: int
    ) -> ServerShoeCharacteristics:
        url = settings.shoe_characteristics_endpoint_template.format(shoe_id=shoe_id)
        response = await self._request("GET", url)
        result = self._parse_result(response, ServerShoeCharacteristics)
        if result.shoe_id != shoe_id:
            raise ShoeServerClientError(
                "Feetfit_Server characteristics do not match the requested shoeId."
            )
        return result

    @staticmethod
    def _validate_context_page(
        result: ServerRecommendationContextPage,
        requested_session_id: int,
        requested_page: int,
        first_page: ServerRecommendationContextPage | None,
    ) -> None:
        if result.measurement_session_id != requested_session_id:
            raise ShoeServerClientError(
                "Feetfit_Server returned a different measurementSessionId "
                f"(requested={requested_session_id}, returned={result.measurement_session_id})."
            )
        if result.current_page != requested_page:
            raise ShoeServerClientError(
                "Feetfit_Server returned an unexpected context page "
                f"(requested={requested_page}, returned={result.current_page})."
            )
        if result.total_pages == 0:
            pagination_is_consistent = (
                result.current_page == 0
                and result.total_elements == 0
                and not result.shoes
                and not result.has_next
            )
        else:
            pagination_is_consistent = (
                result.current_page < result.total_pages
                and result.has_next == (result.current_page + 1 < result.total_pages)
            )
        if not pagination_is_consistent:
            raise ShoeServerClientError("Feetfit_Server returned inconsistent pagination metadata.")
        if first_page is None:
            return
        if (
            result.user_id != first_page.user_id
            or result.measurement_status != first_page.measurement_status
            or result.foot_state != first_page.foot_state
            or result.total_pages != first_page.total_pages
            or result.total_elements != first_page.total_elements
        ):
            raise ShoeServerClientError(
                "Feetfit_Server changed session metadata while paging recommendation context."
            )

    async def forward_recommendations(self, payload: BaseModel) -> httpx.Response:
        return await self._request(
            "POST",
            settings.shoe_recommendation_endpoint,
            json=payload.model_dump(by_alias=True),
            _timeout_seconds=settings.feetfit_server_callback_timeout_seconds,
        )

    async def save_summary(self, shoe_id: int, payload: BaseModel) -> None:
        url = settings.shoe_summary_save_endpoint_template.format(shoe_id=shoe_id)
        response = await self._request(
            "POST",
            url,
            json=payload.model_dump(by_alias=True),
            _timeout_seconds=settings.feetfit_server_callback_timeout_seconds,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ShoeServerClientError(
                f"Feetfit_Server summary callback returned HTTP {response.status_code}.",
                status_code=response.status_code,
            ) from exc

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        timeout_seconds = kwargs.pop(
            "_timeout_seconds", settings.report_proxy_timeout_seconds
        )
        try:
            if self._http_client is not None:
                return await self._http_client.request(method, url, headers=self._headers, **kwargs)
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                return await client.request(method, url, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ShoeServerClientError(f"Feetfit_Server request failed: {exc}") from exc

    @staticmethod
    def _parse_result(
        response: httpx.Response, result_model: type[ResponseModelT]
    ) -> ResponseModelT:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_code, error_message = ShoeServerClient._parse_error(response)
            detail = (
                f" ({error_code}: {error_message})"
                if error_code is not None and error_message is not None
                else ""
            )
            raise ShoeServerClientError(
                f"Feetfit_Server returned HTTP {response.status_code}{detail}.",
                status_code=response.status_code,
            ) from exc

        try:
            envelope_type = ServerApiResponse[result_model]
            # Jackson의 BigDecimal JSON number를 float로 먼저 축소하지 않는다.
            envelope = envelope_type.model_validate(response.json(parse_float=Decimal))
        except (ValueError, ValidationError) as exc:
            raise ShoeServerClientError(
                "Feetfit_Server returned an invalid ApiResponse contract."
            ) from exc

        if not envelope.is_success:
            raise ShoeServerClientError(
                f"Feetfit_Server rejected the request ({envelope.code}: {envelope.message}).",
                status_code=response.status_code,
            )
        if envelope.result is None:
            raise ShoeServerClientError("Feetfit_Server ApiResponse.result is missing.")
        return envelope.result

    @staticmethod
    def _parse_error(response: httpx.Response) -> tuple[str | None, str | None]:
        """오류 ApiResponse가 유효할 때만 code/message를 돌려준다."""
        try:
            payload = response.json(parse_float=Decimal)
        except ValueError:
            return None, None
        if not isinstance(payload, dict) or payload.get("isSuccess") is not False:
            return None, None
        code = payload.get("code")
        message = payload.get("message")
        return (
            code if isinstance(code, str) else None,
            message if isinstance(message, str) else None,
        )
