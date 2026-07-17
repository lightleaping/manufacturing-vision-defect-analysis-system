"""Day 7 FastAPI를 호출하는 Day 8 Dashboard 전용 HTTP Client.

Client는 API 응답을 Dataclass로 변환하고, 연결 실패·Timeout·잘못된 JSON·
Schema 불일치와 FastAPI 오류 응답을 안전한 Dashboard 오류로 통일한다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from src.dashboard.config import DashboardSettings


SAFE_ERROR_MESSAGES: dict[str, str] = {
    "MISSING_FILE": "분석할 이미지 파일이 필요합니다.",
    "EMPTY_FILE": "업로드한 이미지 파일이 비어 있습니다.",
    "UNSUPPORTED_FILE_TYPE": "JPEG 또는 PNG 이미지만 업로드할 수 있습니다.",
    "FILE_TOO_LARGE": "업로드 파일 크기가 허용 범위를 초과했습니다.",
    "IMAGE_TOO_LARGE": "이미지 해상도가 허용 범위를 초과했습니다.",
    "INVALID_IMAGE": "업로드한 파일을 정상적인 이미지로 읽을 수 없습니다.",
    "MODEL_NOT_READY": "FastAPI 추론 모델이 아직 준비되지 않았습니다.",
    "INVALID_MODEL_INPUT": "모델 입력 형식이 올바르지 않습니다.",
    "INVALID_MODEL_OUTPUT": "모델 출력 형식이 올바르지 않습니다.",
    "INFERENCE_FAILED": "이미지 추론 중 내부 오류가 발생했습니다.",
    "API_CONNECTION_ERROR": "FastAPI 서버에 연결할 수 없습니다.",
    "API_TIMEOUT": "FastAPI 응답 시간이 초과되었습니다.",
    "API_INVALID_RESPONSE": "FastAPI 응답 형식을 확인할 수 없습니다.",
    "API_REQUEST_ERROR": "FastAPI 요청을 처리하지 못했습니다.",
}


class DashboardApiError(RuntimeError):
    """Streamlit 화면에 안전하게 표시할 수 있는 API 오류."""

    def __init__(
        self,
        *,
        code: str,
        message: str | None = None,
        status_code: int | None = None,
    ) -> None:
        del message
        safe_message = SAFE_ERROR_MESSAGES.get(
            code,
            "요청을 처리하지 못했습니다.",
        )
        super().__init__(safe_message)
        self.code = code
        self.message = safe_message
        self.status_code = status_code

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "status_code": self.status_code,
        }


@dataclass(frozen=True, slots=True)
class DashboardHealth:
    status: str
    service: str
    model_loaded: bool
    model_name: str
    device: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DashboardPrediction:
    prediction: int
    prediction_class_name: str
    defect_probability: float
    normal_probability: float
    raw_logit: float
    classification_threshold: float
    model_name: str
    model_version: str
    positive_class: str
    original_filename: str
    content_type: str
    image_width: int
    image_height: int
    image_mode: str
    inference_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_mapping(payload: object, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    return payload


def _require_key(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    return payload[key]


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = _require_key(payload, key)
    if not isinstance(value, str) or not value.strip():
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    return value.strip()


def _require_boolean(payload: dict[str, Any], key: str) -> bool:
    value = _require_key(payload, key)
    if not isinstance(value, bool):
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    return value


def _require_integer(payload: dict[str, Any], key: str) -> int:
    value = _require_key(payload, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = _require_key(payload, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    result = float(value)
    if not math.isfinite(result):
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    return result


def parse_health_payload(payload: object) -> DashboardHealth:
    """Health JSON의 필수 Key와 Type을 검증한다."""

    data = _require_mapping(payload, name="health")
    return DashboardHealth(
        status=_require_string(data, "status"),
        service=_require_string(data, "service"),
        model_loaded=_require_boolean(data, "model_loaded"),
        model_name=_require_string(data, "model_name"),
        device=_require_string(data, "device"),
    )


def parse_prediction_payload(payload: object) -> DashboardPrediction:
    """Prediction JSON을 검증하고 명시적 Dataclass로 변환한다."""

    data = _require_mapping(payload, name="prediction")

    prediction = _require_integer(data, "prediction")
    class_name = _require_string(data, "prediction_class_name")
    defect_probability = _require_float(data, "defect_probability")
    normal_probability = _require_float(data, "normal_probability")
    raw_logit = _require_float(data, "raw_logit")
    threshold = _require_float(data, "classification_threshold")
    image_width = _require_integer(data, "image_width")
    image_height = _require_integer(data, "image_height")
    inference_time_ms = _require_float(data, "inference_time_ms")

    if prediction not in {0, 1}:
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    if class_name not in {"NORMAL", "DEFECT"}:
        raise DashboardApiError(code="API_INVALID_RESPONSE")
    if (prediction, class_name) not in {(0, "NORMAL"), (1, "DEFECT")}:
        raise DashboardApiError(code="API_INVALID_RESPONSE")

    for probability in (defect_probability, normal_probability, threshold):
        if not 0.0 <= probability <= 1.0:
            raise DashboardApiError(code="API_INVALID_RESPONSE")

    if not math.isclose(
        defect_probability + normal_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise DashboardApiError(code="API_INVALID_RESPONSE")

    expected_prediction = 1 if defect_probability >= threshold else 0
    if expected_prediction != prediction:
        raise DashboardApiError(code="API_INVALID_RESPONSE")

    if image_width <= 0 or image_height <= 0 or inference_time_ms < 0.0:
        raise DashboardApiError(code="API_INVALID_RESPONSE")

    positive_class = _require_string(data, "positive_class")
    if positive_class != "DEFECT":
        raise DashboardApiError(code="API_INVALID_RESPONSE")

    return DashboardPrediction(
        prediction=prediction,
        prediction_class_name=class_name,
        defect_probability=defect_probability,
        normal_probability=normal_probability,
        raw_logit=raw_logit,
        classification_threshold=threshold,
        model_name=_require_string(data, "model_name"),
        model_version=_require_string(data, "model_version"),
        positive_class=positive_class,
        original_filename=_require_string(data, "original_filename"),
        content_type=_require_string(data, "content_type"),
        image_width=image_width,
        image_height=image_height,
        image_mode=_require_string(data, "image_mode"),
        inference_time_ms=inference_time_ms,
    )


def _parse_json_response(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError as exc:
        raise DashboardApiError(
            code="API_INVALID_RESPONSE",
            status_code=response.status_code,
        ) from exc


def _raise_api_error(response: httpx.Response) -> None:
    """FastAPI 공통 오류 Schema를 해석하되 내부 문자열은 노출하지 않는다."""

    code = "API_REQUEST_ERROR"
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            candidate = detail.get("code")
            if isinstance(candidate, str) and candidate.strip():
                code = candidate.strip()

    if code not in SAFE_ERROR_MESSAGES:
        if response.status_code == 503:
            code = "MODEL_NOT_READY"
        elif response.status_code == 408:
            code = "API_TIMEOUT"
        else:
            code = "API_REQUEST_ERROR"

    raise DashboardApiError(code=code, status_code=response.status_code)


class DashboardApiClient:
    """Streamlit에서 사용하는 동기식 FastAPI Client."""

    def __init__(
        self,
        settings: DashboardSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        timeout = httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.read_timeout_seconds,
            write=settings.read_timeout_seconds,
            pool=settings.connect_timeout_seconds,
        )
        self._client = httpx.Client(
            base_url=settings.api_base_url,
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def __enter__(self) -> "DashboardApiClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_health(self) -> DashboardHealth:
        try:
            response = self._client.get(self.settings.health_endpoint)
        except httpx.TimeoutException as exc:
            raise DashboardApiError(code="API_TIMEOUT") from exc
        except httpx.ConnectError as exc:
            raise DashboardApiError(code="API_CONNECTION_ERROR") from exc
        except httpx.RequestError as exc:
            raise DashboardApiError(code="API_REQUEST_ERROR") from exc

        if not response.is_success:
            _raise_api_error(response)
        return parse_health_payload(_parse_json_response(response))

    def predict_image(
        self,
        *,
        filename: str,
        content_type: str,
        image_bytes: bytes,
    ) -> DashboardPrediction:
        """이미지 Byte를 디스크 저장 없이 multipart/form-data로 전달한다."""

        if not filename.strip():
            raise ValueError("filename must not be empty")
        if not image_bytes:
            raise DashboardApiError(code="EMPTY_FILE")

        files = {
            "file": (
                filename,
                image_bytes,
                content_type,
            )
        }

        try:
            response = self._client.post(
                self.settings.prediction_endpoint,
                files=files,
            )
        except httpx.TimeoutException as exc:
            raise DashboardApiError(code="API_TIMEOUT") from exc
        except httpx.ConnectError as exc:
            raise DashboardApiError(code="API_CONNECTION_ERROR") from exc
        except httpx.RequestError as exc:
            raise DashboardApiError(code="API_REQUEST_ERROR") from exc

        if not response.is_success:
            _raise_api_error(response)
        return parse_prediction_payload(_parse_json_response(response))
