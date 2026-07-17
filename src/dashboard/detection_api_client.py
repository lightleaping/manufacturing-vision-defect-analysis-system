"""Day 13 Detection FastAPI 전용 Streamlit HTTP Client.

[기존 코드 참고]
``src.dashboard.api_client``의 동기식 httpx Client, 안전한 오류 변환,
명시적 Dataclass 변환 방식을 따른다.

[신규 구현]
- ``POST /api/v1/detection/predictions`` 호출
- Score Threshold Query Parameter 전달
- Detection Response의 Class·Score·Box·Model Metadata 검증
- Timeout·연결 실패·4xx·5xx·잘못된 JSON·Schema 누락 방어
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from src.dashboard.config import DashboardSettings


DETECTION_PREDICTION_ENDPOINT = "/api/v1/detection/predictions"

DETECTION_CLASS_NAMES: dict[int, str] = {
    1: "crazing",
    2: "inclusion",
    3: "patches",
    4: "pitted_surface",
    5: "rolled_in_scale",
    6: "scratches",
}

SAFE_DETECTION_ERROR_MESSAGES: dict[str, str] = {
    "MISSING_FILE": "Detection을 수행할 이미지 파일이 필요합니다.",
    "EMPTY_FILE": "업로드한 이미지 파일이 비어 있습니다.",
    "UNSUPPORTED_FILE_TYPE": "JPEG 또는 PNG 이미지만 업로드할 수 있습니다.",
    "FILE_TOO_LARGE": "업로드 파일 크기가 허용 범위를 초과했습니다.",
    "IMAGE_TOO_LARGE": "이미지 해상도가 허용 범위를 초과했습니다.",
    "INVALID_IMAGE": "업로드한 파일을 정상적인 이미지로 읽을 수 없습니다.",
    "DETECTION_MODEL_NOT_READY": "Detection 모델이 아직 준비되지 않았습니다.",
    "INVALID_SCORE_THRESHOLD": "Score Threshold는 0.05부터 0.95 사이여야 합니다.",
    "INVALID_DETECTION_MODEL_INPUT": "Detection 모델 입력 형식이 올바르지 않습니다.",
    "INVALID_DETECTION_MODEL_OUTPUT": "Detection 모델 출력 형식이 올바르지 않습니다.",
    "DETECTION_INFERENCE_FAILED": "Detection 추론 중 내부 오류가 발생했습니다.",
    "API_CONNECTION_ERROR": "FastAPI 서버에 연결할 수 없습니다.",
    "API_TIMEOUT": "FastAPI Detection 응답 시간이 초과되었습니다.",
    "API_INVALID_RESPONSE": "FastAPI Detection 응답 형식을 확인할 수 없습니다.",
    "API_REQUEST_ERROR": "FastAPI Detection 요청을 처리하지 못했습니다.",
}


class DetectionDashboardApiError(RuntimeError):
    """Detection 페이지에 안전하게 표시할 수 있는 API 오류."""

    def __init__(
        self,
        *,
        code: str,
        status_code: int | None = None,
    ) -> None:
        message = SAFE_DETECTION_ERROR_MESSAGES.get(
            code,
            "Detection 요청을 처리하지 못했습니다.",
        )
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "status_code": self.status_code,
        }


@dataclass(frozen=True, slots=True)
class DashboardDetectionBox:
    """원본 업로드 이미지 좌표계의 XYXY Bounding Box."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DashboardDetection:
    """한 개의 결함 Detection Prediction."""

    label_id: int
    label_name: str
    score: float
    box: DashboardDetectionBox

    def to_dict(self) -> dict[str, object]:
        return {
            "label_id": self.label_id,
            "label_name": self.label_name,
            "score": self.score,
            "box": self.box.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DashboardDetectionPrediction:
    """Detection API 응답 전체를 표현하는 Dashboard Dataclass."""

    detections: tuple[DashboardDetection, ...]
    detection_count: int

    score_threshold: float
    iou_threshold: float

    model_name: str
    model_version: str
    architecture: str
    device: str
    checkpoint_epoch: int
    checkpoint_metric_name: str
    checkpoint_metric_value: float

    original_filename: str
    content_type: str
    image_width: int
    image_height: int
    image_mode: str
    model_input_mode: str
    inference_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "detections": [
                detection.to_dict()
                for detection in self.detections
            ],
            "detection_count": self.detection_count,
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "device": self.device,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_metric_name": self.checkpoint_metric_name,
            "checkpoint_metric_value": self.checkpoint_metric_value,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_mode": self.image_mode,
            "model_input_mode": self.model_input_mode,
            "inference_time_ms": self.inference_time_ms,
        }


def _invalid_response(
    *,
    status_code: int | None = None,
) -> DetectionDashboardApiError:
    return DetectionDashboardApiError(
        code="API_INVALID_RESPONSE",
        status_code=status_code,
    )


def _require_mapping(
    payload: object,
    *,
    status_code: int | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise _invalid_response(status_code=status_code)
    return payload


def _require_key(
    payload: dict[str, Any],
    key: str,
    *,
    status_code: int | None = None,
) -> Any:
    if key not in payload:
        raise _invalid_response(status_code=status_code)
    return payload[key]


def _require_string(
    payload: dict[str, Any],
    key: str,
    *,
    status_code: int | None = None,
) -> str:
    value = _require_key(
        payload,
        key,
        status_code=status_code,
    )
    if not isinstance(value, str) or not value.strip():
        raise _invalid_response(status_code=status_code)
    return value.strip()


def _require_integer(
    payload: dict[str, Any],
    key: str,
    *,
    status_code: int | None = None,
) -> int:
    value = _require_key(
        payload,
        key,
        status_code=status_code,
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_response(status_code=status_code)
    return value


def _require_float(
    payload: dict[str, Any],
    key: str,
    *,
    status_code: int | None = None,
) -> float:
    value = _require_key(
        payload,
        key,
        status_code=status_code,
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid_response(status_code=status_code)
    result = float(value)
    if not math.isfinite(result):
        raise _invalid_response(status_code=status_code)
    return result


def validate_dashboard_score_threshold(
    value: float,
) -> float:
    """Dashboard에서 전달할 Score Threshold 범위를 검증한다."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score_threshold must be numeric")

    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("score_threshold must be finite")
    if not 0.05 <= threshold <= 0.95:
        raise ValueError(
            "score_threshold must be between 0.05 and 0.95"
        )
    return threshold


def _parse_detection_box(
    payload: object,
    *,
    image_width: int,
    image_height: int,
    status_code: int | None = None,
) -> DashboardDetectionBox:
    data = _require_mapping(
        payload,
        status_code=status_code,
    )
    xmin = _require_float(
        data,
        "xmin",
        status_code=status_code,
    )
    ymin = _require_float(
        data,
        "ymin",
        status_code=status_code,
    )
    xmax = _require_float(
        data,
        "xmax",
        status_code=status_code,
    )
    ymax = _require_float(
        data,
        "ymax",
        status_code=status_code,
    )

    if (
        xmin < 0.0
        or ymin < 0.0
        or xmax > float(image_width)
        or ymax > float(image_height)
        or xmin >= xmax
        or ymin >= ymax
    ):
        raise _invalid_response(status_code=status_code)

    return DashboardDetectionBox(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )


def _parse_detection(
    payload: object,
    *,
    image_width: int,
    image_height: int,
    score_threshold: float,
    status_code: int | None = None,
) -> DashboardDetection:
    data = _require_mapping(
        payload,
        status_code=status_code,
    )
    label_id = _require_integer(
        data,
        "label_id",
        status_code=status_code,
    )
    label_name = _require_string(
        data,
        "label_name",
        status_code=status_code,
    )
    score = _require_float(
        data,
        "score",
        status_code=status_code,
    )

    expected_name = DETECTION_CLASS_NAMES.get(label_id)
    if expected_name is None or label_name != expected_name:
        raise _invalid_response(status_code=status_code)
    if not score_threshold <= score <= 1.0:
        raise _invalid_response(status_code=status_code)

    return DashboardDetection(
        label_id=label_id,
        label_name=label_name,
        score=score,
        box=_parse_detection_box(
            _require_key(
                data,
                "box",
                status_code=status_code,
            ),
            image_width=image_width,
            image_height=image_height,
            status_code=status_code,
        ),
    )


def parse_detection_prediction_payload(
    payload: object,
    *,
    status_code: int | None = None,
) -> DashboardDetectionPrediction:
    """Detection JSON의 필수 Key·범위·상호 관계를 검증한다."""

    data = _require_mapping(
        payload,
        status_code=status_code,
    )

    image_width = _require_integer(
        data,
        "image_width",
        status_code=status_code,
    )
    image_height = _require_integer(
        data,
        "image_height",
        status_code=status_code,
    )
    detection_count = _require_integer(
        data,
        "detection_count",
        status_code=status_code,
    )
    score_threshold = _require_float(
        data,
        "score_threshold",
        status_code=status_code,
    )
    iou_threshold = _require_float(
        data,
        "iou_threshold",
        status_code=status_code,
    )
    checkpoint_epoch = _require_integer(
        data,
        "checkpoint_epoch",
        status_code=status_code,
    )
    checkpoint_metric_value = _require_float(
        data,
        "checkpoint_metric_value",
        status_code=status_code,
    )
    inference_time_ms = _require_float(
        data,
        "inference_time_ms",
        status_code=status_code,
    )

    if image_width <= 0 or image_height <= 0:
        raise _invalid_response(status_code=status_code)
    if detection_count < 0:
        raise _invalid_response(status_code=status_code)
    if not 0.05 <= score_threshold <= 0.95:
        raise _invalid_response(status_code=status_code)
    if not 0.0 <= iou_threshold <= 1.0:
        raise _invalid_response(status_code=status_code)
    if checkpoint_epoch <= 0:
        raise _invalid_response(status_code=status_code)
    if not 0.0 <= checkpoint_metric_value <= 1.0:
        raise _invalid_response(status_code=status_code)
    if inference_time_ms < 0.0:
        raise _invalid_response(status_code=status_code)

    raw_detections = _require_key(
        data,
        "detections",
        status_code=status_code,
    )
    if not isinstance(raw_detections, list):
        raise _invalid_response(status_code=status_code)

    detections = tuple(
        _parse_detection(
            item,
            image_width=image_width,
            image_height=image_height,
            score_threshold=score_threshold,
            status_code=status_code,
        )
        for item in raw_detections
    )
    if len(detections) != detection_count:
        raise _invalid_response(status_code=status_code)

    scores = [detection.score for detection in detections]
    if scores != sorted(scores, reverse=True):
        raise _invalid_response(status_code=status_code)

    model_input_mode = _require_string(
        data,
        "model_input_mode",
        status_code=status_code,
    )
    if model_input_mode != "RGB":
        raise _invalid_response(status_code=status_code)

    return DashboardDetectionPrediction(
        detections=detections,
        detection_count=detection_count,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        model_name=_require_string(
            data,
            "model_name",
            status_code=status_code,
        ),
        model_version=_require_string(
            data,
            "model_version",
            status_code=status_code,
        ),
        architecture=_require_string(
            data,
            "architecture",
            status_code=status_code,
        ),
        device=_require_string(
            data,
            "device",
            status_code=status_code,
        ),
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_metric_name=_require_string(
            data,
            "checkpoint_metric_name",
            status_code=status_code,
        ),
        checkpoint_metric_value=checkpoint_metric_value,
        original_filename=_require_string(
            data,
            "original_filename",
            status_code=status_code,
        ),
        content_type=_require_string(
            data,
            "content_type",
            status_code=status_code,
        ),
        image_width=image_width,
        image_height=image_height,
        image_mode=_require_string(
            data,
            "image_mode",
            status_code=status_code,
        ),
        model_input_mode=model_input_mode,
        inference_time_ms=inference_time_ms,
    )


def _parse_json_response(
    response: httpx.Response,
) -> object:
    try:
        return response.json()
    except ValueError as error:
        raise _invalid_response(
            status_code=response.status_code,
        ) from error


def _raise_api_error(
    response: httpx.Response,
) -> None:
    code = "API_REQUEST_ERROR"

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            candidate = detail.get("code")
            if (
                isinstance(candidate, str)
                and candidate.strip()
            ):
                code = candidate.strip()

    if code not in SAFE_DETECTION_ERROR_MESSAGES:
        if response.status_code == 503:
            code = "DETECTION_MODEL_NOT_READY"
        elif response.status_code == 408:
            code = "API_TIMEOUT"
        else:
            code = "API_REQUEST_ERROR"

    raise DetectionDashboardApiError(
        code=code,
        status_code=response.status_code,
    )


class DetectionDashboardApiClient:
    """Streamlit Detection 페이지용 동기식 FastAPI Client."""

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

    def __enter__(self) -> "DetectionDashboardApiClient":
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:  # type: ignore[no-untyped-def]
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        self._client.close()

    def detect_image(
        self,
        *,
        filename: str,
        content_type: str,
        image_bytes: bytes,
        score_threshold: float = 0.5,
    ) -> DashboardDetectionPrediction:
        """이미지 Byte와 Threshold를 Detection Endpoint로 전달한다."""

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must not be empty")
        if not isinstance(content_type, str) or not content_type.strip():
            raise ValueError("content_type must not be empty")
        if not isinstance(image_bytes, bytes):
            raise TypeError("image_bytes must be bytes")
        if not image_bytes:
            raise DetectionDashboardApiError(
                code="EMPTY_FILE",
            )

        threshold = validate_dashboard_score_threshold(
            score_threshold
        )
        files = {
            "file": (
                filename,
                image_bytes,
                content_type,
            )
        }

        try:
            response = self._client.post(
                DETECTION_PREDICTION_ENDPOINT,
                params={
                    "score_threshold": (
                        f"{threshold:.6g}"
                    )
                },
                files=files,
            )
        except httpx.TimeoutException as error:
            raise DetectionDashboardApiError(
                code="API_TIMEOUT",
            ) from error
        except httpx.ConnectError as error:
            raise DetectionDashboardApiError(
                code="API_CONNECTION_ERROR",
            ) from error
        except httpx.RequestError as error:
            raise DetectionDashboardApiError(
                code="API_REQUEST_ERROR",
            ) from error

        if not response.is_success:
            _raise_api_error(response)

        return parse_detection_prediction_payload(
            _parse_json_response(response),
            status_code=response.status_code,
        )
