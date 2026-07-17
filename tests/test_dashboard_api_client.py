from __future__ import annotations

import json

import httpx
import pytest

from src.dashboard.api_client import (
    DashboardApiClient,
    DashboardApiError,
    parse_prediction_payload,
)
from src.dashboard.config import DashboardSettings


def _health_payload(*, model_loaded: bool = True) -> dict[str, object]:
    return {
        "status": "ok",
        "service": "Manufacturing Vision Defect Analysis System",
        "model_loaded": model_loaded,
        "model_name": "ResNet18Transfer",
        "device": "cpu",
    }


def _prediction_payload(
    *,
    prediction: int = 1,
    class_name: str = "DEFECT",
    defect_probability: float = 0.9,
) -> dict[str, object]:
    return {
        "prediction": prediction,
        "prediction_class_name": class_name,
        "defect_probability": defect_probability,
        "normal_probability": 1.0 - defect_probability,
        "raw_logit": 2.1972246 if prediction == 1 else -2.1972246,
        "classification_threshold": 0.5,
        "model_name": "ResNet18Transfer",
        "model_version": "resnet18_transfer_best",
        "positive_class": "DEFECT",
        "original_filename": "sample.png",
        "content_type": "image/png",
        "image_width": 300,
        "image_height": 300,
        "image_mode": "RGB",
        "inference_time_ms": 12.5,
    }


def _json_response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def test_health_200_is_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/health"
        return _json_response(200, _health_payload())

    with DashboardApiClient(
        DashboardSettings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        health = client.get_health()

    assert health.model_loaded is True
    assert health.model_name == "ResNet18Transfer"
    assert health.device == "cpu"


@pytest.mark.parametrize(
    ("prediction", "class_name", "defect_probability"),
    [
        (0, "NORMAL", 0.1),
        (1, "DEFECT", 0.9),
    ],
)
def test_prediction_200_is_parsed(
    prediction: int,
    class_name: str,
    defect_probability: float,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/predictions"
        assert request.headers["content-type"].startswith("multipart/form-data")
        return _json_response(
            200,
            _prediction_payload(
                prediction=prediction,
                class_name=class_name,
                defect_probability=defect_probability,
            ),
        )

    with DashboardApiClient(
        DashboardSettings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.predict_image(
            filename="sample.png",
            content_type="image/png",
            image_bytes=b"image-bytes",
        )

    assert result.prediction == prediction
    assert result.prediction_class_name == class_name
    assert result.defect_probability == defect_probability


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [
        (503, "MODEL_NOT_READY"),
        (400, "INVALID_IMAGE"),
        (415, "UNSUPPORTED_FILE_TYPE"),
        (413, "FILE_TOO_LARGE"),
        (413, "IMAGE_TOO_LARGE"),
        (500, "INFERENCE_FAILED"),
    ],
)
def test_fastapi_error_schema_is_mapped(
    status_code: int,
    error_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            status_code,
            {
                "detail": {
                    "code": error_code,
                    "message": r"C:\secret\checkpoint.pt",
                }
            },
        )

    with DashboardApiClient(
        DashboardSettings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(DashboardApiError) as exc_info:
            client.predict_image(
                filename="sample.png",
                content_type="image/png",
                image_bytes=b"image-bytes",
            )

    assert exc_info.value.code == error_code
    assert "secret" not in exc_info.value.message
    assert "checkpoint.pt" not in exc_info.value.message


def test_timeout_is_mapped_without_internal_exception_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret timeout detail", request=request)

    with DashboardApiClient(
        DashboardSettings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(DashboardApiError) as exc_info:
            client.get_health()

    assert exc_info.value.code == "API_TIMEOUT"
    assert "secret" not in exc_info.value.message


def test_connection_error_is_mapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with DashboardApiClient(
        DashboardSettings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(DashboardApiError) as exc_info:
            client.get_health()

    assert exc_info.value.code == "API_CONNECTION_ERROR"


def test_invalid_json_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b"not-json")

    with DashboardApiClient(
        DashboardSettings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(DashboardApiError) as exc_info:
            client.get_health()

    assert exc_info.value.code == "API_INVALID_RESPONSE"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("raw_logit"),
        lambda payload: payload.__setitem__("normal_probability", 0.4),
        lambda payload: payload.__setitem__("prediction_class_name", "NORMAL"),
        lambda payload: payload.__setitem__("positive_class", "NORMAL"),
        lambda payload: payload.__setitem__("defect_probability", float("nan")),
    ],
)
def test_prediction_schema_inconsistency_is_rejected(mutation) -> None:  # type: ignore[no-untyped-def]
    payload = _prediction_payload()
    mutation(payload)

    with pytest.raises(DashboardApiError) as exc_info:
        parse_prediction_payload(payload)

    assert exc_info.value.code == "API_INVALID_RESPONSE"


def test_empty_image_is_rejected_before_http_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP request must not be sent")

    with DashboardApiClient(
        DashboardSettings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(DashboardApiError) as exc_info:
            client.predict_image(
                filename="sample.png",
                content_type="image/png",
                image_bytes=b"",
            )

    assert exc_info.value.code == "EMPTY_FILE"
