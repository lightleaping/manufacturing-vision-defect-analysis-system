from __future__ import annotations

import httpx
import pytest

from src.dashboard.config import DashboardSettings
from src.dashboard.detection_api_client import (
    DETECTION_PREDICTION_ENDPOINT,
    DetectionDashboardApiClient,
    DetectionDashboardApiError,
    parse_detection_prediction_payload,
    validate_dashboard_score_threshold,
)


def valid_payload(*, detections=None):
    if detections is None:
        detections = [
            {
                "label_id": 3,
                "label_name": "patches",
                "score": 0.91,
                "box": {
                    "xmin": 1.0,
                    "ymin": 2.0,
                    "xmax": 9.0,
                    "ymax": 8.0,
                },
            }
        ]
    return {
        "detections": detections,
        "detection_count": len(detections),
        "score_threshold": 0.5,
        "iou_threshold": 0.5,
        "model_name": "FasterRCNNMobileNetV3Large320FPN",
        "model_version": "day12_detection_best",
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "device": "cpu",
        "checkpoint_epoch": 3,
        "checkpoint_metric_name": "map_50",
        "checkpoint_metric_value": 0.677418,
        "original_filename": "sample.png",
        "content_type": "image/png",
        "image_width": 12,
        "image_height": 10,
        "image_mode": "RGB",
        "model_input_mode": "RGB",
        "inference_time_ms": 10.0,
    }


def settings() -> DashboardSettings:
    return DashboardSettings(
        api_base_url="http://testserver",
    )


def test_parse_detection_payload() -> None:
    result = parse_detection_prediction_payload(
        valid_payload()
    )

    assert result.detection_count == 1
    assert result.detections[0].label_name == "patches"
    assert result.detections[0].box.xmax == 9.0
    assert result.checkpoint_epoch == 3


def test_parse_empty_detection_payload() -> None:
    result = parse_detection_prediction_payload(
        valid_payload(detections=[])
    )

    assert result.detection_count == 0
    assert result.detections == ()


@pytest.mark.parametrize(
    "threshold",
    [0.05, 0.5, 0.95],
)
def test_threshold_boundaries(threshold: float) -> None:
    assert validate_dashboard_score_threshold(
        threshold
    ) == threshold


@pytest.mark.parametrize(
    "threshold",
    [0.0, 0.049, 0.951, 1.0, float("nan")],
)
def test_invalid_threshold_rejected(
    threshold: float,
) -> None:
    with pytest.raises(ValueError):
        validate_dashboard_score_threshold(threshold)


def test_detection_count_must_match_list() -> None:
    payload = valid_payload()
    payload["detection_count"] = 2

    with pytest.raises(
        DetectionDashboardApiError
    ):
        parse_detection_prediction_payload(payload)


def test_label_name_must_match_label_id() -> None:
    payload = valid_payload()
    payload["detections"][0]["label_name"] = "scratches"

    with pytest.raises(
        DetectionDashboardApiError
    ):
        parse_detection_prediction_payload(payload)


def test_box_must_stay_in_original_image() -> None:
    payload = valid_payload()
    payload["detections"][0]["box"]["xmax"] = 13.0

    with pytest.raises(
        DetectionDashboardApiError
    ):
        parse_detection_prediction_payload(payload)


def test_detections_must_be_score_sorted() -> None:
    payload = valid_payload(
        detections=[
            {
                "label_id": 3,
                "label_name": "patches",
                "score": 0.6,
                "box": {
                    "xmin": 1.0,
                    "ymin": 1.0,
                    "xmax": 5.0,
                    "ymax": 5.0,
                },
            },
            {
                "label_id": 6,
                "label_name": "scratches",
                "score": 0.9,
                "box": {
                    "xmin": 6.0,
                    "ymin": 2.0,
                    "xmax": 11.0,
                    "ymax": 8.0,
                },
            },
        ]
    )

    with pytest.raises(
        DetectionDashboardApiError
    ):
        parse_detection_prediction_payload(payload)


def test_client_sends_threshold_and_multipart() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["threshold"] = request.url.params[
            "score_threshold"
        ]
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json=valid_payload(),
        )

    with DetectionDashboardApiClient(
        settings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.detect_image(
            filename="sample.png",
            content_type="image/png",
            image_bytes=b"image-bytes",
            score_threshold=0.5,
        )

    assert result.detection_count == 1
    assert captured["path"] == DETECTION_PREDICTION_ENDPOINT
    assert captured["threshold"] == "0.5"
    assert b'name="file"' in captured["body"]
    assert b"sample.png" in captured["body"]


def test_client_handles_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "timeout",
            request=request,
        )

    with DetectionDashboardApiClient(
        settings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(
            DetectionDashboardApiError
        ) as exc_info:
            client.detect_image(
                filename="sample.png",
                content_type="image/png",
                image_bytes=b"x",
            )

    assert exc_info.value.code == "API_TIMEOUT"


def test_client_handles_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "connection",
            request=request,
        )

    with DetectionDashboardApiClient(
        settings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(
            DetectionDashboardApiError
        ) as exc_info:
            client.detect_image(
                filename="sample.png",
                content_type="image/png",
                image_bytes=b"x",
            )

    assert exc_info.value.code == "API_CONNECTION_ERROR"


def test_client_handles_detection_503() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "detail": {
                    "code": "DETECTION_MODEL_NOT_READY",
                    "message": "internal message",
                }
            },
        )

    with DetectionDashboardApiClient(
        settings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(
            DetectionDashboardApiError
        ) as exc_info:
            client.detect_image(
                filename="sample.png",
                content_type="image/png",
                image_bytes=b"x",
            )

    assert exc_info.value.code == "DETECTION_MODEL_NOT_READY"
    assert "internal" not in exc_info.value.message


def test_client_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="not-json",
        )

    with DetectionDashboardApiClient(
        settings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(
            DetectionDashboardApiError
        ) as exc_info:
            client.detect_image(
                filename="sample.png",
                content_type="image/png",
                image_bytes=b"x",
            )

    assert exc_info.value.code == "API_INVALID_RESPONSE"
