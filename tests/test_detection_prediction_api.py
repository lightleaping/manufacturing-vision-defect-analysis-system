from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from src.api.app import create_app
from src.api.detection_config import DetectionApiSettings, resolve_score_threshold
from src.api.detection_inference_service import DetectionInferenceServiceError
from src.api.schemas import (
    DetectionBox,
    DetectionItem,
    DetectionPredictionResponse,
)


class ReadyClassificationService:
    is_ready = True
    model_name = "DummyClassification"
    device_name = "cpu"

    def predict(self, validated_image):
        raise AssertionError("Classification endpoint must not be called.")


class DummyDetectionService:
    is_ready = True
    model_name = "DummyDetection"
    device_name = "cpu"

    def __init__(self, *, empty: bool = False, fail: bool = False) -> None:
        self.empty = empty
        self.fail = fail
        self.received_threshold = None

    def predict(self, validated_image, *, score_threshold=None):
        self.received_threshold = score_threshold
        if self.fail:
            raise DetectionInferenceServiceError(
                code="INVALID_DETECTION_MODEL_OUTPUT",
                message="safe failure",
                status_code=500,
            )

        detections = [] if self.empty else [
            DetectionItem(
                label_id=3,
                label_name="patches",
                score=0.91,
                box=DetectionBox(
                    xmin=1.0,
                    ymin=2.0,
                    xmax=9.0,
                    ymax=8.0,
                ),
            )
        ]
        try:
            threshold = resolve_score_threshold(
                score_threshold,
                settings=DetectionApiSettings(),
            )
        except ValueError as error:
            raise DetectionInferenceServiceError(
                code="INVALID_SCORE_THRESHOLD",
                message=str(error),
                status_code=400,
            ) from error

        return DetectionPredictionResponse(
            detections=detections,
            detection_count=len(detections),
            score_threshold=threshold,
            iou_threshold=0.5,
            model_name="FasterRCNNMobileNetV3Large320FPN",
            model_version="day12_detection_best",
            architecture="fasterrcnn_mobilenet_v3_large_320_fpn",
            device="cpu",
            checkpoint_epoch=3,
            checkpoint_metric_name="map_50",
            checkpoint_metric_value=0.677418,
            original_filename=validated_image.original_filename,
            content_type=validated_image.content_type,
            image_width=validated_image.original_width,
            image_height=validated_image.original_height,
            image_mode=validated_image.original_mode,
            model_input_mode="RGB",
            inference_time_ms=10.0,
        )


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (12, 10), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def build_client(service: DummyDetectionService | None) -> TestClient:
    application = create_app(
        service_factory=lambda: ReadyClassificationService(),
        detection_service_factory=(None if service is None else lambda: service),
    )
    return TestClient(application)


def test_detection_endpoint_returns_schema_and_threshold() -> None:
    service = DummyDetectionService()

    with build_client(service) as client:
        response = client.post(
            "/api/v1/detection/predictions?score_threshold=0.65",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["detection_count"] == 1
    assert payload["detections"][0]["label_name"] == "patches"
    assert payload["score_threshold"] == pytest.approx(0.65)
    assert payload["checkpoint_epoch"] == 3
    assert payload["image_width"] == 12
    assert payload["image_height"] == 10
    assert service.received_threshold == "0.65"


def test_detection_endpoint_accepts_empty_detection() -> None:
    with build_client(DummyDetectionService(empty=True)) as client:
        response = client.post(
            "/api/v1/detection/predictions",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["detections"] == []
    assert response.json()["detection_count"] == 0


def test_detection_model_not_ready_returns_503() -> None:
    with build_client(None) as client:
        response = client.post(
            "/api/v1/detection/predictions",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "DETECTION_MODEL_NOT_READY"


def test_detection_startup_failure_is_not_exposed() -> None:
    def fail_factory():
        raise RuntimeError(r"C:\secret\checkpoint.pt")

    application = create_app(
        service_factory=lambda: ReadyClassificationService(),
        detection_service_factory=fail_factory,
    )
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/detection/predictions",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 503
    assert "secret" not in response.text
    assert "checkpoint" not in response.text.lower()


def test_corrupted_detection_image_returns_project_error() -> None:
    with build_client(DummyDetectionService()) as client:
        response = client.post(
            "/api/v1/detection/predictions",
            files={"file": ("sample.png", b"not-an-image", "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_IMAGE"


def test_missing_detection_file_uses_existing_error_schema() -> None:
    with build_client(DummyDetectionService()) as client:
        response = client.post("/api/v1/detection/predictions")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "MISSING_FILE"


def test_detection_service_error_is_safe() -> None:
    with build_client(DummyDetectionService(fail=True)) as client:
        response = client.post(
            "/api/v1/detection/predictions",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "INVALID_DETECTION_MODEL_OUTPUT",
        "message": "safe failure",
    }


def test_existing_classification_health_contract_is_unchanged() -> None:
    with build_client(DummyDetectionService()) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Manufacturing Vision Defect Analysis System",
        "model_loaded": True,
        "model_name": "DummyClassification",
        "device": "cpu",
    }

def test_invalid_score_threshold_returns_400() -> None:
    with build_client(DummyDetectionService()) as client:
        response = client.post(
            "/api/v1/detection/predictions?score_threshold=nan",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_SCORE_THRESHOLD"


def test_detection_service_factory_runs_once_for_multiple_requests() -> None:
    service = DummyDetectionService()
    factory_calls = 0

    def detection_factory():
        nonlocal factory_calls
        factory_calls += 1
        return service

    application = create_app(
        service_factory=lambda: ReadyClassificationService(),
        detection_service_factory=detection_factory,
    )
    with TestClient(application) as client:
        first = client.post(
            "/api/v1/detection/predictions",
            files={"file": ("first.png", png_bytes(), "image/png")},
        )
        second = client.post(
            "/api/v1/detection/predictions",
            files={"file": ("second.png", png_bytes(), "image/png")},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert factory_calls == 1

