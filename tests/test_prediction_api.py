from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import create_app
from src.api.image_validation import ValidatedImage
from src.api.schemas import PredictionResponse


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    return buffer.getvalue()


class _DummyInferenceService:
    def __init__(
        self,
        *,
        prediction: int = 1,
        fail_with_message: str | None = None,
    ) -> None:
        self.is_ready = True
        self.model_name = "DummyModel"
        self.model_version = "dummy-v1"
        self.device_name = "cpu"
        self.prediction = prediction
        self.fail_with_message = fail_with_message

    def predict(self, image: ValidatedImage) -> PredictionResponse:
        if self.fail_with_message is not None:
            raise RuntimeError(self.fail_with_message)

        defect_probability = 0.9 if self.prediction == 1 else 0.1
        raw_logit = 2.1972246 if self.prediction == 1 else -2.1972246
        class_name = "DEFECT" if self.prediction == 1 else "NORMAL"

        return PredictionResponse(
            prediction=self.prediction,
            prediction_class_name=class_name,
            defect_probability=defect_probability,
            normal_probability=1.0 - defect_probability,
            raw_logit=raw_logit,
            classification_threshold=0.5,
            model_name=self.model_name,
            model_version=self.model_version,
            positive_class="DEFECT",
            original_filename=image.original_filename,
            content_type=image.content_type,
            image_width=image.original_width,
            image_height=image.original_height,
            image_mode=image.original_mode,
            inference_time_ms=0.1,
        )


def test_health_returns_loaded_model_metadata() -> None:
    application = create_app(
        service_factory=lambda: _DummyInferenceService(),
    )

    with TestClient(application) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Manufacturing Vision Defect Analysis System",
        "model_loaded": True,
        "model_name": "DummyModel",
        "device": "cpu",
    }


@pytest.mark.parametrize(
    ("prediction", "expected_class_name"),
    [(1, "DEFECT"), (0, "NORMAL")],
)
def test_prediction_endpoint_returns_expected_class(
    prediction: int,
    expected_class_name: str,
) -> None:
    application = create_app(
        service_factory=lambda: _DummyInferenceService(prediction=prediction),
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/predictions",
            files={"file": ("sample.png", _make_png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == prediction
    assert payload["prediction_class_name"] == expected_class_name
    assert payload["positive_class"] == "DEFECT"
    assert payload["classification_threshold"] == 0.5
    assert payload["image_width"] == 32
    assert payload["image_height"] == 24


def test_missing_file_returns_project_error_schema() -> None:
    application = create_app(
        service_factory=lambda: _DummyInferenceService(),
    )

    with TestClient(application) as client:
        response = client.post("/api/v1/predictions")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "MISSING_FILE"


def test_empty_file_returns_empty_file_error() -> None:
    application = create_app(
        service_factory=lambda: _DummyInferenceService(),
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/predictions",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "EMPTY_FILE"


def test_corrupted_image_returns_invalid_image_error() -> None:
    application = create_app(
        service_factory=lambda: _DummyInferenceService(),
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/predictions",
            files={"file": ("broken.jpg", b"broken", "image/jpeg")},
        )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_IMAGE"


def test_unsupported_content_type_returns_415() -> None:
    application = create_app(
        service_factory=lambda: _DummyInferenceService(),
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/predictions",
            files={"file": ("sample.txt", b"text", "text/plain")},
        )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_model_not_ready_returns_503() -> None:
    application = create_app(service_factory=None)

    with TestClient(application) as client:
        health_response = client.get("/api/v1/health")
        prediction_response = client.post(
            "/api/v1/predictions",
            files={"file": ("sample.png", _make_png_bytes(), "image/png")},
        )

    assert health_response.status_code == 200
    assert health_response.json()["model_loaded"] is False
    assert prediction_response.status_code == 503
    assert prediction_response.json()["detail"]["code"] == "MODEL_NOT_READY"


def test_startup_failure_is_converted_to_model_not_ready() -> None:
    def failing_factory():
        raise RuntimeError(r"C:\secret\checkpoint.pt")

    application = create_app(service_factory=failing_factory)

    with TestClient(application) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    assert "secret" not in response.text
    assert "checkpoint.pt" not in response.text


def test_internal_failure_does_not_expose_path_or_stack_trace() -> None:
    secret_message = (
        r"C:\Users\kflow\Downloads\manufacturing-vision-defect-analysis-system"
        r"\models\checkpoints\resnet18_transfer_best.pt"
    )
    application = create_app(
        service_factory=lambda: _DummyInferenceService(
            fail_with_message=secret_message,
        ),
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/predictions",
            files={"file": ("sample.png", _make_png_bytes(), "image/png")},
        )

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "INFERENCE_FAILED"
    assert "kflow" not in response.text
    assert "resnet18_transfer_best.pt" not in response.text
    assert "Traceback" not in response.text
