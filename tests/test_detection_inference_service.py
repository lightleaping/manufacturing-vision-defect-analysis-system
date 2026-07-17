from __future__ import annotations

from copy import deepcopy

from PIL import Image
import pytest
import torch
from torch import nn

from src.api.detection_config import DetectionApiSettings
from src.api.detection_inference_service import (
    DetectionInferenceService,
    DetectionInferenceServiceError,
)
from src.api.image_validation import ValidatedImage


CLASS_MAPPING = {
    "BACKGROUND": 0,
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
    "rolled_in_scale": 5,
    "scratches": 6,
}


class DummyDetectionModel(nn.Module):
    def __init__(self, prediction: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.prediction = prediction
        self.forward_calls = 0
        self.inference_mode_seen = False

    def forward(self, images):
        assert len(images) == 1
        self.forward_calls += 1
        self.inference_mode_seen = torch.is_inference_mode_enabled()
        return [
            {
                key: value.clone()
                for key, value in self.prediction.items()
            }
        ]


def make_validated_image() -> ValidatedImage:
    image = Image.new("RGB", (20, 10), color=(10, 20, 30))
    return ValidatedImage(
        original_filename="sample.png",
        content_type="image/png",
        original_width=20,
        original_height=10,
        original_mode="RGB",
        decoded_format="PNG",
        rgb_image=image,
    )


def make_service(
    prediction: dict[str, torch.Tensor],
) -> tuple[DetectionInferenceService, DummyDetectionModel]:
    model = DummyDetectionModel(prediction)
    service = DetectionInferenceService(
        model=model,
        checkpoint_epoch_index=2,
        checkpoint_metric_value=0.677418,
        class_mapping=CLASS_MAPPING,
        settings=DetectionApiSettings(),
    )
    return service, model


def test_prediction_filters_sorts_and_returns_original_coordinates() -> None:
    prediction = {
        "boxes": torch.tensor(
            [
                [-2.0, 1.0, 25.0, 9.0],
                [2.0, 2.0, 8.0, 8.0],
                [1.0, 1.0, 4.0, 4.0],
            ],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([3, 2, 1], dtype=torch.int64),
        "scores": torch.tensor([0.80, 0.95, 0.40], dtype=torch.float32),
    }
    service, model = make_service(prediction)
    validated = make_validated_image()
    original_bytes = validated.rgb_image.tobytes()

    response = service.predict(validated, score_threshold="0.5")

    assert response.detection_count == 2
    assert [item.label_name for item in response.detections] == [
        "inclusion",
        "patches",
    ]
    assert response.detections[1].box.xmin == pytest.approx(0.0)
    assert response.detections[1].box.xmax == pytest.approx(20.0)
    assert response.checkpoint_epoch == 3
    assert response.checkpoint_metric_value == pytest.approx(0.677418)
    assert response.image_width == 20
    assert response.image_height == 10
    assert response.model_input_mode == "RGB"
    assert response.inference_time_ms >= 0.0
    assert model.forward_calls == 1
    assert model.inference_mode_seen is True
    assert model.training is False
    assert validated.rgb_image.tobytes() == original_bytes


def test_empty_detection_is_returned_as_valid_response() -> None:
    prediction = {
        "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "scores": torch.tensor([0.49], dtype=torch.float32),
    }
    service, _ = make_service(prediction)

    response = service.predict(make_validated_image())

    assert response.detections == []
    assert response.detection_count == 0
    assert response.score_threshold == pytest.approx(0.5)


@pytest.mark.parametrize("value", ["nan", "inf", "0.01", "1.0", "abc"])
def test_invalid_threshold_is_project_error(value: str) -> None:
    prediction = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
        "scores": torch.empty((0,), dtype=torch.float32),
    }
    service, _ = make_service(prediction)

    with pytest.raises(DetectionInferenceServiceError) as exc_info:
        service.predict(make_validated_image(), score_threshold=value)

    assert exc_info.value.code == "INVALID_SCORE_THRESHOLD"
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "prediction",
    [
        {
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "scores": torch.tensor([float("nan")], dtype=torch.float32),
        },
        {
            "boxes": torch.tensor(
                [[1.0, 1.0, float("inf"), 5.0]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1], dtype=torch.int64),
            "scores": torch.tensor([0.9], dtype=torch.float32),
        },
        {
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
            "scores": torch.tensor([0.9], dtype=torch.float32),
        },
        {
            "boxes": torch.tensor([[4.0, 4.0, 1.0, 1.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "scores": torch.tensor([0.9], dtype=torch.float32),
        },
    ],
)
def test_invalid_model_output_is_rejected(prediction) -> None:
    service, _ = make_service(prediction)

    with pytest.raises(DetectionInferenceServiceError) as exc_info:
        service.predict(make_validated_image())

    assert exc_info.value.code == "INVALID_DETECTION_MODEL_OUTPUT"
    assert exc_info.value.status_code == 500


def test_prediction_tensor_counts_must_match() -> None:
    prediction = {
        "boxes": torch.empty((2, 4), dtype=torch.float32),
        "labels": torch.empty((1,), dtype=torch.int64),
        "scores": torch.empty((2,), dtype=torch.float32),
    }
    service, _ = make_service(prediction)

    with pytest.raises(DetectionInferenceServiceError, match="counts"):
        service.predict(make_validated_image())
