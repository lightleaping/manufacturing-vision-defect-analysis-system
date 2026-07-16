from __future__ import annotations

import math

import pytest
import torch
from PIL import Image
from torch import nn

from src.api.config import ApiSettings
from src.api.image_validation import ValidatedImage
from src.api.inference_service import ImageInferenceService, InferenceServiceError


class _RecordingModel(nn.Module):
    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output
        self.inference_mode_enabled = False
        self.seen_shape: tuple[int, ...] | None = None

    def forward(self, inputs: torch.Tensor) -> object:
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        self.seen_shape = tuple(inputs.shape)
        if isinstance(self.output, torch.Tensor):
            return self.output.to(inputs.device)
        return self.output


def _transform(_: Image.Image) -> torch.Tensor:
    return torch.ones((3, 224, 224), dtype=torch.float32)


def _validated_image() -> ValidatedImage:
    return ValidatedImage(
        original_filename="sample.png",
        content_type="image/png",
        original_width=300,
        original_height=300,
        original_mode="RGB",
        decoded_format="PNG",
        rgb_image=Image.new("RGB", (300, 300), color=(20, 30, 40)),
    )


@pytest.mark.parametrize(
    ("logit", "prediction", "class_name"),
    [
        (4.0, 1, "DEFECT"),
        (-4.0, 0, "NORMAL"),
        (0.0, 1, "DEFECT"),
    ],
)
def test_logit_to_prediction_policy(
    logit: float,
    prediction: int,
    class_name: str,
) -> None:
    model = _RecordingModel(torch.tensor([logit]))
    service = ImageInferenceService(model=model, transform=_transform)
    image = _validated_image()

    result = service.predict(image)

    assert result.prediction == prediction
    assert result.prediction_class_name == class_name
    assert result.normal_probability + result.defect_probability == pytest.approx(1.0)
    assert model.training is False
    assert model.inference_mode_enabled is True
    assert model.seen_shape == (1, 3, 224, 224)
    image.rgb_image.close()


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_model_output_is_rejected(invalid_value: float) -> None:
    service = ImageInferenceService(
        model=_RecordingModel(torch.tensor([invalid_value])),
        transform=_transform,
    )
    image = _validated_image()

    with pytest.raises(InferenceServiceError) as exc_info:
        service.predict(image)

    assert exc_info.value.code == "INVALID_MODEL_OUTPUT"
    image.rgb_image.close()


@pytest.mark.parametrize(
    "invalid_output",
    [torch.tensor(1.0), torch.tensor([[[1.0]]]), torch.tensor([1.0, 2.0])],
)
def test_invalid_output_shape_is_rejected(invalid_output: torch.Tensor) -> None:
    service = ImageInferenceService(
        model=_RecordingModel(invalid_output),
        transform=_transform,
    )
    image = _validated_image()

    with pytest.raises(InferenceServiceError) as exc_info:
        service.predict(image)

    assert exc_info.value.code == "INVALID_MODEL_OUTPUT"
    image.rgb_image.close()


def test_non_tensor_model_output_is_rejected() -> None:
    service = ImageInferenceService(
        model=_RecordingModel([1.0]),
        transform=_transform,
    )
    image = _validated_image()

    with pytest.raises(InferenceServiceError) as exc_info:
        service.predict(image)

    assert exc_info.value.code == "INVALID_MODEL_OUTPUT"
    image.rgb_image.close()


def test_invalid_transform_shape_is_rejected() -> None:
    def invalid_transform(_: Image.Image) -> torch.Tensor:
        return torch.ones((2, 3, 224, 224))

    service = ImageInferenceService(
        model=_RecordingModel(torch.tensor([1.0])),
        transform=invalid_transform,
    )
    image = _validated_image()

    with pytest.raises(InferenceServiceError) as exc_info:
        service.predict(image)

    assert exc_info.value.code == "INVALID_MODEL_INPUT"
    image.rgb_image.close()


def test_non_finite_transform_is_rejected() -> None:
    def invalid_transform(_: Image.Image) -> torch.Tensor:
        return torch.full((3, 224, 224), math.nan)

    service = ImageInferenceService(
        model=_RecordingModel(torch.tensor([1.0])),
        transform=invalid_transform,
    )
    image = _validated_image()

    with pytest.raises(InferenceServiceError) as exc_info:
        service.predict(image)

    assert exc_info.value.code == "INVALID_MODEL_INPUT"
    image.rgb_image.close()


def test_threshold_configuration_must_be_finite() -> None:
    with pytest.raises(ValueError):
        ImageInferenceService(
            model=_RecordingModel(torch.tensor([1.0])),
            transform=_transform,
            settings=ApiSettings(classification_threshold=math.nan),
        )
