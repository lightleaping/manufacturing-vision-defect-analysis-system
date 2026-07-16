from __future__ import annotations

import pytest
import torch
from torch import nn

from src.explainability.gradcam import (
    GradCAM,
    InvalidGradCAMInputError,
    TargetLayerNotFoundError,
    ZeroGradCAMError,
    resolve_target_layer,
)


class DummyBinaryCNN(nn.Module):
    def __init__(self, *, classifier_weight: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 1, bias=False)

        with torch.no_grad():
            self.features[0].weight.fill_(0.10)
            self.classifier.weight.fill_(classifier_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(start_dim=1)
        return self.classifier(pooled).squeeze(dim=1)


class ZeroCAMCNN(DummyBinaryCNN):
    def __init__(self) -> None:
        super().__init__(classifier_weight=0.0)


def _positive_input() -> torch.Tensor:
    return torch.full((1, 3, 16, 16), 0.5, dtype=torch.float32)


def test_resolve_target_layer_supports_numeric_tokens() -> None:
    model = DummyBinaryCNN(classifier_weight=1.0)

    target_layer = resolve_target_layer(model, "features.0")

    assert target_layer is model.features[0]


def test_resolve_target_layer_raises_for_missing_layer() -> None:
    model = DummyBinaryCNN(classifier_weight=1.0)

    with pytest.raises(TargetLayerNotFoundError, match="Target Layer"):
        resolve_target_layer(model, "features.99")


def test_gradcam_defect_target_uses_raw_logit_and_returns_normalized_cam() -> None:
    model = DummyBinaryCNN(classifier_weight=1.0)
    model.eval()

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        result = gradcam.generate(
            input_tensor=_positive_input(),
            target_class="DEFECT",
        )

        assert result.target_class == "DEFECT"
        assert result.target_score_mode == "raw_logit"
        assert result.target_score_value == pytest.approx(result.raw_logit)
        assert result.cam.shape == (16, 16)
        assert float(result.cam.min()) >= 0.0
        assert float(result.cam.max()) <= 1.0
        assert float(result.cam.max()) == pytest.approx(1.0, abs=1e-5)
        assert torch.isfinite(result.cam).all()
        assert result.activation_shape == result.gradient_shape


def test_gradcam_normal_target_uses_negative_raw_logit() -> None:
    model = DummyBinaryCNN(classifier_weight=-1.0)
    model.eval()

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        result = gradcam.generate(
            input_tensor=_positive_input(),
            target_class="NORMAL",
        )

        assert result.prediction == 0
        assert result.target_class == "NORMAL"
        assert result.target_score_mode == "negative_raw_logit"
        assert result.target_score_value == pytest.approx(-result.raw_logit)
        assert torch.isfinite(result.cam).all()
        assert 0.0 <= float(result.cam.min()) <= float(result.cam.max()) <= 1.0


def test_gradcam_defaults_to_predicted_class() -> None:
    model = DummyBinaryCNN(classifier_weight=1.0)
    model.eval()

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        result = gradcam.generate(input_tensor=_positive_input())

    assert result.prediction == 1
    assert result.target_class == "DEFECT"
    assert result.target_score_mode == "raw_logit"


def test_gradcam_supports_frozen_model_by_enabling_input_gradient() -> None:
    model = DummyBinaryCNN(classifier_weight=1.0)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        result = gradcam.generate(
            input_tensor=_positive_input(),
            target_class="DEFECT",
        )

    assert result.cam.shape == (16, 16)
    assert torch.isfinite(result.cam).all()


def test_gradcam_rejects_batch_size_greater_than_one() -> None:
    model = DummyBinaryCNN(classifier_weight=1.0)

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        with pytest.raises(InvalidGradCAMInputError, match="Batch Size 1"):
            gradcam.generate(input_tensor=torch.ones(2, 3, 16, 16))


def test_gradcam_raises_when_cam_is_all_zero() -> None:
    model = ZeroCAMCNN()

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        with pytest.raises(ZeroGradCAMError, match="CAM이 모두 0"):
            gradcam.generate(
                input_tensor=_positive_input(),
                target_class="DEFECT",
            )


def test_gradcam_context_manager_removes_hooks() -> None:
    model = DummyBinaryCNN(classifier_weight=1.0)
    gradcam = GradCAM(model=model, target_layer_name="features.0")

    assert gradcam.is_closed is False

    with gradcam:
        gradcam.generate(input_tensor=_positive_input(), target_class="DEFECT")

    assert gradcam.is_closed is True
    assert len(model.features[0]._forward_hooks) == 0


def test_gradcam_restores_original_training_mode() -> None:
    model = DummyBinaryCNN(classifier_weight=1.0)
    model.train()

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        gradcam.generate(input_tensor=_positive_input(), target_class="DEFECT")

    assert model.training is True
