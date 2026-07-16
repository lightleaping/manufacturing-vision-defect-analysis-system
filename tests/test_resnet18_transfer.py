"""ResNet18 전이학습 Model 단위 테스트.

모든 실제 Model 생성은 weights=None을 사용하므로 인터넷 연결과
사전학습 Weight Cache에 의존하지 않는다.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torchvision.models import ResNet18_Weights

from src.models.resnet18_transfer import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    GRADCAM_TARGET_LAYER_NAME,
    RESNET18_TRANSFER_MODEL_NAME,
    ModelParameterCounts,
    ResNet18Transfer,
    count_model_parameters,
    create_resnet18_transfer_model,
)


EXPECTED_TOTAL_PARAMETERS = 11_177_025
EXPECTED_HEAD_PARAMETERS = 513
EXPECTED_FROZEN_PARAMETERS = 11_176_512


@pytest.fixture(scope="module")
def frozen_model() -> ResNet18Transfer:
    """여러 Test에서 재사용할 Frozen Feature Extractor Model."""
    return ResNet18Transfer(
        weights=None,
        freeze_backbone=True,
    )


@pytest.fixture(scope="module")
def unfrozen_model() -> ResNet18Transfer:
    """전체 Fine-tuning 확장 경로를 검증할 Model."""
    return ResNet18Transfer(
        weights=None,
        freeze_backbone=False,
    )


def test_create_resnet18_transfer_model_without_network_dependency() -> None:
    """Factory가 weights=None 경로로 Model을 정상 생성하는지 확인한다."""
    model = create_resnet18_transfer_model(
        use_pretrained_weights=False,
        freeze_backbone=True,
    )

    assert isinstance(model, ResNet18Transfer)
    assert model.weights is None
    assert model.freeze_backbone is True


def test_resnet18_transfer_public_metadata(
    frozen_model: ResNet18Transfer,
) -> None:
    """Checkpoint·평가·Grad-CAM에서 사용할 공개 Metadata를 확인한다."""
    assert frozen_model.model_name == RESNET18_TRANSFER_MODEL_NAME

    assert (
        frozen_model.classification_threshold
        == DEFAULT_CLASSIFICATION_THRESHOLD
        == 0.5
    )

    assert (
        frozen_model.gradcam_target_layer_name
        == GRADCAM_TARGET_LAYER_NAME
    )


def test_resnet18_transfer_replaces_fc_with_binary_head(
    frozen_model: ResNet18Transfer,
) -> None:
    """기존 1000-Class FC가 512→1 Linear Layer로 교체됐는지 확인한다."""
    assert isinstance(
        frozen_model.classification_head,
        nn.Linear,
    )

    assert frozen_model.classification_head.in_features == 512
    assert frozen_model.classification_head.out_features == 1
    assert frozen_model.classification_head.bias is not None


@pytest.mark.parametrize("batch_size", [1, 2])
def test_resnet18_transfer_forward_returns_one_logit_per_sample(
    frozen_model: ResNet18Transfer,
    batch_size: int,
) -> None:
    """Batch Size 1에서도 Scalar가 아닌 [B] Logit을 반환하는지 확인한다."""
    frozen_model.eval()

    # 단위 테스트 속도를 위해 64×64를 사용한다.
    # 실제 프로젝트 입력 224×224는 별도 Smoke Test에서 확인한다.
    images = torch.randn(
        batch_size,
        3,
        64,
        64,
    )

    with torch.inference_mode():
        logits = frozen_model(images)

    assert logits.shape == (batch_size,)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_resnet18_transfer_does_not_contain_sigmoid_module(
    frozen_model: ResNet18Transfer,
) -> None:
    """BCEWithLogitsLoss 사용을 위해 내부 Sigmoid가 없는지 확인한다."""
    sigmoid_modules = [
        module
        for module in frozen_model.modules()
        if isinstance(module, nn.Sigmoid)
    ]

    assert sigmoid_modules == []


def test_frozen_feature_extractor_parameter_counts(
    frozen_model: ResNet18Transfer,
) -> None:
    """Backbone 동결 시 FC Head의 513개 Parameter만 학습되는지 확인한다."""
    assert frozen_model.parameter_counts() == ModelParameterCounts(
        total=EXPECTED_TOTAL_PARAMETERS,
        trainable=EXPECTED_HEAD_PARAMETERS,
        frozen=EXPECTED_FROZEN_PARAMETERS,
    )


def test_only_classification_head_parameters_are_trainable_when_frozen(
    frozen_model: ResNet18Transfer,
) -> None:
    """Frozen 정책에서 Optimizer 대상이 FC Weight·Bias뿐인지 확인한다."""
    trainable_parameter_names = {
        name
        for name, parameter in frozen_model.named_parameters()
        if parameter.requires_grad
    }

    trainable_parameter_ids = {
        id(parameter)
        for parameter in frozen_model.trainable_parameters()
    }

    head_parameter_ids = {
        id(parameter)
        for parameter in frozen_model.classification_head.parameters()
    }

    assert trainable_parameter_names == {
        "resnet18.fc.weight",
        "resnet18.fc.bias",
    }

    assert trainable_parameter_ids == head_parameter_ids


def test_train_keeps_frozen_backbone_and_batchnorm_in_evaluation_mode(
    frozen_model: ResNet18Transfer,
) -> None:
    """Epoch Runner의 model.train() 후에도 Backbone BN이 고정되는지 확인한다."""
    returned_model = frozen_model.train()

    batch_norm_layers = [
        module
        for module in frozen_model.resnet18.modules()
        if isinstance(module, nn.BatchNorm2d)
    ]

    assert returned_model is frozen_model
    assert frozen_model.training is True
    assert frozen_model.resnet18.training is False
    assert frozen_model.classification_head.training is True
    assert batch_norm_layers

    assert all(
        layer.training is False
        for layer in batch_norm_layers
    )


def test_eval_sets_wrapper_backbone_and_head_to_evaluation_mode(
    frozen_model: ResNet18Transfer,
) -> None:
    """Validation·Test 시 전체 Model이 Evaluation Mode가 되는지 확인한다."""
    returned_model = frozen_model.eval()

    assert returned_model is frozen_model
    assert frozen_model.training is False
    assert frozen_model.resnet18.training is False
    assert frozen_model.classification_head.training is False


def test_unfrozen_model_allows_full_finetuning(
    unfrozen_model: ResNet18Transfer,
) -> None:
    """향후 Fine-tuning 시 전체 Parameter와 BatchNorm이 학습되는지 확인한다."""
    unfrozen_model.train()

    counts = count_model_parameters(unfrozen_model)

    first_batch_norm = next(
        module
        for module in unfrozen_model.resnet18.modules()
        if isinstance(module, nn.BatchNorm2d)
    )

    assert counts.total == EXPECTED_TOTAL_PARAMETERS
    assert counts.trainable == EXPECTED_TOTAL_PARAMETERS
    assert counts.frozen == 0
    assert unfrozen_model.resnet18.training is True
    assert unfrozen_model.classification_head.training is True
    assert first_batch_norm.training is True


def test_gradcam_target_layer_is_last_layer4_conv2(
    frozen_model: ResNet18Transfer,
) -> None:
    """Grad-CAM Target Layer가 layer4 마지막 Block의 conv2인지 확인한다."""
    assert (
        frozen_model.gradcam_target_layer
        is frozen_model.resnet18.layer4[-1].conv2
    )

    assert isinstance(
        frozen_model.gradcam_target_layer,
        nn.Conv2d,
    )

    assert frozen_model.gradcam_target_layer.out_channels == 512


@pytest.mark.parametrize(
    ("invalid_input", "expected_exception", "expected_message"),
    [
        (
            torch.randn(3, 64, 64),
            ValueError,
            r"shape \[B, C, H, W\]",
        ),
        (
            torch.randn(2, 1, 64, 64),
            ValueError,
            "must have 3 channels",
        ),
        (
            torch.ones(
                2,
                3,
                64,
                64,
                dtype=torch.int64,
            ),
            TypeError,
            "floating point dtype",
        ),
    ],
)
def test_resnet18_transfer_rejects_invalid_input(
    frozen_model: ResNet18Transfer,
    invalid_input: torch.Tensor,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """잘못된 Shape·Channel·Dtype 입력에 명확한 예외를 발생시키는지 확인한다."""
    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        frozen_model(invalid_input)


def test_constructor_rejects_invalid_weights_type() -> None:
    """문자열 등 잘못된 Weight 입력을 조기에 차단하는지 확인한다."""
    with pytest.raises(
        TypeError,
        match="ResNet18_Weights or None",
    ):
        ResNet18Transfer(
            weights="DEFAULT",  # type: ignore[arg-type]
            freeze_backbone=True,
        )


def test_constructor_accepts_official_weight_enum_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFAULT Enum 전달 계약만 검증하며 실제 다운로드는 하지 않는다."""
    captured_weights: list[ResNet18_Weights | None] = []

    def fake_resnet18(
        *,
        weights: ResNet18_Weights | None,
        progress: bool,
    ) -> nn.Module:
        captured_weights.append(weights)

        # Patch 내부에서는 weights=None으로
        # 실제 ResNet18 구조만 생성한다.
        from torchvision.models import resnet18

        return resnet18(
            weights=None,
            progress=progress,
        )

    monkeypatch.setattr(
        "src.models.resnet18_transfer.resnet18",
        fake_resnet18,
    )

    model = ResNet18Transfer(
        weights=ResNet18_Weights.DEFAULT,
        freeze_backbone=True,
    )

    assert isinstance(model, ResNet18Transfer)

    assert captured_weights == [
        ResNet18_Weights.DEFAULT,
    ]