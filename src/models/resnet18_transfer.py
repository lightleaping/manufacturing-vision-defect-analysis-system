"""ResNet18 기반 정상·불량 이진 분류 전이학습 모델.

이 모듈은 torchvision의 ResNet18을 제조 이미지 이진 분류 문제에 맞게
다음과 같이 변경한다.

1. ImageNet 사전학습 가중치를 선택적으로 사용한다.
2. 기존 1000개 클래스 출력 FC Layer를 1개 Raw Logit 출력으로 교체한다.
3. Feature Extractor 방식에서는 Backbone Parameter를 동결한다.
4. 동결된 Backbone의 BatchNorm Running Statistics가 학습 중 변하지 않도록
   Backbone은 Evaluation Mode, 새 FC Head만 Training Mode로 유지한다.
5. 향후 Grad-CAM에서 사용할 마지막 Convolution Layer를 공개한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Self

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


RESNET18_TRANSFER_MODEL_NAME = "resnet18_transfer"
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
GRADCAM_TARGET_LAYER_NAME = "resnet18.layer4.1.conv2"

EXPECTED_INPUT_CHANNELS = 3
BINARY_OUTPUT_FEATURES = 1
RESNET18_FEATURE_DIM = 512


@dataclass(frozen=True)
class ModelParameterCounts:
    """Model Parameter 수를 학습 가능 여부에 따라 구분한 결과."""

    total: int
    trainable: int
    frozen: int

    def __post_init__(self) -> None:
        """세 값이 서로 모순되지 않는지 검증한다."""
        if self.total < 0:
            raise ValueError("total parameter count must be non-negative.")

        if self.trainable < 0:
            raise ValueError("trainable parameter count must be non-negative.")

        if self.frozen < 0:
            raise ValueError("frozen parameter count must be non-negative.")

        if self.total != self.trainable + self.frozen:
            raise ValueError(
                "total parameter count must equal trainable + frozen."
            )


def count_model_parameters(model: nn.Module) -> ModelParameterCounts:
    """Model의 Total·Trainable·Frozen Parameter 수를 계산한다.

    Args:
        model:
            Parameter 수를 계산할 PyTorch Model.

    Returns:
        ModelParameterCounts:
            전체, 학습 가능, 동결 Parameter 수.

    Raises:
        TypeError:
            model이 nn.Module이 아닐 때.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module.")

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return ModelParameterCounts(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
    )


class ResNet18Transfer(nn.Module):
    """ResNet18을 사용하는 제조 이미지 이진 분류 전이학습 Model.

    Args:
        weights:
            사용할 torchvision ResNet18 Weight.

            - ResNet18_Weights.DEFAULT:
              ImageNet 사전학습 Weight 사용

            - None:
              무작위 초기화.
              네트워크가 없는 단위 테스트에 사용

        freeze_backbone:
            True이면 새 FC Head를 제외한 Backbone 전체를 동결한다.

        progress:
            사전학습 Weight 다운로드 진행률 표시 여부.

    Input:
        images:
            Shape [B, 3, H, W]인 Floating Point Image Tensor.

    Output:
        Shape [B]인 Raw Logit Tensor.

        Sigmoid는 Model 내부에서 적용하지 않는다.
    """

    model_name = RESNET18_TRANSFER_MODEL_NAME
    classification_threshold = DEFAULT_CLASSIFICATION_THRESHOLD
    gradcam_target_layer_name = GRADCAM_TARGET_LAYER_NAME

    def __init__(
        self,
        *,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        freeze_backbone: bool = True,
        progress: bool = False,
    ) -> None:
        super().__init__()

        if (
            weights is not None
            and not isinstance(weights, ResNet18_Weights)
        ):
            raise TypeError(
                "weights must be "
                "torchvision.models.ResNet18_Weights or None."
            )

        if not isinstance(freeze_backbone, bool):
            raise TypeError("freeze_backbone must be bool.")

        if not isinstance(progress, bool):
            raise TypeError("progress must be bool.")

        self.weights = weights
        self.freeze_backbone = freeze_backbone

        # [신규 구현]
        # weights=None이면 인터넷 연결이나 Weight Cache 없이
        # ResNet18 구조만 생성한다.
        #
        # weights=ResNet18_Weights.DEFAULT이면
        # Cache가 없을 때 torchvision이 다운로드를 시도할 수 있다.
        try:
            backbone = resnet18(
                weights=weights,
                progress=progress,
            )
        except Exception as exc:
            weight_description = (
                "randomly initialized weights"
                if weights is None
                else f"pretrained weights {weights}"
            )

            raise RuntimeError(
                "Failed to create torchvision ResNet18 with "
                f"{weight_description}. "
                "When running offline, use weights=None."
            ) from exc

        original_fc_in_features = backbone.fc.in_features

        if original_fc_in_features != RESNET18_FEATURE_DIM:
            raise RuntimeError(
                "Unexpected ResNet18 FC input feature size: "
                f"expected {RESNET18_FEATURE_DIM}, "
                f"got {original_fc_in_features}."
            )

        # [신규 구현]
        # 기존 ResNet18 전체 Parameter를 먼저 동결한다.
        #
        # 그다음 새 FC Layer를 생성하면 새 FC의 Parameter는
        # requires_grad=True 기본값을 유지한다.
        #
        # 결과:
        # Backbone Parameter = Frozen
        # 새 FC Parameter    = Trainable
        if freeze_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False

        # [신규 구현]
        # 기존 ImageNet 1000-Class Head를 제거하고
        # DEFECT 여부를 나타내는 1개 Raw Logit Head로 교체한다.
        backbone.fc = nn.Linear(
            in_features=original_fc_in_features,
            out_features=BINARY_OUTPUT_FEATURES,
            bias=True,
        )

        self.resnet18 = backbone

        # nn.Module은 생성 직후 기본적으로 Training Mode이다.
        #
        # Frozen Feature Extractor 정책을 생성 직후부터 맞추기 위해
        # Override한 train(True)를 호출한다.
        #
        # 최종 상태:
        # Wrapper     = train
        # Backbone    = eval
        # FC Head     = train
        self.train(mode=True)

    @property
    def classification_head(self) -> nn.Linear:
        """이진 분류를 담당하는 새 FC Head를 반환한다."""
        head = self.resnet18.fc

        if not isinstance(head, nn.Linear):
            raise RuntimeError(
                "ResNet18 classification head must be torch.nn.Linear."
            )

        return head

    @property
    def gradcam_target_layer(self) -> nn.Conv2d:
        """Grad-CAM Target으로 사용할 마지막 conv2를 반환한다."""
        target_layer = self.resnet18.layer4[-1].conv2

        if not isinstance(target_layer, nn.Conv2d):
            raise RuntimeError(
                "Grad-CAM target layer must be torch.nn.Conv2d."
            )

        return target_layer

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Optimizer에 전달할 학습 가능한 Parameter만 반환한다."""
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def parameter_counts(self) -> ModelParameterCounts:
        """현재 Model의 Parameter 수 구분 결과를 반환한다."""
        return count_model_parameters(self)

    def train(self, mode: bool = True) -> Self:
        """Training·Evaluation Mode를 설정한다.

        Frozen Feature Extractor 정책:

        - Wrapper 자체는 요청받은 mode를 유지한다.
        - ResNet18 Backbone 전체는 Evaluation Mode로 고정한다.
        - 새 FC Head만 요청받은 mode를 따른다.

        외부 Epoch Runner가 model.train()을 호출해도
        Backbone 내부 BatchNorm Running Statistics가 갱신되지 않는다.
        """
        if not isinstance(mode, bool):
            raise TypeError("mode must be bool.")

        # 먼저 Wrapper와 모든 하위 Module에 일반적인
        # train/eval 상태를 적용한다.
        super().train(mode)

        if self.freeze_backbone:
            # ResNet18 전체를 Evaluation Mode로 고정한다.
            #
            # Conv·ReLU·Pooling은 train/eval 차이가 거의 없지만
            # BatchNorm Running Statistics 변경을 확실히 차단한다.
            self.resnet18.eval()

            # 새 Classification Head만 요청받은 상태를 적용한다.
            #
            # mode=True  → FC Head Training Mode
            # mode=False → FC Head Evaluation Mode
            self.classification_head.train(mode)

        return self

    def forward(self, images: Tensor) -> Tensor:
        """Image Batch를 받아 Sample별 DEFECT Raw Logit을 반환한다."""
        self._validate_input(images)

        # torchvision ResNet18의 새 FC 출력:
        # [B, 1]
        logits_2d = self.resnet18(images)

        expected_shape = (
            images.shape[0],
            BINARY_OUTPUT_FEATURES,
        )

        if tuple(logits_2d.shape) != expected_shape:
            raise RuntimeError(
                "ResNet18 binary head returned an unexpected shape: "
                f"expected {expected_shape}, "
                f"got {tuple(logits_2d.shape)}."
            )

        # [B, 1] → [B]
        #
        # squeeze()를 인자 없이 사용하면 Batch Size가 1일 때
        # 결과가 Scalar가 될 수 있다.
        #
        # 따라서 Class Dimension인 dim=1만 제거한다.
        return logits_2d.squeeze(dim=1)

    @staticmethod
    def _validate_input(images: Tensor) -> None:
        """Forward 입력의 최소 계약을 검증한다."""
        if not isinstance(images, Tensor):
            raise TypeError("images must be a torch.Tensor.")

        if images.ndim != 4:
            raise ValueError(
                "images must have shape [B, C, H, W], "
                f"but got ndim={images.ndim}."
            )

        if images.shape[0] <= 0:
            raise ValueError(
                "images batch size must be greater than zero."
            )

        if images.shape[1] != EXPECTED_INPUT_CHANNELS:
            raise ValueError(
                f"images must have {EXPECTED_INPUT_CHANNELS} channels, "
                f"but got {images.shape[1]}."
            )

        if images.shape[2] <= 0 or images.shape[3] <= 0:
            raise ValueError(
                "images height and width must be greater than zero."
            )

        if not images.is_floating_point():
            raise TypeError(
                "images must use a floating point dtype."
            )


def create_resnet18_transfer_model(
    *,
    use_pretrained_weights: bool = True,
    freeze_backbone: bool = True,
    progress: bool = False,
) -> ResNet18Transfer:
    """Production·Test 환경을 구분해 ResNet18Transfer를 생성한다.

    Production:

        create_resnet18_transfer_model(
            use_pretrained_weights=True,
            freeze_backbone=True,
        )

    Offline Unit Test:

        create_resnet18_transfer_model(
            use_pretrained_weights=False,
            freeze_backbone=True,
        )
    """
    if not isinstance(use_pretrained_weights, bool):
        raise TypeError(
            "use_pretrained_weights must be bool."
        )

    weights = (
        ResNet18_Weights.DEFAULT
        if use_pretrained_weights
        else None
    )

    return ResNet18Transfer(
        weights=weights,
        freeze_backbone=freeze_backbone,
        progress=progress,
    )


__all__ = [
    "BINARY_OUTPUT_FEATURES",
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "EXPECTED_INPUT_CHANNELS",
    "GRADCAM_TARGET_LAYER_NAME",
    "ModelParameterCounts",
    "RESNET18_FEATURE_DIM",
    "RESNET18_TRANSFER_MODEL_NAME",
    "ResNet18Transfer",
    "count_model_parameters",
    "create_resnet18_transfer_model",
]