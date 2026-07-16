from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

GradCAMTargetClass = Literal["NORMAL", "DEFECT"]

NORMAL_CLASS_NAME: Final[str] = "NORMAL"
DEFECT_CLASS_NAME: Final[str] = "DEFECT"
NORMAL_LABEL: Final[int] = 0
DEFECT_LABEL: Final[int] = 1


class GradCAMError(RuntimeError):
    """Grad-CAM 계산 과정에서 발생하는 공통 예외입니다."""


class TargetLayerNotFoundError(GradCAMError):
    """모델에서 요청한 Target Layer를 찾지 못했을 때 발생합니다."""


class InvalidGradCAMInputError(GradCAMError):
    """입력 Tensor 또는 Target Class가 정책과 맞지 않을 때 발생합니다."""


class GradCAMHookError(GradCAMError):
    """Activation 또는 Gradient Hook 결과가 준비되지 않았을 때 발생합니다."""


class InvalidGradCAMTensorError(GradCAMError):
    """Activation, Gradient, CAM Tensor가 유효하지 않을 때 발생합니다."""


class ZeroGradCAMError(GradCAMError):
    """ReLU 이후 CAM이 모두 0이어서 유효한 Heatmap을 만들 수 없을 때 발생합니다."""


@dataclass(frozen=True)
class GradCAMResult:
    """한 장의 이미지에 대한 Grad-CAM 계산 결과입니다."""

    cam: Tensor
    raw_logit: float
    defect_probability: float
    prediction: int
    prediction_class_name: str
    target_class: GradCAMTargetClass
    target_score_mode: str
    target_score_value: float
    target_layer_name: str
    activation_shape: tuple[int, ...]
    gradient_shape: tuple[int, ...]

    def to_metadata_dict(self) -> dict[str, object]:
        """JSON 저장에 사용할 수 있는 Metadata만 반환합니다."""

        return {
            "raw_logit": self.raw_logit,
            "defect_probability": self.defect_probability,
            "prediction": self.prediction,
            "prediction_class_name": self.prediction_class_name,
            "target_class": self.target_class,
            "target_score_mode": self.target_score_mode,
            "target_score_value": self.target_score_value,
            "target_layer_name": self.target_layer_name,
            "activation_shape": list(self.activation_shape),
            "gradient_shape": list(self.gradient_shape),
            "cam_shape": list(self.cam.shape),
            "cam_min": float(self.cam.min().item()),
            "cam_max": float(self.cam.max().item()),
        }


def resolve_target_layer(model: nn.Module, target_layer_name: str) -> nn.Module:
    """점 표기 경로로 모델 내부 Layer를 찾습니다.

    예:
        resnet18.layer4.1.conv2

    숫자 Token은 ``Sequential`` 또는 ``ModuleList``의 Index로 처리합니다.
    """

    if not isinstance(target_layer_name, str) or not target_layer_name.strip():
        raise TargetLayerNotFoundError("Target Layer 이름은 비어 있지 않은 문자열이어야 합니다.")

    current: object = model

    for token in target_layer_name.split("."):
        if not token:
            raise TargetLayerNotFoundError(
                f"Target Layer 경로 형식이 올바르지 않습니다: {target_layer_name!r}"
            )

        try:
            if token.isdigit():
                current = current[int(token)]  # type: ignore[index]
            else:
                current = getattr(current, token)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise TargetLayerNotFoundError(
                f"모델에서 Target Layer를 찾지 못했습니다: {target_layer_name!r}"
            ) from exc

    if not isinstance(current, nn.Module):
        raise TargetLayerNotFoundError(
            f"요청한 경로가 nn.Module이 아닙니다: {target_layer_name!r}"
        )

    return current


def _normalize_target_class(target_class: str | int | None) -> GradCAMTargetClass | None:
    """다양한 Target Class 입력을 내부 표준 문자열로 변환합니다."""

    if target_class is None:
        return None

    if target_class in (DEFECT_LABEL, DEFECT_CLASS_NAME, "defect", "DEFECT"):
        return DEFECT_CLASS_NAME

    if target_class in (NORMAL_LABEL, NORMAL_CLASS_NAME, "normal", "NORMAL"):
        return NORMAL_CLASS_NAME

    raise InvalidGradCAMInputError(
        "target_class는 NORMAL/DEFECT 또는 0/1 중 하나여야 합니다. "
        f"입력값={target_class!r}"
    )


def _extract_single_logit(model_output: Tensor | object) -> Tensor:
    """Binary Classification 모델 출력에서 단일 Logit을 추출합니다."""

    if not isinstance(model_output, Tensor):
        raise InvalidGradCAMTensorError(
            "모델 출력은 torch.Tensor여야 합니다. "
            f"실제 타입={type(model_output).__name__}"
        )

    flattened = model_output.reshape(-1)

    if flattened.numel() != 1:
        raise InvalidGradCAMTensorError(
            "Batch Size 1 Binary Model은 단일 Logit을 출력해야 합니다. "
            f"출력 Shape={tuple(model_output.shape)}"
        )

    logit = flattened[0]

    if not torch.isfinite(logit):
        raise InvalidGradCAMTensorError("모델 Logit에 NaN 또는 Infinity가 포함되어 있습니다.")

    return logit


class GradCAM:
    """PyTorch Hook 기반 Grad-CAM 구현입니다.

    설계 정책:
    - Batch Size는 1만 허용합니다.
    - Forward Hook에서 Target Layer Activation을 저장합니다.
    - Activation Tensor에 Gradient Hook을 등록해 Backward Gradient를 저장합니다.
    - Binary Logit 모델에서 DEFECT는 raw_logit, NORMAL은 -raw_logit을 Target Score로 사용합니다.
    - Frozen Backbone에서도 Gradient가 흐르도록 입력 Tensor 복사본에 requires_grad_(True)를 적용합니다.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        target_layer_name: str,
        epsilon: float = 1e-8,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon은 0보다 커야 합니다.")

        self.model = model
        self.target_layer_name = target_layer_name
        self.target_layer = resolve_target_layer(model, target_layer_name)
        self.epsilon = float(epsilon)

        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._gradient_handle: torch.utils.hooks.RemovableHandle | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """Hook가 해제되었는지 반환합니다."""

        return self._closed

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs

        if not isinstance(output, Tensor):
            raise InvalidGradCAMTensorError(
                "Target Layer 출력은 torch.Tensor여야 합니다. "
                f"실제 타입={type(output).__name__}"
            )

        if output.ndim != 4:
            raise InvalidGradCAMTensorError(
                "Target Layer Activation은 [B, C, H, W] 4차원 Tensor여야 합니다. "
                f"실제 Shape={tuple(output.shape)}"
            )

        if not output.requires_grad:
            raise GradCAMHookError(
                "Target Layer Activation에 Gradient가 활성화되지 않았습니다. "
                "Grad-CAM 호출 경로에서 torch.no_grad()를 사용하지 않았는지 확인하세요."
            )

        if self._gradient_handle is not None:
            self._gradient_handle.remove()
            self._gradient_handle = None

        self._activations = output
        self._gradients = None
        self._gradient_handle = output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradients: Tensor) -> None:
        """Target Score를 Target Activation으로 미분한 Gradient를 저장합니다."""

        self._gradients = gradients.detach()

    def generate(
        self,
        *,
        input_tensor: Tensor,
        target_class: str | int | None = None,
        resize_to_input: bool = True,
    ) -> GradCAMResult:
        """한 장의 입력 이미지에 대한 Grad-CAM을 계산합니다.

        ``target_class=None``이면 모델의 실제 예측 Class를 Target으로 사용합니다.
        """

        if self._closed:
            raise GradCAMHookError("이미 close()된 GradCAM 객체는 다시 사용할 수 없습니다.")

        if not isinstance(input_tensor, Tensor):
            raise InvalidGradCAMInputError("input_tensor는 torch.Tensor여야 합니다.")

        if input_tensor.ndim != 4:
            raise InvalidGradCAMInputError(
                "input_tensor는 [B, C, H, W] 4차원 Tensor여야 합니다. "
                f"실제 Shape={tuple(input_tensor.shape)}"
            )

        if input_tensor.shape[0] != 1:
            raise InvalidGradCAMInputError(
                "Day 6 Grad-CAM은 안전한 표본별 계산을 위해 Batch Size 1만 허용합니다. "
                f"실제 Batch Size={input_tensor.shape[0]}"
            )

        if input_tensor.shape[1] != 3:
            raise InvalidGradCAMInputError(
                "RGB 모델 입력은 Channel 3이어야 합니다. "
                f"실제 Channel={input_tensor.shape[1]}"
            )

        if not torch.is_floating_point(input_tensor):
            raise InvalidGradCAMInputError("input_tensor는 실수형 Tensor여야 합니다.")

        if not torch.isfinite(input_tensor).all():
            raise InvalidGradCAMInputError("input_tensor에 NaN 또는 Infinity가 포함되어 있습니다.")

        normalized_target_class = _normalize_target_class(target_class)
        original_training_mode = self.model.training

        self._activations = None
        self._gradients = None
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

        # Frozen Backbone도 Activation Gradient를 만들 수 있도록 입력 복사본만 Gradient 추적합니다.
        grad_input = input_tensor.detach().clone().requires_grad_(True)

        try:
            model_output = self.model(grad_input)
            raw_logit_tensor = _extract_single_logit(model_output)
            defect_probability_tensor = torch.sigmoid(raw_logit_tensor)

            prediction = (
                DEFECT_LABEL
                if defect_probability_tensor.item() >= 0.5
                else NORMAL_LABEL
            )
            prediction_class_name = (
                DEFECT_CLASS_NAME if prediction == DEFECT_LABEL else NORMAL_CLASS_NAME
            )

            effective_target_class: GradCAMTargetClass = (
                normalized_target_class
                if normalized_target_class is not None
                else prediction_class_name  # type: ignore[assignment]
            )

            if effective_target_class == DEFECT_CLASS_NAME:
                target_score = raw_logit_tensor
                target_score_mode = "raw_logit"
            else:
                target_score = -raw_logit_tensor
                target_score_mode = "negative_raw_logit"

            if self._activations is None:
                raise GradCAMHookError("Forward Hook가 Activation을 저장하지 못했습니다.")

            target_score.backward()

            if self._gradients is None:
                raise GradCAMHookError("Gradient Hook가 Gradient를 저장하지 못했습니다.")

            activations = self._activations.detach()
            gradients = self._gradients

            if activations.shape != gradients.shape:
                raise InvalidGradCAMTensorError(
                    "Activation과 Gradient Shape가 일치하지 않습니다. "
                    f"activation={tuple(activations.shape)}, gradient={tuple(gradients.shape)}"
                )

            if activations.ndim != 4:
                raise InvalidGradCAMTensorError(
                    "Activation과 Gradient는 [B, C, H, W] 4차원이어야 합니다."
                )

            if not torch.isfinite(activations).all():
                raise InvalidGradCAMTensorError(
                    "Activation에 NaN 또는 Infinity가 포함되어 있습니다."
                )

            if not torch.isfinite(gradients).all():
                raise InvalidGradCAMTensorError(
                    "Gradient에 NaN 또는 Infinity가 포함되어 있습니다."
                )

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)

            if resize_to_input:
                cam = F.interpolate(
                    cam,
                    size=input_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if not torch.isfinite(cam).all():
                raise InvalidGradCAMTensorError("CAM에 NaN 또는 Infinity가 포함되어 있습니다.")

            cam_min = cam.amin(dim=(2, 3), keepdim=True)
            cam = cam - cam_min
            cam_max = cam.amax(dim=(2, 3), keepdim=True)

            if float(cam_max.item()) <= self.epsilon:
                raise ZeroGradCAMError(
                    "ReLU 이후 CAM이 모두 0입니다. 잘못된 Heatmap 저장을 중단합니다."
                )

            cam = cam / (cam_max + self.epsilon)
            cam = cam.squeeze(0).squeeze(0).detach().cpu()

            if cam.ndim != 2:
                raise InvalidGradCAMTensorError(
                    f"최종 CAM은 [H, W] 2차원이어야 합니다. 실제 Shape={tuple(cam.shape)}"
                )

            if not torch.isfinite(cam).all():
                raise InvalidGradCAMTensorError(
                    "정규화된 CAM에 NaN 또는 Infinity가 포함되어 있습니다."
                )

            return GradCAMResult(
                cam=cam,
                raw_logit=float(raw_logit_tensor.detach().cpu().item()),
                defect_probability=float(defect_probability_tensor.detach().cpu().item()),
                prediction=prediction,
                prediction_class_name=prediction_class_name,
                target_class=effective_target_class,
                target_score_mode=target_score_mode,
                target_score_value=float(target_score.detach().cpu().item()),
                target_layer_name=self.target_layer_name,
                activation_shape=tuple(activations.shape),
                gradient_shape=tuple(gradients.shape),
            )
        finally:
            self.model.zero_grad(set_to_none=True)
            if original_training_mode:
                self.model.train()
            else:
                self.model.eval()

    def close(self) -> None:
        """등록한 Forward/Gradient Hook를 모두 해제합니다."""

        if self._closed:
            return

        if self._gradient_handle is not None:
            self._gradient_handle.remove()
            self._gradient_handle = None

        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None

        self._activations = None
        self._gradients = None
        self._closed = True

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()
