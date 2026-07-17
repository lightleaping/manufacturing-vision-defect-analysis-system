"""Day 12 Detection Training Step·Inference Step·Tiny Overfit 진단.

[신규 구현]
- Detection 가변 Batch를 CPU로 복사해 이동한다.
- 필수 Loss, NaN·inf Loss, NaN·inf Gradient를 검사한다.
- 실제 ``backward``와 ``optimizer.step``을 수행한다.
- 호출자 Image·Target과 Model Mode를 원래 상태로 복원한다.
- 같은 Batch 반복 학습으로 학습 경로가 연결됐는지 진단한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


EXPECTED_DETECTION_LOSS_KEYS = (
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
)


@dataclass(frozen=True, slots=True)
class DetectionTrainingStepResult:
    """한 Batch 실제 학습 결과."""

    losses: dict[str, float]
    total_loss: float
    elapsed_seconds: float
    gradient_parameter_count: int
    learning_rates: tuple[float, ...]
    batch_size: int
    inputs_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "losses": dict(self.losses),
            "total_loss": self.total_loss,
            "elapsed_seconds": self.elapsed_seconds,
            "gradient_parameter_count": self.gradient_parameter_count,
            "learning_rates": list(self.learning_rates),
            "batch_size": self.batch_size,
            "inputs_unchanged": self.inputs_unchanged,
        }


@dataclass(frozen=True, slots=True)
class DetectionInferenceStepResult:
    """한 Batch Evaluation Prediction과 실행 정보."""

    predictions: tuple[dict[str, Tensor], ...]
    elapsed_seconds: float
    batch_size: int
    inputs_unchanged: bool

    def summary(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "batch_size": self.batch_size,
            "inputs_unchanged": self.inputs_unchanged,
            "prediction_box_counts": [
                int(prediction["boxes"].shape[0])
                for prediction in self.predictions
            ],
        }


def _resolve_cpu_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str):
        resolved = torch.device(device)
    elif isinstance(device, torch.device):
        resolved = device
    else:
        raise TypeError("device must be str or torch.device.")
    if resolved.type != "cpu":
        raise ValueError("The verified Day 12 training path supports CPU only.")
    return resolved


def _validate_batch(
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
) -> None:
    if isinstance(images, Tensor) or not isinstance(images, Sequence):
        raise TypeError("images must be a sequence of tensors.")
    if isinstance(targets, Mapping) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence of mappings.")
    if not images:
        raise ValueError("Detection batch must not be empty.")
    if len(images) != len(targets):
        raise ValueError("images and targets must have the same length.")

    for index, (image, target) in enumerate(zip(images, targets)):
        if not isinstance(image, Tensor):
            raise TypeError(f"images[{index}] must be torch.Tensor.")
        if image.dtype != torch.float32:
            raise TypeError(f"images[{index}] must use float32.")
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"images[{index}] must have shape [3, H, W].")
        if image.numel() == 0 or not bool(torch.isfinite(image).all()):
            raise ValueError(f"images[{index}] must be finite and non-empty.")
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise ValueError(f"images[{index}] must use range [0, 1].")

        if not isinstance(target, Mapping):
            raise TypeError(f"targets[{index}] must be a mapping.")
        missing = {"boxes", "labels"} - set(target)
        if missing:
            raise KeyError(f"targets[{index}] is missing keys: {sorted(missing)}.")
        if any(not isinstance(value, Tensor) for value in target.values()):
            raise TypeError(f"targets[{index}] values must all be tensors.")

        boxes = target["boxes"]
        labels = target["labels"]
        if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise TypeError(f"targets[{index}]['boxes'] must be FloatTensor[N, 4].")
        if not bool(torch.isfinite(boxes).all()):
            raise ValueError(f"targets[{index}]['boxes'] contains NaN or infinity.")
        if boxes.numel() and not bool(
            ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all()
        ):
            raise ValueError(f"targets[{index}]['boxes'] must have positive area.")
        if labels.dtype != torch.int64 or labels.ndim != 1:
            raise TypeError(f"targets[{index}]['labels'] must be Int64Tensor[N].")
        if labels.shape[0] != boxes.shape[0]:
            raise ValueError(f"targets[{index}] box and label counts differ.")
        if labels.numel() and int(labels.min()) < 1:
            raise ValueError(f"targets[{index}] labels must be foreground indexes.")


def _clone_batch(
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    return (
        [image.clone() for image in images],
        [
            {key: value.clone() for key, value in target.items()}
            for target in targets
        ],
    )


def _batch_unchanged(
    originals: tuple[list[Tensor], list[dict[str, Tensor]]],
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
) -> bool:
    original_images, original_targets = originals
    if len(original_images) != len(images) or len(original_targets) != len(targets):
        return False
    if any(not torch.equal(before, after) for before, after in zip(original_images, images)):
        return False
    for before, after in zip(original_targets, targets):
        if set(before) != set(after):
            return False
        if any(not torch.equal(before[key], after[key]) for key in before):
            return False
    return True


def _move_batch(
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
    device: torch.device,
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    return (
        [image.to(device=device) for image in images],
        [
            {key: value.to(device=device) for key, value in target.items()}
            for target in targets
        ],
    )


def _validate_loss_dict(loss_dict: Any) -> tuple[dict[str, float], Tensor]:
    if not isinstance(loss_dict, Mapping):
        raise TypeError("Detection model training output must be a loss mapping.")
    missing = set(EXPECTED_DETECTION_LOSS_KEYS) - set(loss_dict)
    if missing:
        raise KeyError(f"Detection loss mapping is missing: {sorted(missing)}.")

    loss_tensors: list[Tensor] = []
    values: dict[str, float] = {}
    for name, loss in loss_dict.items():
        if not isinstance(name, str) or not isinstance(loss, Tensor):
            raise TypeError("Detection loss keys must be str and values tensors.")
        if loss.numel() != 1:
            raise ValueError(f"Loss {name!r} must be scalar.")
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"Loss {name!r} contains NaN or infinity.")
        loss_tensors.append(loss.reshape(()))
        values[name] = float(loss.detach().cpu().item())

    total_loss = torch.stack(loss_tensors).sum()
    if not bool(torch.isfinite(total_loss)):
        raise FloatingPointError("Total detection loss contains NaN or infinity.")
    return values, total_loss


def _validate_gradients(model: nn.Module) -> int:
    gradient_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_count += 1
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                f"Gradient for parameter {name!r} contains NaN or infinity."
            )
    if gradient_count == 0:
        raise RuntimeError("Backward produced no parameter gradients.")
    return gradient_count


def run_detection_training_step(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
    device: str | torch.device = "cpu",
) -> DetectionTrainingStepResult:
    """한 Batch에 대해 Forward·Backward·Optimizer Step을 실제 수행한다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    resolved_device = _resolve_cpu_device(device)
    _validate_batch(images, targets)
    originals = _clone_batch(images, targets)
    model_images, model_targets = _move_batch(images, targets, resolved_device)
    original_mode = bool(model.training)

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    try:
        model.train()
        loss_dict = model(model_images, model_targets)
        loss_values, total_loss_tensor = _validate_loss_dict(loss_dict)
        total_loss_tensor.backward()
        gradient_count = _validate_gradients(model)
        optimizer.step()
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        raise
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        model.train(mode=original_mode)

    elapsed = time.perf_counter() - started
    unchanged = _batch_unchanged(originals, images, targets)
    if not unchanged:
        raise RuntimeError("Detection training step mutated caller inputs.")

    return DetectionTrainingStepResult(
        losses=loss_values,
        total_loss=float(total_loss_tensor.detach().cpu().item()),
        elapsed_seconds=float(elapsed),
        gradient_parameter_count=gradient_count,
        learning_rates=tuple(
            float(group["lr"])
            for group in optimizer.param_groups
        ),
        batch_size=len(images),
        inputs_unchanged=unchanged,
    )


def _validate_predictions(
    predictions: Any,
    *,
    expected_count: int,
    num_classes: int,
) -> tuple[dict[str, Tensor], ...]:
    if not isinstance(predictions, Sequence) or isinstance(predictions, Tensor):
        raise TypeError("Detection evaluation output must be a sequence.")
    if len(predictions) != expected_count:
        raise ValueError("Prediction count must match image count.")

    normalized: list[dict[str, Tensor]] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            raise TypeError(f"predictions[{index}] must be a mapping.")
        missing = {"boxes", "labels", "scores"} - set(prediction)
        if missing:
            raise KeyError(f"predictions[{index}] is missing: {sorted(missing)}.")
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]
        if not isinstance(boxes, Tensor) or boxes.dtype != torch.float32:
            raise TypeError("prediction boxes must be float32 tensors.")
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("prediction boxes must have shape [N, 4].")
        if not isinstance(labels, Tensor) or labels.dtype != torch.int64:
            raise TypeError("prediction labels must be int64 tensors.")
        if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
            raise ValueError("prediction label count must match boxes.")
        if labels.numel() and (
            int(labels.min()) < 1 or int(labels.max()) >= num_classes
        ):
            raise ValueError("prediction labels are outside foreground range.")
        if not isinstance(scores, Tensor) or not scores.dtype.is_floating_point:
            raise TypeError("prediction scores must be floating tensors.")
        if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
            raise ValueError("prediction score count must match boxes.")
        if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
            raise ValueError("prediction boxes or scores contain NaN or infinity.")
        if scores.numel() and not bool(((scores >= 0.0) & (scores <= 1.0)).all()):
            raise ValueError("prediction scores must be in [0, 1].")
        normalized.append(
            {
                "boxes": boxes.detach().cpu().clone(),
                "labels": labels.detach().cpu().clone(),
                "scores": scores.detach().cpu().clone(),
            }
        )
    return tuple(normalized)


def run_detection_inference_step(
    *,
    model: nn.Module,
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
    num_classes: int,
    device: str | torch.device = "cpu",
) -> DetectionInferenceStepResult:
    """Parameter를 변경하지 않고 Evaluation Prediction을 반환한다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
        raise ValueError("num_classes must include background and foreground.")
    resolved_device = _resolve_cpu_device(device)
    _validate_batch(images, targets)
    originals = _clone_batch(images, targets)
    model_images, _ = _move_batch(images, targets, resolved_device)
    original_mode = bool(model.training)

    started = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            predictions = model(model_images)
        normalized = _validate_predictions(
            predictions,
            expected_count=len(images),
            num_classes=num_classes,
        )
    finally:
        model.train(mode=original_mode)

    unchanged = _batch_unchanged(originals, images, targets)
    if not unchanged:
        raise RuntimeError("Detection inference step mutated caller inputs.")
    return DetectionInferenceStepResult(
        predictions=normalized,
        elapsed_seconds=float(time.perf_counter() - started),
        batch_size=len(images),
        inputs_unchanged=unchanged,
    )


def run_tiny_overfit_diagnostic(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
    steps: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """같은 Batch를 여러 번 학습해 Loss 변화와 학습 연결을 기록한다.

    짧은 Pilot에서는 Loss 감소를 성능 합격 조건으로 강제하지 않는다. 모든
    Step이 finite이고 Parameter Gradient가 생성되는지가 구조 합격 기준이며,
    감소 여부는 후속 Epoch 결정용 진단 정보다.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive int.")

    results = [
        run_detection_training_step(
            model=model,
            optimizer=optimizer,
            images=images,
            targets=targets,
            device=device,
        )
        for _ in range(steps)
    ]
    losses = [result.total_loss for result in results]
    finite = all(math.isfinite(value) for value in losses)
    return {
        "steps": steps,
        "step_results": [result.to_dict() for result in results],
        "initial_total_loss": losses[0],
        "final_total_loss": losses[-1],
        "minimum_total_loss": min(losses),
        "loss_decrease_observed": min(losses) < losses[0] if len(losses) > 1 else None,
        "all_losses_finite": finite,
        "average_step_seconds": sum(
            result.elapsed_seconds for result in results
        ) / len(results),
    }
