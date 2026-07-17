"""Detection Model의 Training Loss와 Evaluation Prediction 계약 검증."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import Tensor, nn


EXPECTED_LOSS_KEYS = (
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
)
EXPECTED_PREDICTION_KEYS = ("boxes", "labels", "scores")


@dataclass(frozen=True, slots=True)
class DetectionModelSmokeResult:
    payload: dict[str, Any]


def _clone_targets(
    targets: Sequence[Mapping[str, Tensor]],
) -> list[dict[str, Tensor]]:
    return [
        {key: value.clone() for key, value in target.items()}
        for target in targets
    ]


def _validate_inputs(
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
    *,
    num_classes: int,
) -> None:
    if isinstance(images, Tensor) or not isinstance(images, Sequence):
        raise TypeError("images must be a sequence of image tensors.")
    if not images:
        raise ValueError("images must not be empty.")
    if not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence of mappings.")
    if len(images) != len(targets):
        raise ValueError("images and targets must have the same length.")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be int.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    for image_index, image in enumerate(images):
        if not isinstance(image, Tensor):
            raise TypeError(f"images[{image_index}] must be torch.Tensor.")
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"images[{image_index}] must have shape [3, H, W]."
            )
        if image.dtype != torch.float32:
            raise TypeError(f"images[{image_index}] must be float32.")
        if not bool(torch.isfinite(image).all()):
            raise ValueError(f"images[{image_index}] contains non-finite values.")
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise ValueError(f"images[{image_index}] must be in [0, 1].")

        target = targets[image_index]
        if not isinstance(target, Mapping):
            raise TypeError(f"targets[{image_index}] must be a mapping.")
        if "boxes" not in target or "labels" not in target:
            raise KeyError("Every target must contain boxes and labels.")
        boxes = target["boxes"]
        labels = target["labels"]
        if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("target boxes must be FloatTensor[N, 4].")
        if labels.dtype != torch.int64 or labels.ndim != 1:
            raise ValueError("target labels must be Int64Tensor[N].")
        if boxes.shape[0] != labels.shape[0]:
            raise ValueError("target boxes and labels count must match.")
        if boxes.shape[0] == 0:
            raise ValueError("Smoke Test target must contain at least one box.")
        if not bool(torch.isfinite(boxes).all()):
            raise ValueError("target boxes contain non-finite values.")
        if not bool((boxes[:, 2] > boxes[:, 0]).all()):
            raise ValueError("Every box must satisfy xmax > xmin.")
        if not bool((boxes[:, 3] > boxes[:, 1]).all()):
            raise ValueError("Every box must satisfy ymax > ymin.")
        if int(labels.min()) < 1 or int(labels.max()) >= num_classes:
            raise ValueError("Target labels must be in [1, num_classes - 1].")


def _verify_unchanged(
    before_images: Sequence[Tensor],
    after_images: Sequence[Tensor],
    before_targets: Sequence[Mapping[str, Tensor]],
    after_targets: Sequence[Mapping[str, Tensor]],
) -> bool:
    images_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(before_images, after_images)
    )
    targets_unchanged = all(
        set(before) == set(after)
        and all(torch.equal(before[key], after[key]) for key in before)
        for before, after in zip(before_targets, after_targets)
    )
    return images_unchanged and targets_unchanged


def _validate_loss_dict(losses: Any) -> dict[str, float]:
    if not isinstance(losses, Mapping):
        raise TypeError("Training forward must return a loss mapping.")
    missing = [key for key in EXPECTED_LOSS_KEYS if key not in losses]
    if missing:
        raise KeyError(f"Training loss mapping is missing keys: {missing}.")

    values: dict[str, float] = {}
    for key in EXPECTED_LOSS_KEYS:
        value = losses[key]
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise TypeError(f"{key} must be a scalar Tensor.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} is not finite.")
        values[key] = float(value.detach().cpu().item())
    return values


def _validate_predictions(
    predictions: Any,
    *,
    image_count: int,
    num_classes: int,
) -> list[dict[str, Any]]:
    if not isinstance(predictions, Sequence):
        raise TypeError("Evaluation forward must return a prediction sequence.")
    if len(predictions) != image_count:
        raise ValueError("Prediction count must match image count.")

    summaries: list[dict[str, Any]] = []
    for prediction_index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            raise TypeError("Every prediction must be a mapping.")
        missing = [key for key in EXPECTED_PREDICTION_KEYS if key not in prediction]
        if missing:
            raise KeyError(f"Prediction is missing keys: {missing}.")

        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]
        if not all(isinstance(value, Tensor) for value in (boxes, labels, scores)):
            raise TypeError("Prediction boxes, labels and scores must be tensors.")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Prediction boxes must have shape [N, 4].")
        if labels.ndim != 1 or scores.ndim != 1:
            raise ValueError("Prediction labels and scores must have shape [N].")
        if not (boxes.shape[0] == labels.shape[0] == scores.shape[0]):
            raise ValueError("Prediction boxes, labels and scores count must match.")
        if boxes.dtype != torch.float32:
            raise TypeError("Prediction boxes must be float32.")
        if labels.dtype != torch.int64:
            raise TypeError("Prediction labels must be int64.")
        if scores.dtype != torch.float32:
            raise TypeError("Prediction scores must be float32.")
        if not bool(torch.isfinite(boxes).all()):
            raise ValueError("Prediction boxes contain non-finite values.")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Prediction scores contain non-finite values.")
        if scores.numel() and not bool(((scores >= 0.0) & (scores <= 1.0)).all()):
            raise ValueError("Prediction scores must be in [0, 1].")
        if labels.numel() and (
            int(labels.min()) < 1 or int(labels.max()) >= num_classes
        ):
            raise ValueError("Prediction labels are outside the model class range.")
        if boxes.numel() and (
            not bool((boxes[:, 2] >= boxes[:, 0]).all())
            or not bool((boxes[:, 3] >= boxes[:, 1]).all())
        ):
            raise ValueError("Prediction boxes contain reversed coordinates.")

        summaries.append(
            {
                "prediction_index": prediction_index,
                "box_count": int(boxes.shape[0]),
                "boxes_shape": list(boxes.shape),
                "labels_shape": list(labels.shape),
                "scores_shape": list(scores.shape),
                "max_score": (
                    None if scores.numel() == 0 else float(scores.max().item())
                ),
                "min_score": (
                    None if scores.numel() == 0 else float(scores.min().item())
                ),
                "unique_labels": sorted(
                    {int(value) for value in labels.detach().cpu().tolist()}
                ),
            }
        )
    return summaries


def run_detection_model_smoke_validation(
    *,
    model: nn.Module,
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
    num_classes: int,
    device: str | torch.device = "cpu",
) -> DetectionModelSmokeResult:
    """같은 입력으로 Training Loss와 Evaluation Prediction을 검증한다.

    역전파는 Day 12 범위이므로 수행하지 않는다. Training Mode Loss 계산도
    ``torch.no_grad()``에서 실행해 Day 11 CPU 메모리 사용량을 줄인다.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise ValueError("Day 11 Smoke Test supports CPU only.")

    _validate_inputs(images, targets, num_classes=num_classes)
    original_mode = bool(model.training)
    original_images = [image.clone() for image in images]
    original_targets = _clone_targets(targets)
    model_images = [image.to(resolved_device) for image in images]
    model_targets = [
        {key: value.to(resolved_device) for key, value in target.items()}
        for target in targets
    ]

    try:
        model.train()
        training_started = time.perf_counter()
        with torch.no_grad():
            losses = model(model_images, model_targets)
        training_seconds = time.perf_counter() - training_started
        loss_values = _validate_loss_dict(losses)

        model.eval()
        evaluation_started = time.perf_counter()
        with torch.inference_mode():
            predictions = model(model_images)
        evaluation_seconds = time.perf_counter() - evaluation_started
        prediction_summaries = _validate_predictions(
            predictions,
            image_count=len(images),
            num_classes=num_classes,
        )
    finally:
        model.train(mode=original_mode)

    inputs_unchanged = _verify_unchanged(
        original_images,
        images,
        original_targets,
        targets,
    )
    checks = {
        "model_device_is_cpu": all(
            parameter.device.type == "cpu" for parameter in model.parameters()
        ),
        "expected_loss_keys_present": True,
        "all_losses_finite": all(
            torch.isfinite(torch.tensor(value)).item()
            for value in loss_values.values()
        ),
        "prediction_count_matches_input": (
            len(prediction_summaries) == len(images)
        ),
        "prediction_contract_valid": True,
        "inputs_unchanged": inputs_unchanged,
        "model_mode_restored": bool(model.training) == original_mode,
    }
    payload: dict[str, Any] = {
        "training_forward": {
            "mode": "train",
            "gradient_tracking": False,
            "backward_executed": False,
            "elapsed_seconds": round(training_seconds, 6),
            "loss_keys": list(loss_values),
            "losses": loss_values,
            "total_loss": float(sum(loss_values.values())),
        },
        "evaluation_forward": {
            "mode": "eval",
            "elapsed_seconds": round(evaluation_seconds, 6),
            "predictions": prediction_summaries,
        },
        "checks": checks,
        "validation_passed": all(checks.values()),
    }
    return DetectionModelSmokeResult(payload=payload)
