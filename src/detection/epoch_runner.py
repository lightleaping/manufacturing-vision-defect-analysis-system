"""Day 12 Detection 전체 Epoch 학습과 Validation 평가 실행기.

[기존 코드 참고]
- ``src.detection.trainer``의 검증된 1-Batch Forward·Backward·Inference를
  그대로 재사용한다.
- ``src.detection.metrics``의 Class-aware Greedy Matching과 AP@0.50을
  Validation Best Checkpoint 선택 기준으로 사용한다.

[신규 구현]
- 전체 DataLoader를 순회하면서 Loss와 실행 시간을 집계한다.
- Validation Prediction을 모아 Precision·Recall·F1·mAP@0.50을 계산한다.
- NaN·inf와 빈 DataLoader를 즉시 거부한다.
- 호출자가 진행 상황을 출력할 수 있도록 Progress Callback을 지원한다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from src.detection.metrics import calculate_detection_metrics
from src.detection.trainer import (
    run_detection_inference_step,
    run_detection_training_step,
)


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class DetectionTrainingEpochResult:
    """한 Epoch 학습 집계 결과."""

    epoch_index: int
    batch_count: int
    sample_count: int
    average_losses: dict[str, float]
    minimum_total_loss: float
    maximum_total_loss: float
    average_batch_seconds: float
    elapsed_seconds: float
    learning_rates: tuple[float, ...]
    all_losses_finite: bool
    all_inputs_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch_index": self.epoch_index,
            "batch_count": self.batch_count,
            "sample_count": self.sample_count,
            "average_losses": dict(self.average_losses),
            "minimum_total_loss": self.minimum_total_loss,
            "maximum_total_loss": self.maximum_total_loss,
            "average_batch_seconds": self.average_batch_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "learning_rates": list(self.learning_rates),
            "all_losses_finite": self.all_losses_finite,
            "all_inputs_unchanged": self.all_inputs_unchanged,
        }


@dataclass(frozen=True, slots=True)
class DetectionEvaluationEpochResult:
    """Validation/Test 전체 Prediction과 Metric 집계 결과."""

    split: str
    batch_count: int
    sample_count: int
    prediction_box_count: int
    elapsed_seconds: float
    average_batch_seconds: float
    metrics: dict[str, Any]
    predictions: tuple[dict[str, Tensor], ...]
    targets: tuple[dict[str, Tensor], ...]
    all_inputs_unchanged: bool

    def summary(self) -> dict[str, Any]:
        """Raw Tensor를 제외한 JSON 저장용 요약을 반환한다."""
        return {
            "split": self.split,
            "batch_count": self.batch_count,
            "sample_count": self.sample_count,
            "prediction_box_count": self.prediction_box_count,
            "elapsed_seconds": self.elapsed_seconds,
            "average_batch_seconds": self.average_batch_seconds,
            "all_inputs_unchanged": self.all_inputs_unchanged,
            "metrics": self.metrics,
        }


def _validate_epoch_index(epoch_index: int) -> None:
    if (
        not isinstance(epoch_index, int)
        or isinstance(epoch_index, bool)
        or epoch_index < 0
    ):
        raise ValueError("epoch_index must be a non-negative int.")


def _validate_log_interval(log_interval: int) -> None:
    if (
        not isinstance(log_interval, int)
        or isinstance(log_interval, bool)
        or log_interval <= 0
    ):
        raise ValueError("log_interval must be a positive int.")


def _notify(
    callback: ProgressCallback | None,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(dict(payload))


def _plain_target(target: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Metric에 필요한 Box·Label만 CPU 복사한다."""
    return {
        "boxes": target["boxes"].detach().cpu().clone(),
        "labels": target["labels"].detach().cpu().clone(),
    }


def run_detection_training_epoch(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: Iterable[
        tuple[Sequence[Tensor], Sequence[Mapping[str, Tensor]]]
    ],
    epoch_index: int,
    device: str | torch.device = "cpu",
    log_interval: int = 100,
    progress_callback: ProgressCallback | None = None,
) -> DetectionTrainingEpochResult:
    """전체 Train DataLoader를 한 번 순회해 실제 Optimizer Step을 수행한다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    if not isinstance(data_loader, Iterable):
        raise TypeError("data_loader must be iterable.")
    _validate_epoch_index(epoch_index)
    _validate_log_interval(log_interval)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")

    epoch_started = time.perf_counter()
    batch_count = 0
    sample_count = 0
    total_losses: list[float] = []
    batch_seconds: list[float] = []
    loss_sums: dict[str, float] = {}
    all_inputs_unchanged = True
    learning_rates: tuple[float, ...] = tuple(
        float(group["lr"]) for group in optimizer.param_groups
    )

    for batch_index, batch in enumerate(data_loader):
        if (
            not isinstance(batch, (tuple, list))
            or len(batch) != 2
        ):
            raise TypeError(
                "Every Detection DataLoader batch must be (images, targets)."
            )
        images, targets = batch
        step = run_detection_training_step(
            model=model,
            optimizer=optimizer,
            images=images,
            targets=targets,
            device=device,
        )
        batch_count += 1
        sample_count += step.batch_size
        total_losses.append(step.total_loss)
        batch_seconds.append(step.elapsed_seconds)
        learning_rates = step.learning_rates
        all_inputs_unchanged = (
            all_inputs_unchanged and step.inputs_unchanged
        )
        for name, value in step.losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value)
        loss_sums["total_loss"] = (
            loss_sums.get("total_loss", 0.0) + step.total_loss
        )

        should_log = (
            batch_count == 1
            or batch_count % log_interval == 0
        )
        if should_log:
            _notify(
                progress_callback,
                {
                    "event": "train_progress",
                    "epoch_index": epoch_index,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "sample_count": sample_count,
                    "latest_total_loss": step.total_loss,
                    "average_total_loss": (
                        loss_sums["total_loss"] / batch_count
                    ),
                    "elapsed_seconds": (
                        time.perf_counter() - epoch_started
                    ),
                },
            )

    if batch_count == 0:
        raise ValueError("Training DataLoader produced no batches.")
    if not total_losses or not all(
        math.isfinite(value) for value in total_losses
    ):
        raise FloatingPointError("Training epoch contains NaN or infinity.")
    if not all_inputs_unchanged:
        raise RuntimeError("Training epoch mutated caller inputs.")

    average_losses = {
        name: value / batch_count
        for name, value in sorted(loss_sums.items())
    }
    result = DetectionTrainingEpochResult(
        epoch_index=epoch_index,
        batch_count=batch_count,
        sample_count=sample_count,
        average_losses=average_losses,
        minimum_total_loss=min(total_losses),
        maximum_total_loss=max(total_losses),
        average_batch_seconds=sum(batch_seconds) / batch_count,
        elapsed_seconds=time.perf_counter() - epoch_started,
        learning_rates=learning_rates,
        all_losses_finite=True,
        all_inputs_unchanged=True,
    )
    _notify(
        progress_callback,
        {
            "event": "train_epoch_complete",
            **result.to_dict(),
        },
    )
    return result


def run_detection_evaluation_epoch(
    *,
    model: nn.Module,
    data_loader: Iterable[
        tuple[Sequence[Tensor], Sequence[Mapping[str, Tensor]]]
    ],
    split: str,
    num_classes: int,
    index_to_class: Mapping[int, str],
    score_threshold: float,
    iou_threshold: float,
    device: str | torch.device = "cpu",
    log_interval: int = 50,
    progress_callback: ProgressCallback | None = None,
) -> DetectionEvaluationEpochResult:
    """전체 Validation/Test DataLoader에서 Prediction과 지표를 계산한다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    if not isinstance(data_loader, Iterable):
        raise TypeError("data_loader must be iterable.")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty str.")
    if (
        not isinstance(num_classes, int)
        or isinstance(num_classes, bool)
        or num_classes < 2
    ):
        raise ValueError("num_classes must include background and foreground.")
    _validate_log_interval(log_interval)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")

    started = time.perf_counter()
    predictions: list[dict[str, Tensor]] = []
    targets_for_metrics: list[dict[str, Tensor]] = []
    batch_seconds: list[float] = []
    batch_count = 0
    sample_count = 0
    all_inputs_unchanged = True

    for batch_index, batch in enumerate(data_loader):
        if (
            not isinstance(batch, (tuple, list))
            or len(batch) != 2
        ):
            raise TypeError(
                "Every Detection DataLoader batch must be (images, targets)."
            )
        images, targets = batch
        inference = run_detection_inference_step(
            model=model,
            images=images,
            targets=targets,
            num_classes=num_classes,
            device=device,
        )
        predictions.extend(inference.predictions)
        targets_for_metrics.extend(
            _plain_target(target) for target in targets
        )
        batch_count += 1
        sample_count += inference.batch_size
        batch_seconds.append(inference.elapsed_seconds)
        all_inputs_unchanged = (
            all_inputs_unchanged and inference.inputs_unchanged
        )

        should_log = (
            batch_count == 1
            or batch_count % log_interval == 0
        )
        if should_log:
            _notify(
                progress_callback,
                {
                    "event": "evaluation_progress",
                    "split": split,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "sample_count": sample_count,
                    "prediction_box_count": sum(
                        int(prediction["boxes"].shape[0])
                        for prediction in predictions
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

    if batch_count == 0 or not predictions:
        raise ValueError("Evaluation DataLoader produced no batches.")
    if not all_inputs_unchanged:
        raise RuntimeError("Evaluation epoch mutated caller inputs.")

    metrics = calculate_detection_metrics(
        predictions=predictions,
        targets=targets_for_metrics,
        index_to_class=index_to_class,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    result = DetectionEvaluationEpochResult(
        split=split,
        batch_count=batch_count,
        sample_count=sample_count,
        prediction_box_count=sum(
            int(prediction["boxes"].shape[0])
            for prediction in predictions
        ),
        elapsed_seconds=time.perf_counter() - started,
        average_batch_seconds=sum(batch_seconds) / batch_count,
        metrics=metrics,
        predictions=tuple(predictions),
        targets=tuple(targets_for_metrics),
        all_inputs_unchanged=True,
    )
    _notify(
        progress_callback,
        {
            "event": "evaluation_complete",
            **result.summary(),
        },
    )
    return result


def build_detection_checkpoint_class_mapping(
    index_to_class: Mapping[int, str],
) -> dict[str, int]:
    """Checkpoint 계약에 맞게 Background만 ``BACKGROUND``로 고정한다."""
    if not isinstance(index_to_class, Mapping):
        raise TypeError("index_to_class must be a mapping.")
    normalized: dict[str, int] = {}
    for index, class_name in sorted(index_to_class.items()):
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Class indexes must be non-negative ints.")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("Class names must be non-empty strings.")
        key = "BACKGROUND" if index == 0 else class_name
        if key in normalized:
            raise ValueError("Checkpoint class names must be unique.")
        normalized[key] = index
    if normalized.get("BACKGROUND") != 0:
        raise ValueError("index_to_class must contain background index 0.")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Class indexes must be unique.")
    return normalized
