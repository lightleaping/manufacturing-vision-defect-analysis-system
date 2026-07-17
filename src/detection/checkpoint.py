"""Day 12 Detection Checkpoint 저장·복원.

[기존 코드 참고]
Classification 학습 코드의 원자적 저장 원칙을 Detection에도 적용한다.

[신규 구현]
- ``latest``와 ``best`` 두 파일만 유지하는 저장 구조
- Model·Optimizer·Scheduler·Config·Class Mapping·History 통합 저장
- CPU ``map_location`` 복원
- 필수 Key와 Class Mapping 일치 검증
- 임시 파일 저장 후 ``Path.replace``를 사용하는 원자적 교체
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchvision


REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "training_config",
        "class_mapping",
        "best_metric",
        "history",
        "torch_version",
        "torchvision_version",
    }
)


@dataclass(frozen=True, slots=True)
class DetectionCheckpointState:
    """Checkpoint 복원 뒤 Trainer가 이어서 사용할 Metadata."""

    epoch: int
    next_epoch: int
    best_metric: float
    training_config: dict[str, Any]
    class_mapping: dict[str, int]
    history: list[dict[str, Any]]


def _validate_checkpoint_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path.")
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Checkpoint path must use .pt or .pth extension.")
    return path


def _plain_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _validate_class_mapping(
    class_mapping: Mapping[str, int],
) -> dict[str, int]:
    mapping = dict(class_mapping)
    if not mapping:
        raise ValueError("class_mapping must not be empty.")
    if any(not isinstance(name, str) or not name for name in mapping):
        raise ValueError("Every class_mapping key must be a non-empty str.")
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in mapping.values()
    ):
        raise ValueError("Every class_mapping value must be a non-negative int.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("class_mapping indexes must be unique.")
    if mapping.get("BACKGROUND") != 0:
        raise ValueError("class_mapping must reserve BACKGROUND=0.")
    return mapping


def build_detection_checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    training_config: Mapping[str, Any],
    class_mapping: Mapping[str, int],
    best_metric: float,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """학습 상태를 재개 가능한 단일 Payload로 구성한다."""
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a non-negative int.")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    if scheduler is not None and not isinstance(scheduler, LRScheduler):
        raise TypeError("scheduler must be LRScheduler or None.")

    config = _plain_mapping(training_config, name="training_config")
    mapping = _validate_class_mapping(class_mapping)

    if (
        not isinstance(best_metric, (int, float))
        or isinstance(best_metric, bool)
        or not torch.isfinite(torch.tensor(float(best_metric)))
    ):
        raise ValueError("best_metric must be a finite number.")

    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence of mappings.")
    normalized_history: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise TypeError(f"history[{index}] must be a mapping.")
        normalized_history.append(dict(item))

    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "training_config": config,
        "class_mapping": mapping,
        "best_metric": float(best_metric),
        "history": normalized_history,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
    }


def save_detection_checkpoint(
    *,
    payload: Mapping[str, Any],
    latest_path: Path,
    best_path: Path | None = None,
    is_best: bool = False,
) -> tuple[Path, Path | None]:
    """Checkpoint를 임시 파일에 쓴 뒤 latest·best 경로로 교체한다."""
    latest = _validate_checkpoint_path(latest_path)
    best = None if best_path is None else _validate_checkpoint_path(best_path)
    if not isinstance(is_best, bool):
        raise TypeError("is_best must be bool.")
    if is_best and best is None:
        raise ValueError("best_path is required when is_best=True.")
    if best is not None and latest.resolve() == best.resolve():
        raise ValueError("latest_path and best_path must differ.")

    normalized_payload = dict(payload)
    missing = REQUIRED_CHECKPOINT_KEYS - set(normalized_payload)
    if missing:
        raise KeyError(f"Checkpoint payload is missing keys: {sorted(missing)}.")

    latest.parent.mkdir(parents=True, exist_ok=True)
    latest_temp = latest.with_name(f".{latest.name}.tmp")
    best_temp = (
        None if best is None else best.with_name(f".{best.name}.tmp")
    )

    try:
        torch.save(normalized_payload, latest_temp)
        latest_temp.replace(latest)

        saved_best: Path | None = None
        if is_best and best is not None and best_temp is not None:
            best.parent.mkdir(parents=True, exist_ok=True)
            torch.save(normalized_payload, best_temp)
            best_temp.replace(best)
            saved_best = best
        return latest, saved_best
    except Exception as error:
        raise OSError(
            "Detection checkpoint save failed. "
            f"Latest destination: {latest}."
        ) from error
    finally:
        for temporary_path in (latest_temp, best_temp):
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def load_detection_checkpoint_payload(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """신뢰할 수 있는 프로젝트 Checkpoint를 CPU 기준으로 읽고 Schema를 검증한다."""
    checkpoint_path = _validate_checkpoint_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Detection checkpoint does not exist: {checkpoint_path}."
        )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=torch.device(map_location),
            weights_only=True,
        )
    except TypeError:
        # 오래된 PyTorch와의 호환용 분기다.
        payload = torch.load(
            checkpoint_path,
            map_location=torch.device(map_location),
        )
    except Exception as error:
        raise ValueError(
            f"Could not load detection checkpoint: {checkpoint_path}."
        ) from error

    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint top-level value must be a mapping.")
    normalized = dict(payload)
    missing = REQUIRED_CHECKPOINT_KEYS - set(normalized)
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {sorted(missing)}.")

    epoch = normalized["epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("Checkpoint epoch must be a non-negative int.")
    normalized["class_mapping"] = _validate_class_mapping(
        normalized["class_mapping"]
    )
    if not isinstance(normalized["history"], list):
        raise TypeError("Checkpoint history must be a list.")
    return normalized


def restore_detection_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    expected_class_mapping: Mapping[str, int] | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> DetectionCheckpointState:
    """Model과 선택적인 Optimizer·Scheduler 상태를 복원한다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    if optimizer is not None and not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be Optimizer or None.")
    if scheduler is not None and not isinstance(scheduler, LRScheduler):
        raise TypeError("scheduler must be LRScheduler or None.")
    if not isinstance(strict, bool):
        raise TypeError("strict must be bool.")

    payload = load_detection_checkpoint_payload(
        path,
        map_location=map_location,
    )
    saved_mapping = _validate_class_mapping(payload["class_mapping"])
    if expected_class_mapping is not None:
        expected = _validate_class_mapping(expected_class_mapping)
        if saved_mapping != expected:
            raise ValueError(
                "Checkpoint class_mapping does not match the current project."
            )

    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    saved_scheduler_state = payload["scheduler_state_dict"]
    if scheduler is not None:
        if saved_scheduler_state is None:
            raise ValueError(
                "Current scheduler was provided but checkpoint has no scheduler state."
            )
        scheduler.load_state_dict(saved_scheduler_state)

    best_metric = float(payload["best_metric"])
    if not torch.isfinite(torch.tensor(best_metric)):
        raise ValueError("Checkpoint best_metric must be finite.")

    history = payload["history"]
    if any(not isinstance(item, Mapping) for item in history):
        raise TypeError("Every checkpoint history item must be a mapping.")

    epoch = int(payload["epoch"])
    return DetectionCheckpointState(
        epoch=epoch,
        next_epoch=epoch + 1,
        best_metric=best_metric,
        training_config=dict(payload["training_config"]),
        class_mapping=saved_mapping,
        history=[dict(item) for item in history],
    )
