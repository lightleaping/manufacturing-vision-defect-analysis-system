"""Day 12 Detection Optimizer·Scheduler와 Backbone Freeze 정책.

[기존 코드 참고]
Torchvision Detection Fine-tuning 예제의 SGD 기본 조합을 CPU 환경에 맞게
명시적인 Config 기반 Factory로 분리한다.

[신규 구현]
- Backbone만 선택적으로 Freeze·Unfreeze한다.
- Optimizer에는 전체 Parameter를 등록해 이후 Unfreeze 시 재생성하지 않는다.
- SGD와 AdamW, StepLR와 Scheduler 없음만 허용한다.
- 총·학습 가능 Parameter 수를 Artifact에 기록할 수 있게 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler, StepLR

from src.detection.training_config import DetectionTrainingConfig


@dataclass(frozen=True, slots=True)
class DetectionOptimizationResult:
    """생성된 Optimizer·Scheduler와 재현 가능한 Metadata."""

    optimizer: Optimizer
    scheduler: LRScheduler | None
    metadata: dict[str, Any]


def count_detection_parameters(model: nn.Module) -> dict[str, int]:
    """모델 전체·현재 학습 가능 Parameter 수를 센다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }


def set_detection_backbone_trainable(
    model: nn.Module,
    *,
    trainable: bool,
) -> dict[str, Any]:
    """Faster R-CNN Backbone의 ``requires_grad``만 안전하게 변경한다.

    RPN과 ROI Head는 그대로 학습 가능 상태를 유지한다. 초기 Head 적응 뒤
    Backbone을 여는 단계 학습을 위해 사용한다.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    if not isinstance(trainable, bool):
        raise TypeError("trainable must be bool.")

    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise TypeError("model must expose a torch.nn.Module backbone.")

    changed_parameter_tensors = 0
    for parameter in backbone.parameters():
        if parameter.requires_grad != trainable:
            changed_parameter_tensors += 1
        parameter.requires_grad_(trainable)

    counts = count_detection_parameters(model)
    return {
        "backbone_trainable": trainable,
        "changed_parameter_tensors": changed_parameter_tensors,
        **counts,
    }


def create_detection_optimizer(
    *,
    model: nn.Module,
    config: DetectionTrainingConfig,
) -> Optimizer:
    """전체 Parameter를 등록하는 Detection Optimizer를 만든다.

    Freeze된 Parameter도 Optimizer에 등록한다. 현재는 Gradient가 없어 갱신되지
    않지만 이후 Unfreeze하면 같은 Optimizer가 바로 해당 Parameter를 갱신한다.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    if not isinstance(config, DetectionTrainingConfig):
        raise TypeError("config must be DetectionTrainingConfig.")

    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("model must contain parameters.")

    if config.optimizer_name == "sgd":
        return SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "adamw":
        return AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name!r}.")


def create_detection_scheduler(
    *,
    optimizer: Optimizer,
    config: DetectionTrainingConfig,
) -> LRScheduler | None:
    """단순한 StepLR 또는 Scheduler 없음 정책을 적용한다."""
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    if not isinstance(config, DetectionTrainingConfig):
        raise TypeError("config must be DetectionTrainingConfig.")

    if config.scheduler_name == "none":
        return None
    if config.scheduler_name == "step_lr":
        return StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler_name!r}.")


def build_detection_optimization(
    *,
    model: nn.Module,
    config: DetectionTrainingConfig,
) -> DetectionOptimizationResult:
    """Optimizer·Scheduler와 설정 Metadata를 한 번에 만든다."""
    optimizer = create_detection_optimizer(model=model, config=config)
    scheduler = create_detection_scheduler(
        optimizer=optimizer,
        config=config,
    )
    counts = count_detection_parameters(model)
    return DetectionOptimizationResult(
        optimizer=optimizer,
        scheduler=scheduler,
        metadata={
            **counts,
            "optimizer_name": config.optimizer_name,
            "learning_rate": config.learning_rate,
            "momentum": (
                config.momentum if config.optimizer_name == "sgd" else None
            ),
            "weight_decay": config.weight_decay,
            "scheduler_name": config.scheduler_name,
            "scheduler_step_size": (
                config.scheduler_step_size
                if config.scheduler_name == "step_lr"
                else None
            ),
            "scheduler_gamma": (
                config.scheduler_gamma
                if config.scheduler_name == "step_lr"
                else None
            ),
            "optimizer_contains_all_parameters": True,
        },
    )
