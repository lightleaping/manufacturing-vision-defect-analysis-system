"""Detection Optimizer·Scheduler와 Freeze 정책 테스트."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR

from src.detection.optimization import (
    build_detection_optimization,
    count_detection_parameters,
    create_detection_optimizer,
    create_detection_scheduler,
    set_detection_backbone_trainable,
)
from src.detection.training_config import DetectionTrainingConfig


class TinyDetectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)


def test_freeze_and_unfreeze_only_backbone() -> None:
    model = TinyDetectionModel()
    frozen = set_detection_backbone_trainable(model, trainable=False)
    assert frozen["backbone_trainable"] is False
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())

    unfrozen = set_detection_backbone_trainable(model, trainable=True)
    assert unfrozen["backbone_trainable"] is True
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_parameter_counts_match_numel() -> None:
    model = TinyDetectionModel()
    expected = sum(parameter.numel() for parameter in model.parameters())
    assert count_detection_parameters(model)["total_parameters"] == expected


def test_build_sgd_and_step_lr_registers_all_parameters() -> None:
    model = TinyDetectionModel()
    set_detection_backbone_trainable(model, trainable=False)
    config = DetectionTrainingConfig(epochs=1, freeze_backbone_epochs=1)
    result = build_detection_optimization(model=model, config=config)
    assert isinstance(result.optimizer, SGD)
    assert isinstance(result.scheduler, StepLR)
    registered = sum(
        parameter.numel()
        for group in result.optimizer.param_groups
        for parameter in group["params"]
    )
    assert registered == sum(parameter.numel() for parameter in model.parameters())


def test_adamw_and_no_scheduler() -> None:
    model = TinyDetectionModel()
    config = DetectionTrainingConfig(
        epochs=1,
        freeze_backbone_epochs=0,
        optimizer_name="adamw",
        scheduler_name="none",
    )
    optimizer = create_detection_optimizer(model=model, config=config)
    scheduler = create_detection_scheduler(optimizer=optimizer, config=config)
    assert isinstance(optimizer, AdamW)
    assert scheduler is None


def test_freeze_rejects_model_without_backbone() -> None:
    with pytest.raises(TypeError, match="backbone"):
        set_detection_backbone_trainable(nn.Linear(2, 1), trainable=False)
