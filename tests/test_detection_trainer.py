"""합성 Detection Model로 실제 Backward·Inference 방어 로직을 검증한다."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import SGD

from src.detection.trainer import (
    run_detection_inference_step,
    run_detection_training_step,
    run_tiny_overfit_diagnostic,
)


class FakeDetectionModel(nn.Module):
    def __init__(self, *, nan_loss: bool = False) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.value = nn.Parameter(torch.tensor(1.0))
        self.nan_loss = nan_loss

    def forward(self, images, targets=None):
        if self.training:
            assert targets is not None
            base = self.value.square()
            if self.nan_loss:
                base = base * torch.tensor(float("nan"))
            return {
                "loss_classifier": base,
                "loss_box_reg": base * 0.5,
                "loss_objectness": base * 0.25,
                "loss_rpn_box_reg": base * 0.125,
            }
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
            for _ in images
        ]


def _batch():
    images = [torch.zeros((3, 16, 16), dtype=torch.float32)]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor([0], dtype=torch.int64),
            "area": torch.tensor([64.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
        }
    ]
    return images, targets


def test_training_step_is_finite_and_changes_parameter() -> None:
    model = FakeDetectionModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    images, targets = _batch()
    before = model.value.detach().clone()
    result = run_detection_training_step(
        model=model,
        optimizer=optimizer,
        images=images,
        targets=targets,
    )
    assert result.total_loss > 0.0
    assert result.gradient_parameter_count > 0
    assert result.inputs_unchanged is True
    assert not torch.equal(before, model.value.detach())


def test_inference_does_not_change_parameter_or_mode() -> None:
    model = FakeDetectionModel()
    model.train()
    images, targets = _batch()
    before = model.value.detach().clone()
    result = run_detection_inference_step(
        model=model,
        images=images,
        targets=targets,
        num_classes=2,
    )
    assert model.training is True
    assert torch.equal(before, model.value.detach())
    assert result.predictions[0]["scores"].item() == pytest.approx(0.9)


def test_nan_loss_is_rejected_before_optimizer_step() -> None:
    model = FakeDetectionModel(nan_loss=True)
    optimizer = SGD(model.parameters(), lr=0.1)
    images, targets = _batch()
    before = model.value.detach().clone()
    with pytest.raises(FloatingPointError, match="NaN"):
        run_detection_training_step(
            model=model,
            optimizer=optimizer,
            images=images,
            targets=targets,
        )
    assert torch.equal(before, model.value.detach())


def test_empty_batch_is_rejected() -> None:
    model = FakeDetectionModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="empty"):
        run_detection_training_step(
            model=model,
            optimizer=optimizer,
            images=[],
            targets=[],
        )


def test_tiny_overfit_records_loss_decrease() -> None:
    model = FakeDetectionModel()
    optimizer = SGD(model.parameters(), lr=0.05)
    images, targets = _batch()
    result = run_tiny_overfit_diagnostic(
        model=model,
        optimizer=optimizer,
        images=images,
        targets=targets,
        steps=3,
    )
    assert result["all_losses_finite"] is True
    assert result["loss_decrease_observed"] is True
    assert result["final_total_loss"] < result["initial_total_loss"]
