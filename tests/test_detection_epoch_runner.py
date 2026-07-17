from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.optim import SGD

from src.detection.epoch_runner import (
    build_detection_checkpoint_class_mapping,
    run_detection_evaluation_epoch,
    run_detection_training_epoch,
)


class TinyDetectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            base = (self.weight - 0.25).pow(2)
            return {
                "loss_classifier": base,
                "loss_box_reg": base * 0.5,
                "loss_objectness": base * 0.25,
                "loss_rpn_box_reg": base * 0.125,
            }
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9]),
            }
            for _ in images
        ]


def _batch():
    image = torch.zeros((3, 16, 16), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    return ([image], [target])


def test_training_epoch_aggregates_two_batches() -> None:
    model = TinyDetectionModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    events = []
    result = run_detection_training_epoch(
        model=model,
        optimizer=optimizer,
        data_loader=[_batch(), _batch()],
        epoch_index=0,
        log_interval=1,
        progress_callback=events.append,
    )

    assert result.batch_count == 2
    assert result.sample_count == 2
    assert result.all_losses_finite is True
    assert result.all_inputs_unchanged is True
    assert math.isfinite(result.average_losses["total_loss"])
    assert any(event["event"] == "train_epoch_complete" for event in events)


def test_training_epoch_rejects_empty_loader() -> None:
    model = TinyDetectionModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="no batches"):
        run_detection_training_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=[],
            epoch_index=0,
        )


def test_evaluation_epoch_returns_perfect_metrics() -> None:
    model = TinyDetectionModel()
    result = run_detection_evaluation_epoch(
        model=model,
        data_loader=[_batch(), _batch()],
        split="validation",
        num_classes=2,
        index_to_class={0: "background", 1: "defect"},
        score_threshold=0.5,
        iou_threshold=0.5,
        log_interval=1,
    )

    assert result.sample_count == 2
    assert result.prediction_box_count == 2
    assert result.metrics["overall"]["precision"] == pytest.approx(1.0)
    assert result.metrics["overall"]["recall"] == pytest.approx(1.0)
    assert result.metrics["overall"]["map_50"] == pytest.approx(1.0)


def test_checkpoint_mapping_normalizes_only_background() -> None:
    assert build_detection_checkpoint_class_mapping(
        {0: "background", 1: "crazing", 2: "scratches"}
    ) == {
        "BACKGROUND": 0,
        "crazing": 1,
        "scratches": 2,
    }


def test_checkpoint_mapping_requires_background_index_zero() -> None:
    with pytest.raises(ValueError, match="background index 0"):
        build_detection_checkpoint_class_mapping({1: "crazing"})
