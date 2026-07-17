from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import SGD

from src.detection.training_resume import (
    extract_best_metric_value,
    is_better_metric,
    set_optimizer_learning_rate,
    validate_resume_epoch_range,
)


def test_set_optimizer_learning_rate_updates_every_group() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    optimizer = SGD(
        [
            {"params": model[0].parameters(), "lr": 0.1},
            {"params": model[1].parameters(), "lr": 0.01},
        ],
        momentum=0.9,
    )

    values = set_optimizer_learning_rate(
        optimizer,
        learning_rate=0.001,
    )

    assert values == pytest.approx((0.001, 0.001))
    assert all(group["lr"] == pytest.approx(0.001) for group in optimizer.param_groups)


def test_set_optimizer_learning_rate_rejects_non_positive_value() -> None:
    model = nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="positive finite"):
        set_optimizer_learning_rate(optimizer, learning_rate=0.0)


def test_extract_best_metric_value_reads_overall_map() -> None:
    summary = {"metrics": {"overall": {"map_50": 0.42}}}
    assert extract_best_metric_value(
        metric_name="map_50",
        validation_summary=summary,
    ) == pytest.approx(0.42)


def test_extract_best_metric_value_rejects_none() -> None:
    summary = {"metrics": {"overall": {"map_50": None}}}
    with pytest.raises(ValueError, match="is None"):
        extract_best_metric_value(
            metric_name="map_50",
            validation_summary=summary,
        )


def test_is_better_metric_respects_direction() -> None:
    assert is_better_metric(
        metric_name="map_50",
        candidate=0.5,
        current_best=0.4,
    ) is True
    assert is_better_metric(
        metric_name="validation_loss",
        candidate=0.5,
        current_best=0.7,
    ) is True
    assert is_better_metric(
        metric_name="f1",
        candidate=0.4,
        current_best=0.4,
    ) is False


def test_validate_resume_epoch_range_starts_after_checkpoint() -> None:
    assert list(
        validate_resume_epoch_range(
            checkpoint_epoch=0,
            total_epochs=3,
        )
    ) == [1, 2]
    with pytest.raises(ValueError, match="already reached"):
        validate_resume_epoch_range(
            checkpoint_epoch=2,
            total_epochs=3,
        )
