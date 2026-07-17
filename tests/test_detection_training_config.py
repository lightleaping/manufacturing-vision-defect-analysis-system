"""Day 12 DetectionTrainingConfig 테스트."""

from __future__ import annotations

import json
import math

import pytest

from src.detection.training_config import DetectionTrainingConfig


def test_default_config_is_cpu_safe_and_serializable() -> None:
    config = DetectionTrainingConfig()
    payload = config.to_dict()

    assert payload["batch_size"] == 1
    assert payload["epochs"] == 3
    assert payload["optimizer_name"] == "sgd"
    assert payload["scheduler_name"] == "step_lr"
    assert payload["score_threshold"] == 0.5
    assert payload["iou_threshold"] == 0.5
    assert payload["freeze_backbone_epochs"] == 1
    assert payload["duplicate_box_policy"] == "preserve"
    assert payload["num_workers"] == 0
    assert payload["pin_memory"] is False
    assert payload["drop_last"] is False
    json.dumps(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("batch_size", 0),
        ("epochs", 0),
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("score_threshold", 1.1),
        ("iou_threshold", 0.0),
        ("horizontal_flip_probability", -0.1),
        ("torch_num_threads", 0),
    ],
)
def test_invalid_numeric_config_is_rejected(
    field_name: str,
    value: object,
) -> None:
    kwargs = {field_name: value}
    with pytest.raises((TypeError, ValueError)):
        DetectionTrainingConfig(**kwargs)


def test_freeze_epochs_cannot_exceed_total_epochs() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        DetectionTrainingConfig(epochs=2, freeze_backbone_epochs=3)


def test_verified_cpu_loader_policy_is_enforced() -> None:
    with pytest.raises(ValueError, match="pin_memory=False"):
        DetectionTrainingConfig(pin_memory=True)
    with pytest.raises(ValueError, match="drop_last=False"):
        DetectionTrainingConfig(drop_last=True)
    with pytest.raises(ValueError, match="persistent_workers"):
        DetectionTrainingConfig(
            num_workers=0,
            persistent_workers=True,
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_non_finite_values_are_rejected(bad_value: float) -> None:
    assert not math.isfinite(bad_value)
    with pytest.raises(ValueError, match="finite"):
        DetectionTrainingConfig(learning_rate=bad_value)
