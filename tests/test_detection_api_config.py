from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.api.detection_config import (
    DetectionApiSettings,
    resolve_score_threshold,
)


def test_default_detection_settings_match_day12_policy() -> None:
    settings = DetectionApiSettings()

    assert settings.architecture == "fasterrcnn_mobilenet_v3_large_320_fpn"
    assert settings.device == "cpu"
    assert settings.default_score_threshold == pytest.approx(0.5)
    assert settings.iou_threshold == pytest.approx(0.5)
    assert settings.checkpoint_path.as_posix().endswith(
        "models/detection/day12_detection_best.pt"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0.5),
        ("", 0.5),
        ("0.05", 0.05),
        (0.5, 0.5),
        (0.95, 0.95),
    ],
)
def test_score_threshold_is_resolved(value, expected: float) -> None:
    assert resolve_score_threshold(
        value,
        settings=DetectionApiSettings(),
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    ["invalid", "nan", "inf", "-inf", 0.049, 0.951, True],
)
def test_invalid_score_threshold_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        resolve_score_threshold(value, settings=DetectionApiSettings())


def test_non_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="CPU"):
        DetectionApiSettings(device="cuda")


def test_checkpoint_extension_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="extension"):
        DetectionApiSettings(checkpoint_path=tmp_path / "model.bin")
