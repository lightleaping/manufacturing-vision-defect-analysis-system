from __future__ import annotations

import json

import pytest

from src.opencv_analysis.config import OpenCVAnalysisConfig


def test_default_config_is_valid_and_json_serializable() -> None:
    config = OpenCVAnalysisConfig()

    payload = config.to_dict()

    assert payload["gaussian_kernel_size"] == [5, 5]
    assert payload["canny_low_threshold"] < payload["canny_high_threshold"]
    json.dumps(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gaussian_kernel_size", (4, 5)),
        ("morphology_kernel_size", (3, 0)),
        ("adaptive_threshold_block_size", 10),
        ("adaptive_threshold_block_size", 1),
    ],
)
def test_odd_kernel_and_block_size_validation(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        OpenCVAnalysisConfig(**{field: value})


def test_canny_low_must_be_lower_than_high() -> None:
    with pytest.raises(ValueError, match="lower"):
        OpenCVAnalysisConfig(canny_low_threshold=150, canny_high_threshold=100)


def test_clahe_clip_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        OpenCVAnalysisConfig(clahe_clip_limit=0)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_min_contour_area_ratio_range(value: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        OpenCVAnalysisConfig(min_contour_area_ratio=value)


def test_zero_morphology_iterations_are_allowed() -> None:
    config = OpenCVAnalysisConfig(
        morphology_open_iterations=0,
        morphology_close_iterations=0,
    )

    assert config.morphology_open_iterations == 0
    assert config.morphology_close_iterations == 0
