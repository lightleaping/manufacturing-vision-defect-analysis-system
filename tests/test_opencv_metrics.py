from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.opencv_analysis.config import OpenCVAnalysisConfig
from src.opencv_analysis.metrics import calculate_metrics
from src.opencv_analysis.pipeline import run_opencv_pipeline


def _metrics_for_constant(value: int):
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    return calculate_metrics(run_opencv_pipeline(image))


def test_black_image_brightness_and_contrast() -> None:
    metrics = _metrics_for_constant(0)

    assert metrics.mean_brightness == 0.0
    assert metrics.brightness_standard_deviation == 0.0
    assert metrics.edge_pixel_ratio == 0.0
    assert metrics.histogram_peak == 0


def test_white_image_brightness_and_contrast() -> None:
    metrics = _metrics_for_constant(255)

    assert metrics.mean_brightness == 255.0
    assert metrics.brightness_standard_deviation == 0.0
    assert metrics.edge_pixel_ratio == 0.0
    assert metrics.histogram_peak == 255


def test_rectangle_metrics_have_expected_ranges() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[25:75, 25:75] = 0

    result = run_opencv_pipeline(
        image,
        OpenCVAnalysisConfig(min_contour_area_ratio=0.001),
    )
    metrics = calculate_metrics(result)

    assert metrics.width == 100
    assert metrics.height == 100
    assert metrics.channels == 3
    assert metrics.contour_count >= 1
    assert 0 < metrics.largest_contour_area_ratio <= 1
    assert 0 < metrics.average_contour_area_ratio <= 1
    assert 0 < metrics.edge_pixel_ratio < 1
    assert 0 < metrics.threshold_foreground_ratio < 1


def test_metrics_are_json_serializable_and_finite() -> None:
    gradient = np.tile(np.arange(128, dtype=np.uint8), (64, 1))
    result = run_opencv_pipeline(gradient)

    payload = calculate_metrics(result).to_dict()

    json.dumps(payload)
    for value in payload.values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_invalid_result_rejected() -> None:
    with pytest.raises(TypeError):
        calculate_metrics(object())  # type: ignore[arg-type]
