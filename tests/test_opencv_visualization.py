from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.opencv_analysis.metrics import calculate_metrics
from src.opencv_analysis.pipeline import run_opencv_pipeline
from src.opencv_analysis.visualization import (
    CONTOUR_PANEL_TITLES,
    PIPELINE_PANEL_TITLES,
    save_contour_analysis_figure,
    save_histogram_and_metrics_figure,
    save_pipeline_overview_figure,
)


def _sample_result():
    image = np.full((80, 100, 3), 220, dtype=np.uint8)
    image[20:60, 30:70] = 40
    result = run_opencv_pipeline(image)
    return result, calculate_metrics(result)


@pytest.mark.parametrize(
    "filename, saver_name",
    [
        ("overview.png", "overview"),
        ("histogram.png", "histogram"),
        ("contours.png", "contours"),
    ],
)
def test_visualization_creates_decodable_png(
    tmp_path: Path,
    filename: str,
    saver_name: str,
) -> None:
    result, metrics = _sample_result()
    path = tmp_path / filename

    if saver_name == "overview":
        saved = save_pipeline_overview_figure(result, path, dpi=80)
    elif saver_name == "histogram":
        saved = save_histogram_and_metrics_figure(result, metrics, path, dpi=80)
    else:
        saved = save_contour_analysis_figure(result, metrics, path, dpi=80)

    assert saved == path
    assert path.is_file()
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 100
        assert image.height > 100


def test_required_panel_titles_are_fixed() -> None:
    assert PIPELINE_PANEL_TITLES == (
        "Original",
        "Grayscale",
        "CLAHE",
        "Gaussian Blur",
        "Canny Edge",
        "Adaptive Threshold",
        "Morphology",
        "Contour Candidates",
    )
    assert CONTOUR_PANEL_TITLES == (
        "Original",
        "Morphology Mask",
        "Contour Candidates",
    )


def test_visualization_requires_png_extension(tmp_path: Path) -> None:
    result, _ = _sample_result()

    with pytest.raises(ValueError, match=".png"):
        save_pipeline_overview_figure(result, tmp_path / "overview.jpg")
