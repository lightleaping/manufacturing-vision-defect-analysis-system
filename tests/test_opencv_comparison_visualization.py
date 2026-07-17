from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.opencv_analysis.comparison_visualization import (
    DAY10_CONTOUR_COLUMN_TITLES,
    DAY10_PIPELINE_COLUMN_TITLES,
    save_day10_contour_analysis,
    save_day10_histogram_and_metrics,
    save_day10_pipeline_overview,
)
from src.opencv_analysis.sample_analysis import (
    ImageSampleSpec,
    analyze_image_samples,
)


def _make_samples(tmp_path: Path):
    specs = []
    for index, suffix in enumerate((".png", ".jpg", ".jpeg"), start=1):
        relative = f"data/sample_{index}{suffix}"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)

        image = np.full((32, 40, 3), 30 * index, dtype=np.uint8)
        image[6:26, 10:30] = (220, 180, 50)
        Image.fromarray(image, mode="RGB").save(path)

        specs.append(
            ImageSampleSpec(
                sample_id=f"sample_{index}",
                dataset_name="Synthetic",
                semantic_role=f"Role {index}",
                class_name=f"Class {index}",
                relative_path=relative,
            )
        )
    return analyze_image_samples(tmp_path, specs)


def _assert_valid_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        image.load()
        assert image.format == "PNG"
        assert image.width > 0
        assert image.height > 0


def test_day10_panel_title_counts() -> None:
    assert len(DAY10_PIPELINE_COLUMN_TITLES) == 8
    assert len(DAY10_CONTOUR_COLUMN_TITLES) == 3


@pytest.mark.parametrize(
    "writer,filename",
    [
        (save_day10_pipeline_overview, "pipeline.png"),
        (save_day10_histogram_and_metrics, "histogram.png"),
        (save_day10_contour_analysis, "contour.png"),
    ],
)
def test_comparison_figure_writers_create_decodable_png(
    tmp_path: Path,
    writer,
    filename: str,
) -> None:
    samples = _make_samples(tmp_path)
    output = tmp_path / "figures" / filename

    returned = writer(samples, output, dpi=60)

    assert returned == output
    _assert_valid_png(output)


@pytest.mark.parametrize(
    "writer",
    [
        save_day10_pipeline_overview,
        save_day10_histogram_and_metrics,
        save_day10_contour_analysis,
    ],
)
def test_comparison_figure_writers_reject_empty_samples(
    tmp_path: Path,
    writer,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        writer((), tmp_path / "figure.png")


def test_comparison_figure_rejects_non_png_path(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path)
    with pytest.raises(ValueError, match=".png"):
        save_day10_pipeline_overview(samples, tmp_path / "figure.jpg")


def test_comparison_figure_rejects_non_positive_dpi(tmp_path: Path) -> None:
    samples = _make_samples(tmp_path)
    with pytest.raises(ValueError, match="greater than 0"):
        save_day10_contour_analysis(
            samples,
            tmp_path / "figure.png",
            dpi=0,
        )
