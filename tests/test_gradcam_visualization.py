from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from src.explainability.gradcam_visualization import (
    GradCAMVisualizationError,
    GradCAMVisualizationRecord,
    colorize_cam,
    create_gradcam_visuals,
    load_rgb_image,
    resize_cam,
    save_gradcam_overview,
)


def _create_rgb_image(path: Path, *, width: int = 30, height: int = 20) -> Path:
    horizontal = np.linspace(0, 255, width, dtype=np.uint8)
    red = np.tile(horizontal, (height, 1))
    green = np.flip(red, axis=1)
    blue = np.full((height, width), 128, dtype=np.uint8)
    image = np.stack([red, green, blue], axis=2)
    Image.fromarray(image, mode="RGB").save(path)
    return path


def _dummy_cam() -> np.ndarray:
    cam = np.zeros((7, 7), dtype=np.float32)
    cam[2:5, 2:5] = 1.0
    return cam


def test_load_rgb_image_returns_uint8_rgb(tmp_path: Path) -> None:
    image_path = _create_rgb_image(tmp_path / "sample.png")

    image = load_rgb_image(image_path)

    assert image.shape == (20, 30, 3)
    assert image.dtype == np.uint8


def test_load_rgb_image_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GradCAMVisualizationError, match="존재하지 않습니다"):
        load_rgb_image(tmp_path / "missing.png")


def test_load_rgb_image_raises_for_corrupted_file(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "corrupted.png"
    corrupted_path.write_bytes(b"not-an-image")

    with pytest.raises(GradCAMVisualizationError, match="RGB로 변환"):
        load_rgb_image(corrupted_path)


def test_resize_cam_matches_original_size_and_range() -> None:
    resized = resize_cam(_dummy_cam(), height=20, width=30)

    assert resized.shape == (20, 30)
    assert resized.dtype == np.float32
    assert np.isfinite(resized).all()
    assert 0.0 <= float(resized.min()) <= float(resized.max()) <= 1.0


def test_colorize_cam_returns_rgb_float_heatmap() -> None:
    heatmap = colorize_cam(_dummy_cam())

    assert heatmap.shape == (7, 7, 3)
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_create_gradcam_visuals_returns_original_heatmap_and_overlay(
    tmp_path: Path,
) -> None:
    image_path = _create_rgb_image(tmp_path / "sample.png")
    original_uint8 = load_rgb_image(image_path)

    original, heatmap, overlay = create_gradcam_visuals(
        original_rgb=original_uint8,
        cam=_dummy_cam(),
        alpha=0.4,
    )

    assert original.shape == heatmap.shape == overlay.shape == (20, 30, 3)
    assert original.dtype == heatmap.dtype == overlay.dtype == np.float32
    assert np.isfinite(overlay).all()
    assert 0.0 <= float(overlay.min()) <= float(overlay.max()) <= 1.0
    assert not np.array_equal(original, overlay)


def test_save_gradcam_overview_creates_nonempty_png_atomically(
    tmp_path: Path,
) -> None:
    image_path = _create_rgb_image(tmp_path / "sample.png")
    output_path = tmp_path / "overview.png"
    record = GradCAMVisualizationRecord(
        sample_index=1,
        image_path=str(image_path),
        selection_type="HIGH_CONFIDENCE_TRUE_POSITIVE",
        ground_truth_class_name="DEFECT",
        prediction_class_name="DEFECT",
        defect_probability=0.99,
        target_class="DEFECT",
        target_layer_name="features.0",
        cam=_dummy_cam(),
    )
    figure_numbers_before = tuple(plt.get_fignums())

    saved_path = save_gradcam_overview(
        records=[record],
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert list(tmp_path.glob(".overview_*.png")) == []
    assert tuple(plt.get_fignums()) == figure_numbers_before


def test_save_gradcam_overview_rejects_empty_records(tmp_path: Path) -> None:
    with pytest.raises(GradCAMVisualizationError, match="표본이 없습니다"):
        save_gradcam_overview(records=[], output_path=tmp_path / "overview.png")
