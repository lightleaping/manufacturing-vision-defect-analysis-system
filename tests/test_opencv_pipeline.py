from __future__ import annotations

import numpy as np

from src.opencv_analysis.config import OpenCVAnalysisConfig
from src.opencv_analysis.pipeline import run_opencv_pipeline


def _rectangle_image() -> np.ndarray:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[20:45, 18:48] = 0
    return image


def test_pipeline_stage_shapes_dtypes_and_ranges() -> None:
    image = _rectangle_image()

    result = run_opencv_pipeline(image)

    assert result.original_bgr.shape == (64, 64, 3)
    for stage in (
        result.grayscale,
        result.clahe,
        result.blurred,
        result.edges,
        result.adaptive_threshold,
        result.morphology,
    ):
        assert stage.shape == (64, 64)
        assert stage.dtype == np.uint8
        assert int(stage.min()) >= 0
        assert int(stage.max()) <= 255
    assert result.histogram.shape == (256,)
    assert result.histogram.dtype == np.float64
    assert result.contour_overlay_bgr.shape == (64, 64, 3)


def test_pipeline_does_not_modify_input() -> None:
    image = _rectangle_image()
    original = image.copy()

    run_opencv_pipeline(image)

    assert np.array_equal(image, original)


def test_pipeline_is_deterministic() -> None:
    image = _rectangle_image()
    config = OpenCVAnalysisConfig()

    first = run_opencv_pipeline(image, config)
    second = run_opencv_pipeline(image, config)

    for first_stage, second_stage in (
        (first.grayscale, second.grayscale),
        (first.histogram, second.histogram),
        (first.clahe, second.clahe),
        (first.blurred, second.blurred),
        (first.edges, second.edges),
        (first.adaptive_threshold, second.adaptive_threshold),
        (first.morphology, second.morphology),
        (first.contour_overlay_bgr, second.contour_overlay_bgr),
    ):
        assert np.array_equal(first_stage, second_stage)
    assert first.contour_areas == second.contour_areas
    assert first.otsu_threshold == second.otsu_threshold


def test_constant_black_and_white_images_have_no_canny_edges() -> None:
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)

    black_result = run_opencv_pipeline(black)
    white_result = run_opencv_pipeline(white)

    assert np.count_nonzero(black_result.edges) == 0
    assert np.count_nonzero(white_result.edges) == 0


def test_rectangle_creates_edges_and_contour_candidates() -> None:
    result = run_opencv_pipeline(
        _rectangle_image(),
        OpenCVAnalysisConfig(min_contour_area_ratio=0.001),
    )

    assert np.count_nonzero(result.edges) > 0
    assert np.count_nonzero(result.morphology) > 0
    assert len(result.contours) >= 1
    assert result.contour_areas[0] > 0


def test_noise_filter_removes_tiny_contours() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[10, 10] = 0
    image[30:50, 30:50] = 0

    result = run_opencv_pipeline(
        image,
        OpenCVAnalysisConfig(
            morphology_open_iterations=0,
            morphology_close_iterations=0,
            min_contour_area_ratio=0.01,
        ),
    )

    assert all(area >= 64 * 64 * 0.01 for area in result.contour_areas)


def test_pipeline_accepts_grayscale_input_and_normalizes_original_to_bgr() -> None:
    grayscale = np.tile(np.arange(64, dtype=np.uint8), (64, 1))

    result = run_opencv_pipeline(grayscale)

    assert result.grayscale.shape == (64, 64)
    assert result.original_bgr.shape == (64, 64, 3)
    assert np.array_equal(result.original_bgr[..., 0], grayscale)
