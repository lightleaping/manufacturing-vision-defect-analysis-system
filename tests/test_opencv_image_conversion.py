from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.opencv_analysis.image_conversion import (
    bgr_to_rgb,
    pillow_to_bgr,
    pillow_to_grayscale,
    rgb_to_bgr,
    to_grayscale,
)


def test_pillow_rgb_to_opencv_bgr_channel_order() -> None:
    image = Image.new("RGB", (1, 1), color=(10, 20, 30))

    bgr = pillow_to_bgr(image)

    assert bgr.shape == (1, 1, 3)
    assert bgr.dtype == np.uint8
    assert bgr[0, 0].tolist() == [30, 20, 10]


def test_pillow_l_to_grayscale() -> None:
    image = Image.new("L", (2, 3), color=77)

    grayscale = pillow_to_grayscale(image)

    assert grayscale.shape == (3, 2)
    assert grayscale.dtype == np.uint8
    assert np.all(grayscale == 77)


def test_bgr_rgb_round_trip() -> None:
    bgr = np.array([[[3, 2, 1], [30, 20, 10]]], dtype=np.uint8)

    rgb = bgr_to_rgb(bgr)
    restored = rgb_to_bgr(rgb)

    assert rgb[0, 0].tolist() == [1, 2, 3]
    assert np.array_equal(restored, bgr)


def test_to_grayscale_returns_copy_for_grayscale_input() -> None:
    source = np.full((4, 5), 80, dtype=np.uint8)

    result = to_grayscale(source)
    result[0, 0] = 1

    assert source[0, 0] == 80


@pytest.mark.parametrize(
    "invalid",
    [
        np.empty((0, 0), dtype=np.uint8),
        np.zeros((3, 3, 4), dtype=np.uint8),
        np.zeros((3, 3, 3), dtype=np.float32),
    ],
)
def test_invalid_array_rejected(invalid: np.ndarray) -> None:
    expected_exception = TypeError if invalid.dtype != np.uint8 else ValueError
    with pytest.raises(expected_exception):
        to_grayscale(invalid)


def test_non_array_rejected() -> None:
    with pytest.raises(TypeError):
        bgr_to_rgb([[1, 2, 3]])  # type: ignore[arg-type]
