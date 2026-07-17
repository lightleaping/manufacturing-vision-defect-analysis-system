"""Pillow, RGB, BGR, Grayscale 사이의 안전한 변환 함수."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _validate_uint8_array(
    image: np.ndarray,
    *,
    name: str,
    allowed_shapes: tuple[str, ...],
) -> None:
    """OpenCV 처리 전에 배열의 타입·크기·채널을 검증한다."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} dtype must be uint8")

    shape_kind: str
    if image.ndim == 2:
        shape_kind = "grayscale"
    elif image.ndim == 3 and image.shape[2] == 3:
        shape_kind = "three_channel"
    else:
        shape_kind = "unsupported"

    if shape_kind not in allowed_shapes:
        raise ValueError(
            f"{name} must have one of these shapes: {', '.join(allowed_shapes)}"
        )


def pillow_to_bgr(image: Image.Image) -> np.ndarray:
    """Pillow 이미지를 OpenCV 표준 3채널 BGR uint8 배열로 변환한다.

    PNG의 RGBA, 팔레트 모드, Grayscale 입력도 먼저 RGB로 정규화한다.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pillow_to_grayscale(image: Image.Image) -> np.ndarray:
    """Pillow 이미지를 2차원 Grayscale uint8 배열로 변환한다."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    return np.asarray(image.convert("L"), dtype=np.uint8).copy()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV BGR 배열을 표시용 RGB 배열로 변환한다."""
    _validate_uint8_array(
        image,
        name="image",
        allowed_shapes=("three_channel",),
    )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """RGB 배열을 OpenCV BGR 배열로 변환한다."""
    _validate_uint8_array(
        image,
        name="image",
        allowed_shapes=("three_channel",),
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR 또는 Grayscale uint8 배열을 2차원 Grayscale로 정규화한다."""
    _validate_uint8_array(
        image,
        name="image",
        allowed_shapes=("grayscale", "three_channel"),
    )
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
