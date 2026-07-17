"""Detection Image·Box 공동 Transform 테스트."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.detection.transforms import (
    Compose,
    RandomHorizontalFlip,
    ToFloatTensor,
    create_detection_transform,
)


def create_target() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor(
            [[1.0, 1.0, 5.0, 4.0], [0.0, 0.0, 10.0, 8.0]],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
        "image_id": torch.tensor([0], dtype=torch.int64),
        "area": torch.tensor([12.0, 80.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0, 0], dtype=torch.int64),
    }


def test_to_float_tensor_converts_rgb_image_to_zero_one_range() -> None:
    image = Image.new("RGB", (10, 8), color=(255, 128, 0))
    target = create_target()

    transformed, transformed_target = ToFloatTensor()(image, target)

    assert transformed.shape == (3, 8, 10)
    assert transformed.dtype == torch.float32
    assert transformed.min().item() == 0.0
    assert transformed.max().item() == 1.0
    assert torch.equal(transformed_target["boxes"], target["boxes"])


def test_grayscale_pillow_image_is_converted_to_rgb() -> None:
    image = Image.new("L", (10, 8), color=128)
    transformed, _ = ToFloatTensor()(image, create_target())

    assert transformed.shape == (3, 8, 10)


def test_horizontal_flip_changes_image_and_boxes_together() -> None:
    image = Image.new("RGB", (10, 8), color=(1, 2, 3))
    target = create_target()
    original_boxes = target["boxes"].clone()

    flipped_image, flipped_target = RandomHorizontalFlip(1.0)(image, target)

    assert flipped_image.size == image.size
    assert torch.equal(
        flipped_target["boxes"],
        torch.tensor(
            [[5.0, 1.0, 9.0, 4.0], [0.0, 0.0, 10.0, 8.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(target["boxes"], original_boxes)


def test_zero_probability_transform_is_deterministic() -> None:
    transform = create_detection_transform(
        training=True,
        horizontal_flip_probability=0.0,
    )
    image = Image.new("RGB", (10, 8), color=(10, 20, 30))
    target = create_target()

    first_image, first_target = transform(image, target)
    second_image, second_target = transform(image, target)

    assert torch.equal(first_image, second_image)
    assert torch.equal(first_target["boxes"], second_target["boxes"])


def test_compose_preserves_input_target() -> None:
    image = Image.new("RGB", (10, 8), color=(10, 20, 30))
    target = create_target()
    original = {key: value.clone() for key, value in target.items()}

    transform = Compose([RandomHorizontalFlip(1.0), ToFloatTensor()])
    _, _ = transform(image, target)

    for key in target:
        assert torch.equal(target[key], original[key])


def test_invalid_flip_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RandomHorizontalFlip(1.1)
