"""Day 12 Detection IoU 테스트."""

from __future__ import annotations

import pytest
import torch

from src.detection.iou import box_iou, box_iou_matrix, validate_xyxy_boxes


def tensor(values):
    return torch.tensor(values, dtype=torch.float32)


def test_identical_boxes_have_iou_one() -> None:
    box = tensor([0.0, 0.0, 10.0, 10.0])
    assert box_iou(box, box) == pytest.approx(1.0)


def test_partial_overlap_iou() -> None:
    first = tensor([0.0, 0.0, 10.0, 10.0])
    second = tensor([5.0, 5.0, 15.0, 15.0])
    assert box_iou(first, second) == pytest.approx(25.0 / 175.0)


def test_non_overlapping_and_edge_touching_boxes_have_zero_iou() -> None:
    first = tensor([0.0, 0.0, 10.0, 10.0])
    separated = tensor([20.0, 20.0, 30.0, 30.0])
    touching = tensor([10.0, 0.0, 20.0, 10.0])
    assert box_iou(first, separated) == pytest.approx(0.0)
    assert box_iou(first, touching) == pytest.approx(0.0)


def test_iou_matrix_shape_and_values() -> None:
    first = tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 20.0, 20.0],
        ]
    )
    second = tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
        ]
    )
    matrix = box_iou_matrix(first, second)

    assert matrix.shape == (2, 2)
    assert matrix[0, 0].item() == pytest.approx(1.0)
    assert matrix[0, 1].item() == pytest.approx(25.0 / 175.0)
    assert matrix[1, 0].item() == pytest.approx(0.0)


def test_empty_box_sets_return_correct_matrix_shape() -> None:
    empty = torch.empty((0, 4), dtype=torch.float32)
    one = tensor([[0.0, 0.0, 1.0, 1.0]])
    assert box_iou_matrix(empty, one).shape == (0, 1)
    assert box_iou_matrix(one, empty).shape == (1, 0)


@pytest.mark.parametrize(
    "boxes",
    [
        tensor([[0.0, 0.0, 0.0, 1.0]]),
        tensor([[1.0, 0.0, 0.0, 1.0]]),
        tensor([[0.0, 0.0, 1.0, float("nan")]]),
    ],
)
def test_invalid_boxes_are_rejected(boxes: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        validate_xyxy_boxes(boxes)
