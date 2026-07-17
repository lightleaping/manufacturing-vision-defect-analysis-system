from __future__ import annotations

import pytest
import torch

from src.detection.evaluation import (
    DEFAULT_IOU_THRESHOLDS,
    calculate_detection_iou_sweep,
    validate_iou_thresholds,
)


INDEX_TO_CLASS = {0: "background", 1: "crazing", 2: "inclusion"}


def _prediction(boxes, labels, scores):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def _target(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_default_iou_thresholds_are_coco_style_range() -> None:
    assert DEFAULT_IOU_THRESHOLDS == (
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    )


def test_perfect_predictions_have_map_one_for_full_iou_sweep() -> None:
    predictions = [
        _prediction([[0, 0, 10, 10]], [1], [0.9]),
        _prediction([[5, 5, 15, 15]], [2], [0.8]),
    ]
    targets = [
        _target([[0, 0, 10, 10]], [1]),
        _target([[5, 5, 15, 15]], [2]),
    ]

    result = calculate_detection_iou_sweep(
        predictions=predictions,
        targets=targets,
        index_to_class=INDEX_TO_CLASS,
    )

    assert result["summary"]["map_50"] == pytest.approx(1.0)
    assert result["summary"]["map_50_95"] == pytest.approx(1.0)
    assert result["class_map_50_95"]["crazing"] == pytest.approx(1.0)
    assert result["definition"]["official_coco_eval"] is False


def test_map_decreases_at_stricter_iou_thresholds() -> None:
    predictions = [_prediction([[0, 0, 8, 10]], [1], [0.9])]
    targets = [_target([[0, 0, 10, 10]], [1])]

    result = calculate_detection_iou_sweep(
        predictions=predictions,
        targets=targets,
        index_to_class=INDEX_TO_CLASS,
        iou_thresholds=(0.5, 0.75, 0.85),
    )

    assert result["threshold_metrics"]["0.50"]["overall"]["map_at_iou"] == 1.0
    assert result["threshold_metrics"]["0.75"]["overall"]["map_at_iou"] == 1.0
    assert result["threshold_metrics"]["0.85"]["overall"]["map_at_iou"] == 0.0


@pytest.mark.parametrize(
    "thresholds, error",
    [
        ((), ValueError),
        ((0.5, 0.5), ValueError),
        ((0.75, 0.5), ValueError),
        ((0.0,), ValueError),
        ((1.1,), ValueError),
        (("0.5",), TypeError),
    ],
)
def test_invalid_iou_thresholds_are_rejected(thresholds, error) -> None:
    with pytest.raises(error):
        validate_iou_thresholds(thresholds)
