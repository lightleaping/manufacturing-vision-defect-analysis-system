from __future__ import annotations

import pytest
import torch

from src.detection.failure_analysis import analyze_detection_failures


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


def test_all_required_failure_categories_are_distinguished() -> None:
    predictions = [
        _prediction([[0, 0, 10, 10], [0, 0, 10, 10]], [1, 1], [0.9, 0.8]),
        _prediction([[0, 0, 10, 10]], [2], [0.9]),
        _prediction([[5, 0, 15, 10]], [1], [0.9]),
        _prediction([[20, 20, 30, 30]], [1], [0.9]),
        _prediction([[0, 0, 10, 10]], [1], [0.4]),
        _prediction([], [], []),
    ]
    targets = [
        _target([[0, 0, 10, 10]], [1]),
        _target([[0, 0, 10, 10]], [1]),
        _target([[0, 0, 10, 10]], [1]),
        _target([], []),
        _target([[0, 0, 10, 10]], [1]),
        _target([[0, 0, 10, 10]], [1]),
    ]

    result = analyze_detection_failures(
        predictions=predictions,
        targets=targets,
        index_to_class=INDEX_TO_CLASS,
        sample_ids=[f"sample_{index}" for index in range(6)],
    )

    assert result["summary"]["counts"] == {
        "false_positive": 1,
        "false_negative": 1,
        "wrong_class": 1,
        "low_iou_localization": 1,
        "duplicate_prediction": 1,
        "low_confidence_correct_detection": 1,
    }
    assert result["summary"]["images_with_failures"] == 6
    wrong = result["representative_samples"]["wrong_class"][0]
    assert wrong["predicted_class"] == "inclusion"
    assert wrong["ground_truth_class"] == "crazing"


def test_perfect_prediction_has_no_failure_events() -> None:
    result = analyze_detection_failures(
        predictions=[_prediction([[0, 0, 10, 10]], [1], [0.9])],
        targets=[_target([[0, 0, 10, 10]], [1])],
        index_to_class=INDEX_TO_CLASS,
    )

    assert result["summary"]["event_count"] == 0
    assert result["summary"]["images_with_failures"] == 0


def test_invalid_failure_threshold_relationships_are_rejected() -> None:
    prediction = [_prediction([], [], [])]
    target = [_target([[0, 0, 10, 10]], [1])]

    with pytest.raises(ValueError, match="localization_iou_floor"):
        analyze_detection_failures(
            predictions=prediction,
            targets=target,
            index_to_class=INDEX_TO_CLASS,
            localization_iou_floor=0.5,
            iou_threshold=0.5,
        )
    with pytest.raises(ValueError, match="low_confidence_floor"):
        analyze_detection_failures(
            predictions=prediction,
            targets=target,
            index_to_class=INDEX_TO_CLASS,
            low_confidence_floor=0.5,
            score_threshold=0.5,
        )
