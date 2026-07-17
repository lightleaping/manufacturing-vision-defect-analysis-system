"""Day 12 Detection Matching·Metric·AP 테스트."""

from __future__ import annotations

import pytest
import torch

from src.detection.metrics import (
    calculate_detection_metrics,
    match_predictions_to_ground_truth,
)


INDEX_TO_CLASS = {
    0: "BACKGROUND",
    1: "crazing",
    2: "scratches",
}


def target(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def prediction(boxes, labels, scores):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def test_one_correct_prediction_is_true_positive() -> None:
    result = match_predictions_to_ground_truth(
        prediction=prediction(
            [[0.0, 0.0, 10.0, 10.0]],
            [1],
            [0.9],
        ),
        target=target(
            [[0.0, 0.0, 10.0, 10.0]],
            [1],
        ),
    )
    assert result.true_positive_prediction_indexes == (0,)
    assert result.false_positive_prediction_indexes == ()
    assert result.false_negative_ground_truth_indexes == ()
    assert result.matches[0].iou == pytest.approx(1.0)


def test_duplicate_prediction_becomes_false_positive() -> None:
    result = match_predictions_to_ground_truth(
        prediction=prediction(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
            ],
            [1, 1],
            [0.9, 0.8],
        ),
        target=target(
            [[0.0, 0.0, 10.0, 10.0]],
            [1],
        ),
    )
    assert result.true_positive_prediction_indexes == (0,)
    assert result.false_positive_prediction_indexes == (1,)
    assert result.false_negative_ground_truth_indexes == ()


def test_wrong_class_is_fp_and_fn() -> None:
    metrics = calculate_detection_metrics(
        predictions=[
            prediction(
                [[0.0, 0.0, 10.0, 10.0]],
                [2],
                [0.9],
            )
        ],
        targets=[
            target(
                [[0.0, 0.0, 10.0, 10.0]],
                [1],
            )
        ],
        index_to_class=INDEX_TO_CLASS,
    )
    assert metrics["overall"]["tp"] == 0
    assert metrics["overall"]["fp"] == 1
    assert metrics["overall"]["fn"] == 1
    assert metrics["class_metrics"]["scratches"]["fp"] == 1
    assert metrics["class_metrics"]["crazing"]["fn"] == 1


def test_score_and_iou_thresholds_are_inclusive() -> None:
    result = match_predictions_to_ground_truth(
        prediction=prediction(
            [[0.0, 0.0, 10.0, 10.0]],
            [1],
            [0.5],
        ),
        target=target(
            [[0.0, 0.0, 20.0, 10.0]],
            [1],
        ),
        score_threshold=0.5,
        iou_threshold=0.5,
    )
    assert result.true_positive_prediction_indexes == (0,)


def test_low_score_prediction_is_ignored_and_ground_truth_is_fn() -> None:
    result = match_predictions_to_ground_truth(
        prediction=prediction(
            [[0.0, 0.0, 10.0, 10.0]],
            [1],
            [0.49],
        ),
        target=target(
            [[0.0, 0.0, 10.0, 10.0]],
            [1],
        ),
        score_threshold=0.5,
    )
    assert result.kept_prediction_indexes == ()
    assert result.false_positive_prediction_indexes == ()
    assert result.false_negative_ground_truth_indexes == (0,)


def test_perfect_dataset_has_precision_recall_f1_and_map_one() -> None:
    metrics = calculate_detection_metrics(
        predictions=[
            prediction(
                [[0.0, 0.0, 10.0, 10.0]],
                [1],
                [0.95],
            ),
            prediction(
                [[5.0, 5.0, 15.0, 15.0]],
                [2],
                [0.90],
            ),
        ],
        targets=[
            target(
                [[0.0, 0.0, 10.0, 10.0]],
                [1],
            ),
            target(
                [[5.0, 5.0, 15.0, 15.0]],
                [2],
            ),
        ],
        index_to_class=INDEX_TO_CLASS,
    )
    overall = metrics["overall"]
    assert overall["tp"] == 2
    assert overall["fp"] == 0
    assert overall["fn"] == 0
    assert overall["precision"] == pytest.approx(1.0)
    assert overall["recall"] == pytest.approx(1.0)
    assert overall["f1"] == pytest.approx(1.0)
    assert overall["map"] == pytest.approx(1.0)
    assert overall["map_50"] == pytest.approx(1.0)


def test_average_precision_penalizes_high_score_false_positive() -> None:
    metrics = calculate_detection_metrics(
        predictions=[
            prediction(
                [
                    [20.0, 20.0, 30.0, 30.0],
                    [0.0, 0.0, 10.0, 10.0],
                ],
                [1, 1],
                [0.95, 0.90],
            )
        ],
        targets=[
            target(
                [[0.0, 0.0, 10.0, 10.0]],
                [1],
            )
        ],
        index_to_class=INDEX_TO_CLASS,
    )
    assert metrics["class_metrics"]["crazing"]["ap"] == pytest.approx(0.5)
    assert metrics["overall"]["map"] == pytest.approx(0.5)
    assert metrics["overall"]["map_50"] == pytest.approx(0.5)


def test_class_without_ground_truth_has_none_ap_and_is_excluded_from_map() -> None:
    metrics = calculate_detection_metrics(
        predictions=[
            prediction(
                [[0.0, 0.0, 10.0, 10.0]],
                [1],
                [0.9],
            )
        ],
        targets=[
            target(
                [[0.0, 0.0, 10.0, 10.0]],
                [1],
            )
        ],
        index_to_class=INDEX_TO_CLASS,
    )
    assert metrics["class_metrics"]["scratches"]["ap"] is None
    assert metrics["overall"]["map_class_count"] == 1
    assert metrics["overall"]["map"] == pytest.approx(1.0)
    assert metrics["overall"]["map_50"] == pytest.approx(1.0)
