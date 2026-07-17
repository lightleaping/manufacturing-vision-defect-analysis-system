"""IoU 기반 Detection Matching·Precision·Recall·AP@0.50.

정의
----
1. Operating point:
   - ``score >= score_threshold`` Prediction만 사용한다.
   - 같은 Class끼리만 비교한다.
   - Score 내림차순 Prediction이 아직 사용하지 않은 Ground Truth 중
     IoU가 가장 큰 Box를 선택한다.
   - ``IoU >= iou_threshold``면 TP, 아니면 FP다.
   - 사용되지 않은 Ground Truth는 FN이다.

2. AP:
   - Class별 모든 Prediction을 Dataset 전체에서 Score 내림차순으로 정렬한다.
   - 같은 이미지·같은 Class Ground Truth와 1:1 Greedy Matching한다.
   - Precision Envelope의 모든 Recall 변화 구간을 적분하는 all-point AP다.
   - Ground Truth가 없는 Class는 AP=None이며 mAP 평균에서 제외한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from src.detection.iou import box_iou_matrix, validate_xyxy_boxes


@dataclass(frozen=True, slots=True)
class MatchedPair:
    prediction_index: int
    ground_truth_index: int
    label: int
    score: float
    iou: float


@dataclass(frozen=True, slots=True)
class ImageMatchingResult:
    matches: tuple[MatchedPair, ...]
    true_positive_prediction_indexes: tuple[int, ...]
    false_positive_prediction_indexes: tuple[int, ...]
    false_negative_ground_truth_indexes: tuple[int, ...]
    kept_prediction_indexes: tuple[int, ...]


def _validate_probability(name: str, value: float, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    lower_ok = numeric > 0.0 if positive else numeric >= 0.0
    if not lower_ok or numeric > 1.0:
        operator = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be in {operator}.")
    return numeric


def _validate_labels(
    labels: Tensor,
    *,
    expected_count: int,
    name: str,
) -> Tensor:
    if not isinstance(labels, Tensor):
        raise TypeError(f"{name} must be torch.Tensor.")
    if labels.dtype != torch.int64 or labels.ndim != 1:
        raise TypeError(f"{name} must be Int64Tensor[N].")
    if labels.shape[0] != expected_count:
        raise ValueError(f"{name} count must match boxes.")
    if labels.numel() and int(labels.min()) < 1:
        raise ValueError(f"{name} must contain positive foreground labels.")
    return labels


def _validate_prediction(
    prediction: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(prediction, Mapping):
        raise TypeError("prediction must be a mapping.")
    missing = {"boxes", "labels", "scores"} - set(prediction)
    if missing:
        raise KeyError(f"prediction is missing keys: {sorted(missing)}.")
    boxes = validate_xyxy_boxes(prediction["boxes"], name="prediction boxes")
    labels = _validate_labels(
        prediction["labels"],
        expected_count=boxes.shape[0],
        name="prediction labels",
    )
    scores = prediction["scores"]
    if not isinstance(scores, Tensor):
        raise TypeError("prediction scores must be torch.Tensor.")
    if not scores.dtype.is_floating_point or scores.ndim != 1:
        raise TypeError("prediction scores must be FloatTensor[N].")
    if scores.shape[0] != boxes.shape[0]:
        raise ValueError("prediction score count must match boxes.")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("prediction scores contain NaN or infinity.")
    if scores.numel() and not bool(((scores >= 0.0) & (scores <= 1.0)).all()):
        raise ValueError("prediction scores must be in [0, 1].")
    return boxes, labels, scores


def _validate_target(
    target: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    if not isinstance(target, Mapping):
        raise TypeError("target must be a mapping.")
    missing = {"boxes", "labels"} - set(target)
    if missing:
        raise KeyError(f"target is missing keys: {sorted(missing)}.")
    boxes = validate_xyxy_boxes(target["boxes"], name="target boxes")
    labels = _validate_labels(
        target["labels"],
        expected_count=boxes.shape[0],
        name="target labels",
    )
    return boxes, labels


def match_predictions_to_ground_truth(
    *,
    prediction: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> ImageMatchingResult:
    """한 이미지의 Prediction을 Ground Truth와 Class-aware 1:1 Matching한다."""
    score_cutoff = _validate_probability("score_threshold", score_threshold)
    iou_cutoff = _validate_probability(
        "iou_threshold",
        iou_threshold,
        positive=True,
    )
    prediction_boxes, prediction_labels, prediction_scores = _validate_prediction(
        prediction
    )
    target_boxes, target_labels = _validate_target(target)

    kept = torch.where(prediction_scores >= score_cutoff)[0]
    if kept.numel() > 0:
        order = torch.argsort(prediction_scores[kept], descending=True)
        kept = kept[order]

    unmatched_ground_truth = set(range(target_boxes.shape[0]))
    matches: list[MatchedPair] = []
    true_positive_indexes: list[int] = []
    false_positive_indexes: list[int] = []

    for prediction_index_tensor in kept:
        prediction_index = int(prediction_index_tensor.item())
        label = int(prediction_labels[prediction_index].item())
        candidates = [
            ground_truth_index
            for ground_truth_index in unmatched_ground_truth
            if int(target_labels[ground_truth_index].item()) == label
        ]
        if not candidates:
            false_positive_indexes.append(prediction_index)
            continue

        candidate_boxes = target_boxes[candidates]
        ious = box_iou_matrix(
            prediction_boxes[prediction_index].reshape(1, 4),
            candidate_boxes,
        )[0]
        best_position = int(torch.argmax(ious).item())
        best_iou = float(ious[best_position].item())
        best_ground_truth_index = candidates[best_position]

        if best_iou >= iou_cutoff:
            unmatched_ground_truth.remove(best_ground_truth_index)
            true_positive_indexes.append(prediction_index)
            matches.append(
                MatchedPair(
                    prediction_index=prediction_index,
                    ground_truth_index=best_ground_truth_index,
                    label=label,
                    score=float(prediction_scores[prediction_index].item()),
                    iou=best_iou,
                )
            )
        else:
            false_positive_indexes.append(prediction_index)

    return ImageMatchingResult(
        matches=tuple(matches),
        true_positive_prediction_indexes=tuple(true_positive_indexes),
        false_positive_prediction_indexes=tuple(false_positive_indexes),
        false_negative_ground_truth_indexes=tuple(
            sorted(unmatched_ground_truth)
        ),
        kept_prediction_indexes=tuple(int(index.item()) for index in kept),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _f1(precision: float, recall: float) -> float:
    return (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )


def _average_precision_for_class(
    *,
    predictions: Sequence[Mapping[str, Tensor]],
    targets: Sequence[Mapping[str, Tensor]],
    label: int,
    iou_threshold: float,
) -> dict[str, Any]:
    ground_truth_by_image: dict[int, Tensor] = {}
    total_ground_truth = 0
    for image_index, target in enumerate(targets):
        target_boxes, target_labels = _validate_target(target)
        mask = target_labels == label
        selected = target_boxes[mask]
        ground_truth_by_image[image_index] = selected
        total_ground_truth += int(selected.shape[0])

    ranked_predictions: list[tuple[float, int, Tensor]] = []
    for image_index, prediction in enumerate(predictions):
        boxes, labels, scores = _validate_prediction(prediction)
        indexes = torch.where(labels == label)[0]
        for index in indexes.tolist():
            ranked_predictions.append(
                (
                    float(scores[index].item()),
                    image_index,
                    boxes[index].detach().clone(),
                )
            )
    ranked_predictions.sort(key=lambda item: item[0], reverse=True)

    if total_ground_truth == 0:
        return {
            "ap": None,
            "ground_truth_count": 0,
            "prediction_count": len(ranked_predictions),
            "precision_curve": [],
            "recall_curve": [],
        }

    matched = {
        image_index: set()
        for image_index in ground_truth_by_image
    }
    true_positives: list[int] = []
    false_positives: list[int] = []

    for _, image_index, predicted_box in ranked_predictions:
        image_ground_truth = ground_truth_by_image[image_index]
        available_indexes = [
            index
            for index in range(image_ground_truth.shape[0])
            if index not in matched[image_index]
        ]
        if not available_indexes:
            true_positives.append(0)
            false_positives.append(1)
            continue

        ious = box_iou_matrix(
            predicted_box.reshape(1, 4),
            image_ground_truth[available_indexes],
        )[0]
        best_position = int(torch.argmax(ious).item())
        best_iou = float(ious[best_position].item())
        best_ground_truth_index = available_indexes[best_position]
        if best_iou >= iou_threshold:
            matched[image_index].add(best_ground_truth_index)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp: list[int] = []
    cumulative_fp: list[int] = []
    tp_sum = 0
    fp_sum = 0
    for tp, fp in zip(true_positives, false_positives):
        tp_sum += tp
        fp_sum += fp
        cumulative_tp.append(tp_sum)
        cumulative_fp.append(fp_sum)

    precisions = [
        _safe_ratio(tp, tp + fp)
        for tp, fp in zip(cumulative_tp, cumulative_fp)
    ]
    recalls = [
        tp / total_ground_truth
        for tp in cumulative_tp
    ]

    recall_envelope = [0.0, *recalls, 1.0]
    precision_envelope = [0.0, *precisions, 0.0]
    for index in range(len(precision_envelope) - 2, -1, -1):
        precision_envelope[index] = max(
            precision_envelope[index],
            precision_envelope[index + 1],
        )

    average_precision = 0.0
    for index in range(1, len(recall_envelope)):
        recall_change = recall_envelope[index] - recall_envelope[index - 1]
        if recall_change > 0.0:
            average_precision += (
                recall_change * precision_envelope[index]
            )

    return {
        "ap": float(average_precision),
        "ground_truth_count": total_ground_truth,
        "prediction_count": len(ranked_predictions),
        "precision_curve": [float(value) for value in precisions],
        "recall_curve": [float(value) for value in recalls],
    }


def calculate_detection_metrics(
    *,
    predictions: Sequence[Mapping[str, Tensor]],
    targets: Sequence[Mapping[str, Tensor]],
    index_to_class: Mapping[int, str],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Dataset 전체의 Operating-point 지표와 AP@IoU를 계산한다."""
    if isinstance(predictions, (str, bytes)) or not isinstance(
        predictions,
        Sequence,
    ):
        raise TypeError("predictions must be a sequence.")
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence.")
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")
    if not predictions:
        raise ValueError("predictions and targets must not be empty.")

    class_names = dict(index_to_class)
    foreground_labels = sorted(
        label
        for label in class_names
        if isinstance(label, int) and label > 0
    )
    if not foreground_labels:
        raise ValueError("index_to_class must contain foreground labels.")
    if any(
        not isinstance(class_names[label], str) or not class_names[label]
        for label in foreground_labels
    ):
        raise ValueError("Every foreground class name must be non-empty.")

    score_cutoff = _validate_probability("score_threshold", score_threshold)
    iou_cutoff = _validate_probability(
        "iou_threshold",
        iou_threshold,
        positive=True,
    )

    counts = {
        label: {"tp": 0, "fp": 0, "fn": 0, "ious": []}
        for label in foreground_labels
    }
    image_results: list[dict[str, Any]] = []

    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        prediction_boxes, prediction_labels, _ = _validate_prediction(prediction)
        target_boxes, target_labels = _validate_target(target)

        unknown_prediction_labels = set(prediction_labels.tolist()) - set(
            foreground_labels
        )
        unknown_target_labels = set(target_labels.tolist()) - set(
            foreground_labels
        )
        if unknown_prediction_labels or unknown_target_labels:
            raise ValueError(
                "Prediction or target contains labels missing from index_to_class."
            )

        matching = match_predictions_to_ground_truth(
            prediction=prediction,
            target=target,
            score_threshold=score_cutoff,
            iou_threshold=iou_cutoff,
        )
        for pair in matching.matches:
            counts[pair.label]["tp"] += 1
            counts[pair.label]["ious"].append(pair.iou)
        for prediction_index in matching.false_positive_prediction_indexes:
            label = int(prediction_labels[prediction_index].item())
            counts[label]["fp"] += 1
        for ground_truth_index in matching.false_negative_ground_truth_indexes:
            label = int(target_labels[ground_truth_index].item())
            counts[label]["fn"] += 1

        image_results.append(
            {
                "image_index": image_index,
                "kept_prediction_count": len(matching.kept_prediction_indexes),
                "true_positive_count": len(
                    matching.true_positive_prediction_indexes
                ),
                "false_positive_count": len(
                    matching.false_positive_prediction_indexes
                ),
                "false_negative_count": len(
                    matching.false_negative_ground_truth_indexes
                ),
                "matched_mean_iou": (
                    None
                    if not matching.matches
                    else sum(pair.iou for pair in matching.matches)
                    / len(matching.matches)
                ),
            }
        )

    class_metrics: dict[str, dict[str, Any]] = {}
    ap_values: list[float] = []
    for label in foreground_labels:
        class_name = class_names[label]
        tp = int(counts[label]["tp"])
        fp = int(counts[label]["fp"])
        fn = int(counts[label]["fn"])
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        ious = [float(value) for value in counts[label]["ious"]]
        ap_result = _average_precision_for_class(
            predictions=predictions,
            targets=targets,
            label=label,
            iou_threshold=iou_cutoff,
        )
        if ap_result["ap"] is not None:
            ap_values.append(float(ap_result["ap"]))

        ap_value = ap_result["ap"]
        ap_50_value = ap_value if abs(iou_cutoff - 0.5) < 1e-12 else None
        class_metrics[class_name] = {
            "label": label,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ground_truth_count": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "mean_matched_iou": (
                None if not ious else sum(ious) / len(ious)
            ),
            "ap": ap_value,
            "ap_at_iou": ap_value,
            "ap_50": ap_50_value,
            "ap_ground_truth_count": ap_result["ground_truth_count"],
            "ap_prediction_count": ap_result["prediction_count"],
        }

    total_tp = sum(item["tp"] for item in counts.values())
    total_fp = sum(item["fp"] for item in counts.values())
    total_fn = sum(item["fn"] for item in counts.values())
    all_ious = [
        float(value)
        for item in counts.values()
        for value in item["ious"]
    ]
    precision = _safe_ratio(total_tp, total_tp + total_fp)
    recall = _safe_ratio(total_tp, total_tp + total_fn)

    map_value = None if not ap_values else sum(ap_values) / len(ap_values)
    map_50_value = map_value if abs(iou_cutoff - 0.5) < 1e-12 else None

    return {
        "schema_version": 1,
        "matching_policy": {
            "score_threshold": score_cutoff,
            "iou_threshold": iou_cutoff,
            "same_class_only": True,
            "prediction_order": "score_descending",
            "one_to_one_matching": True,
            "threshold_comparison": "inclusive",
            "ap_method": "all_point_interpolated_precision_envelope",
        },
        "overall": {
            "image_count": len(predictions),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "mean_matched_iou": (
                None if not all_ious else sum(all_ious) / len(all_ious)
            ),
            "map": map_value,
            "map_at_iou": map_value,
            "map_50": map_50_value,
            "map_class_count": len(ap_values),
        },
        "class_metrics": class_metrics,
        "image_metrics": image_results,
    }
