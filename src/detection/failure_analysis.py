"""Day 12 Detection Failure Analysis.

실패 유형
---------
- false_positive
- false_negative
- wrong_class
- low_iou_localization
- duplicate_prediction
- low_confidence_correct_detection

분류 원칙
---------
1. 운영 Score Threshold 이상 Prediction은 기존 Class-aware Matching으로 먼저
   TP·FP를 결정한다.
2. FP Prediction은 Duplicate → Wrong Class → Low IoU → 일반 FP 순서로
   하나의 Primary Category를 가진다.
3. FN Ground Truth는 이미 Wrong Class 또는 Low IoU Event에 연결됐다면 별도
   일반 FN Event를 중복 생성하지 않는다.
4. Score Threshold 미만이지만 같은 Class·IoU 기준을 만족하는 Prediction이
   있으면 Low Confidence Correct Detection으로 기록한다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
from torch import Tensor

from src.detection.iou import box_iou_matrix
from src.detection.metrics import match_predictions_to_ground_truth


FAILURE_CATEGORIES: tuple[str, ...] = (
    "false_positive",
    "false_negative",
    "wrong_class",
    "low_iou_localization",
    "duplicate_prediction",
    "low_confidence_correct_detection",
)


def _validate_probability(name: str, value: float, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    minimum_ok = numeric > 0.0 if positive else numeric >= 0.0
    if not minimum_ok or numeric > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}.")
    return numeric


def _validate_sample_ids(
    sample_ids: Sequence[str] | None,
    expected_count: int,
) -> tuple[str, ...]:
    if sample_ids is None:
        return tuple(f"image_{index}" for index in range(expected_count))
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise TypeError("sample_ids must be a sequence of strings or None.")
    normalized = tuple(sample_ids)
    if len(normalized) != expected_count:
        raise ValueError("sample_ids count must match predictions.")
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ValueError("Every sample id must be a non-empty string.")
    return normalized


def _prediction_components(
    prediction: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]
    except KeyError as error:
        raise KeyError("Prediction must contain boxes, labels, scores.") from error
    if not isinstance(boxes, Tensor) or boxes.dtype != torch.float32:
        raise TypeError("Prediction boxes must be float32 Tensor.")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("Prediction boxes must have shape [N, 4].")
    if not isinstance(labels, Tensor) or labels.dtype != torch.int64:
        raise TypeError("Prediction labels must be int64 Tensor.")
    if not isinstance(scores, Tensor) or not scores.dtype.is_floating_point:
        raise TypeError("Prediction scores must be floating Tensor.")
    if labels.shape != (boxes.shape[0],) or scores.shape != (boxes.shape[0],):
        raise ValueError("Prediction box, label, score counts must match.")
    return boxes, labels, scores


def _target_components(
    target: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    try:
        boxes = target["boxes"]
        labels = target["labels"]
    except KeyError as error:
        raise KeyError("Target must contain boxes and labels.") from error
    if not isinstance(boxes, Tensor) or boxes.dtype != torch.float32:
        raise TypeError("Target boxes must be float32 Tensor.")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("Target boxes must have shape [N, 4].")
    if not isinstance(labels, Tensor) or labels.dtype != torch.int64:
        raise TypeError("Target labels must be int64 Tensor.")
    if labels.shape != (boxes.shape[0],):
        raise ValueError("Target box and label counts must match.")
    return boxes, labels


def _best_candidate(
    *,
    prediction_box: Tensor,
    target_boxes: Tensor,
    candidate_indexes: Sequence[int],
) -> tuple[int | None, float]:
    if not candidate_indexes:
        return None, 0.0
    ious = box_iou_matrix(
        prediction_box.reshape(1, 4),
        target_boxes[list(candidate_indexes)],
    )[0]
    position = int(torch.argmax(ious).item())
    return int(candidate_indexes[position]), float(ious[position].item())


def _event(
    *,
    category: str,
    image_index: int,
    sample_id: str,
    prediction_index: int | None,
    ground_truth_index: int | None,
    predicted_label: int | None,
    ground_truth_label: int | None,
    score: float | None,
    iou: float | None,
    index_to_class: Mapping[int, str],
) -> dict[str, Any]:
    return {
        "category": category,
        "image_index": image_index,
        "sample_id": sample_id,
        "prediction_index": prediction_index,
        "ground_truth_index": ground_truth_index,
        "predicted_label": predicted_label,
        "predicted_class": (
            None if predicted_label is None else index_to_class[predicted_label]
        ),
        "ground_truth_label": ground_truth_label,
        "ground_truth_class": (
            None if ground_truth_label is None else index_to_class[ground_truth_label]
        ),
        "score": score,
        "iou": iou,
    }


def _representative_sort_key(event: Mapping[str, Any]) -> tuple[float, float, int]:
    iou = event.get("iou")
    score = event.get("score")
    return (
        -1.0 if iou is None else float(iou),
        -1.0 if score is None else float(score),
        -int(event["image_index"]),
    )


def analyze_detection_failures(
    *,
    predictions: Sequence[Mapping[str, Tensor]],
    targets: Sequence[Mapping[str, Tensor]],
    index_to_class: Mapping[int, str],
    sample_ids: Sequence[str] | None = None,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    localization_iou_floor: float = 0.1,
    low_confidence_floor: float = 0.05,
    representative_limit: int = 3,
) -> dict[str, Any]:
    """Dataset 전체 Prediction을 실패 유형별로 분석한다."""
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence.")
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence.")
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be non-empty and equal length.")
    if not isinstance(index_to_class, Mapping):
        raise TypeError("index_to_class must be a mapping.")
    class_names = dict(index_to_class)
    foreground = {index for index in class_names if isinstance(index, int) and index > 0}
    if not foreground:
        raise ValueError("index_to_class must contain foreground labels.")
    score_cutoff = _validate_probability("score_threshold", score_threshold)
    iou_cutoff = _validate_probability("iou_threshold", iou_threshold, positive=True)
    localization_floor = _validate_probability(
        "localization_iou_floor",
        localization_iou_floor,
    )
    low_score_floor = _validate_probability(
        "low_confidence_floor",
        low_confidence_floor,
    )
    if localization_floor >= iou_cutoff:
        raise ValueError("localization_iou_floor must be smaller than iou_threshold.")
    if low_score_floor >= score_cutoff:
        raise ValueError("low_confidence_floor must be smaller than score_threshold.")
    if not isinstance(representative_limit, int) or isinstance(representative_limit, bool):
        raise TypeError("representative_limit must be int.")
    if representative_limit <= 0:
        raise ValueError("representative_limit must be positive.")

    ids = _validate_sample_ids(sample_ids, len(predictions))
    events: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []

    for image_index, (prediction, target, sample_id) in enumerate(
        zip(predictions, targets, ids)
    ):
        prediction_boxes, prediction_labels, prediction_scores = _prediction_components(
            prediction
        )
        target_boxes, target_labels = _target_components(target)
        unknown = (
            set(prediction_labels.tolist()) | set(target_labels.tolist())
        ) - foreground
        if unknown:
            raise ValueError(f"Unknown foreground labels: {sorted(unknown)}.")

        matching = match_predictions_to_ground_truth(
            prediction=prediction,
            target=target,
            score_threshold=score_cutoff,
            iou_threshold=iou_cutoff,
        )
        matched_gt = {pair.ground_truth_index for pair in matching.matches}
        matched_pred = {pair.prediction_index for pair in matching.matches}
        associated_fn_gt: set[int] = set()
        image_events: list[dict[str, Any]] = []

        for prediction_index in matching.false_positive_prediction_indexes:
            predicted_label = int(prediction_labels[prediction_index].item())
            score = float(prediction_scores[prediction_index].item())
            same_class_indexes = [
                index
                for index, label in enumerate(target_labels.tolist())
                if int(label) == predicted_label
            ]
            duplicate_indexes = [
                index for index in same_class_indexes if index in matched_gt
            ]
            duplicate_gt, duplicate_iou = _best_candidate(
                prediction_box=prediction_boxes[prediction_index],
                target_boxes=target_boxes,
                candidate_indexes=duplicate_indexes,
            )
            if duplicate_gt is not None and duplicate_iou >= iou_cutoff:
                item = _event(
                    category="duplicate_prediction",
                    image_index=image_index,
                    sample_id=sample_id,
                    prediction_index=prediction_index,
                    ground_truth_index=duplicate_gt,
                    predicted_label=predicted_label,
                    ground_truth_label=int(target_labels[duplicate_gt].item()),
                    score=score,
                    iou=duplicate_iou,
                    index_to_class=class_names,
                )
                events.append(item)
                image_events.append(item)
                continue

            wrong_indexes = [
                index
                for index, label in enumerate(target_labels.tolist())
                if int(label) != predicted_label and index not in matched_gt
            ]
            wrong_gt, wrong_iou = _best_candidate(
                prediction_box=prediction_boxes[prediction_index],
                target_boxes=target_boxes,
                candidate_indexes=wrong_indexes,
            )
            if wrong_gt is not None and wrong_iou >= iou_cutoff:
                associated_fn_gt.add(wrong_gt)
                item = _event(
                    category="wrong_class",
                    image_index=image_index,
                    sample_id=sample_id,
                    prediction_index=prediction_index,
                    ground_truth_index=wrong_gt,
                    predicted_label=predicted_label,
                    ground_truth_label=int(target_labels[wrong_gt].item()),
                    score=score,
                    iou=wrong_iou,
                    index_to_class=class_names,
                )
                events.append(item)
                image_events.append(item)
                continue

            localization_indexes = [
                index for index in same_class_indexes if index not in matched_gt
            ]
            localization_gt, localization_iou = _best_candidate(
                prediction_box=prediction_boxes[prediction_index],
                target_boxes=target_boxes,
                candidate_indexes=localization_indexes,
            )
            if (
                localization_gt is not None
                and localization_floor <= localization_iou < iou_cutoff
            ):
                associated_fn_gt.add(localization_gt)
                item = _event(
                    category="low_iou_localization",
                    image_index=image_index,
                    sample_id=sample_id,
                    prediction_index=prediction_index,
                    ground_truth_index=localization_gt,
                    predicted_label=predicted_label,
                    ground_truth_label=int(target_labels[localization_gt].item()),
                    score=score,
                    iou=localization_iou,
                    index_to_class=class_names,
                )
                events.append(item)
                image_events.append(item)
                continue

            item = _event(
                category="false_positive",
                image_index=image_index,
                sample_id=sample_id,
                prediction_index=prediction_index,
                ground_truth_index=None,
                predicted_label=predicted_label,
                ground_truth_label=None,
                score=score,
                iou=None,
                index_to_class=class_names,
            )
            events.append(item)
            image_events.append(item)

        for ground_truth_index in matching.false_negative_ground_truth_indexes:
            if ground_truth_index in associated_fn_gt:
                continue
            ground_truth_label = int(target_labels[ground_truth_index].item())
            low_conf_indexes = [
                index
                for index, (label, score) in enumerate(
                    zip(prediction_labels.tolist(), prediction_scores.tolist())
                )
                if (
                    int(label) == ground_truth_label
                    and low_score_floor <= float(score) < score_cutoff
                    and index not in matched_pred
                )
            ]
            if low_conf_indexes:
                candidate_boxes = prediction_boxes[low_conf_indexes]
                ious = box_iou_matrix(
                    candidate_boxes,
                    target_boxes[ground_truth_index].reshape(1, 4),
                )[:, 0]
                best_position = int(torch.argmax(ious).item())
                best_prediction = low_conf_indexes[best_position]
                best_iou = float(ious[best_position].item())
                if best_iou >= iou_cutoff:
                    item = _event(
                        category="low_confidence_correct_detection",
                        image_index=image_index,
                        sample_id=sample_id,
                        prediction_index=best_prediction,
                        ground_truth_index=ground_truth_index,
                        predicted_label=int(prediction_labels[best_prediction].item()),
                        ground_truth_label=ground_truth_label,
                        score=float(prediction_scores[best_prediction].item()),
                        iou=best_iou,
                        index_to_class=class_names,
                    )
                    events.append(item)
                    image_events.append(item)
                    continue

            item = _event(
                category="false_negative",
                image_index=image_index,
                sample_id=sample_id,
                prediction_index=None,
                ground_truth_index=ground_truth_index,
                predicted_label=None,
                ground_truth_label=ground_truth_label,
                score=None,
                iou=None,
                index_to_class=class_names,
            )
            events.append(item)
            image_events.append(item)

        per_image.append(
            {
                "image_index": image_index,
                "sample_id": sample_id,
                "ground_truth_count": int(target_boxes.shape[0]),
                "prediction_count": int(prediction_boxes.shape[0]),
                "kept_prediction_count": len(matching.kept_prediction_indexes),
                "true_positive_count": len(matching.matches),
                "standard_false_positive_count": len(
                    matching.false_positive_prediction_indexes
                ),
                "standard_false_negative_count": len(
                    matching.false_negative_ground_truth_indexes
                ),
                "failure_event_count": len(image_events),
                "failure_categories": sorted(
                    {str(item["category"]) for item in image_events}
                ),
            }
        )

    counts = Counter(str(item["category"]) for item in events)
    representatives: dict[str, list[dict[str, Any]]] = {}
    for category in FAILURE_CATEGORIES:
        selected = sorted(
            (item for item in events if item["category"] == category),
            key=_representative_sort_key,
            reverse=True,
        )[:representative_limit]
        representatives[category] = selected

    by_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in events:
        class_name = item["ground_truth_class"] or item["predicted_class"]
        if class_name is not None:
            by_class[str(class_name)][str(item["category"])].append(item)
    class_representatives = {
        class_name: {
            category: sorted(values, key=_representative_sort_key, reverse=True)[:1]
            for category, values in sorted(category_map.items())
        }
        for class_name, category_map in sorted(by_class.items())
    }

    return {
        "schema_version": 1,
        "policy": {
            "score_threshold": score_cutoff,
            "iou_threshold": iou_cutoff,
            "localization_iou_floor": localization_floor,
            "low_confidence_floor": low_score_floor,
            "false_positive_priority": [
                "duplicate_prediction",
                "wrong_class",
                "low_iou_localization",
                "false_positive",
            ],
            "wrong_class_requires_iou_at_least": iou_cutoff,
            "low_iou_interval": [localization_floor, iou_cutoff],
            "test_split_mutation": False,
        },
        "summary": {
            "image_count": len(predictions),
            "event_count": len(events),
            "counts": {
                category: int(counts.get(category, 0))
                for category in FAILURE_CATEGORIES
            },
            "images_with_failures": sum(
                1 for item in per_image if item["failure_event_count"] > 0
            ),
        },
        "representative_samples": representatives,
        "class_representatives": class_representatives,
        "per_image": per_image,
        "events": events,
    }
