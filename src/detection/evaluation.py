"""Day 12 Detection 최종 평가 보조 함수.

[그대로 재사용]
``src.detection.metrics.calculate_detection_metrics``가 구현한 Class-aware
1:1 Greedy Matching과 all-point AP를 각 IoU Threshold에 반복 적용한다.

[신규 구현]
- IoU 0.50~0.95 Sweep
- 프로젝트 mAP@0.50:0.95 계산
- 공식 COCO API와 다른 범위를 Artifact에 명확히 기록

주의
----
이 모듈의 ``map_50_95``는 pycocotools COCOeval 결과가 아니다. 같은
Prediction과 Ground Truth에 기존 프로젝트 AP 정의를 10개 IoU Threshold로
반복한 평균이다. maxDets, area range, 101-point interpolation은 적용하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from torch import Tensor

from src.detection.metrics import calculate_detection_metrics


DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.50 + 0.05 * index, 2)
    for index in range(10)
)


def validate_iou_thresholds(
    thresholds: Sequence[float],
) -> tuple[float, ...]:
    """중복 없는 오름차순 IoU Threshold를 검증한다."""
    if isinstance(thresholds, (str, bytes)) or not isinstance(
        thresholds,
        Sequence,
    ):
        raise TypeError("thresholds must be a sequence of numbers.")
    normalized: list[float] = []
    for index, value in enumerate(thresholds):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"thresholds[{index}] must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 < numeric <= 1.0:
            raise ValueError("Every IoU threshold must be finite and in (0, 1].")
        normalized.append(numeric)
    if not normalized:
        raise ValueError("thresholds must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("thresholds must not contain duplicates.")
    if normalized != sorted(normalized):
        raise ValueError("thresholds must be sorted in ascending order.")
    return tuple(normalized)


def calculate_detection_iou_sweep(
    *,
    predictions: Sequence[Mapping[str, Tensor]],
    targets: Sequence[Mapping[str, Tensor]],
    index_to_class: Mapping[int, str],
    score_threshold: float = 0.5,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """기존 Project AP 정의를 여러 IoU Threshold에 반복 적용한다."""
    thresholds = validate_iou_thresholds(iou_thresholds)
    threshold_metrics: dict[str, dict[str, Any]] = {}
    map_values: list[float] = []
    class_ap_values: dict[str, list[float]] = {}

    for threshold in thresholds:
        metrics = calculate_detection_metrics(
            predictions=predictions,
            targets=targets,
            index_to_class=index_to_class,
            score_threshold=score_threshold,
            iou_threshold=threshold,
        )
        key = f"{threshold:.2f}"
        threshold_metrics[key] = metrics
        map_value = metrics["overall"]["map_at_iou"]
        if map_value is not None:
            map_values.append(float(map_value))
        for class_name, class_metrics in metrics["class_metrics"].items():
            ap_value = class_metrics["ap_at_iou"]
            if ap_value is not None:
                class_ap_values.setdefault(class_name, []).append(
                    float(ap_value)
                )

    first_key = f"{thresholds[0]:.2f}"
    map_50 = (
        threshold_metrics.get("0.50", {})
        .get("overall", {})
        .get("map_at_iou")
    )
    class_map_50_95 = {
        class_name: (
            None
            if not values
            else sum(values) / len(values)
        )
        for class_name, values in sorted(class_ap_values.items())
    }

    return {
        "schema_version": 1,
        "definition": {
            "name": "project_all_point_ap_iou_sweep",
            "thresholds": list(thresholds),
            "score_threshold_for_operating_metrics": float(score_threshold),
            "ap_uses_all_ranked_predictions": True,
            "official_coco_eval": False,
            "not_applied": [
                "pycocotools",
                "COCO maxDets",
                "COCO area ranges",
                "COCO 101-point interpolation",
            ],
        },
        "summary": {
            "map_50": None if map_50 is None else float(map_50),
            "map_50_95": (
                None
                if not map_values
                else sum(map_values) / len(map_values)
            ),
            "threshold_count": len(thresholds),
            "first_threshold": thresholds[0],
            "last_threshold": thresholds[-1],
            "first_threshold_key": first_key,
        },
        "class_map_50_95": class_map_50_95,
        "threshold_metrics": threshold_metrics,
    }
