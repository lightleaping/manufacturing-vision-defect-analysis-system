"""ResNet18 오분류 표본 분석 모듈.

Day 4 평가 JSON을 읽고 다음 작업을 수행한다.

1. JSON 및 표본 Schema 검증
2. correct 값과 실제 Label/Prediction 관계 교차 검증
3. False Positive와 False Negative 구분
4. Threshold Distance 계산
5. Wrong Prediction Confidence 계산
6. 오분류 통계와 순위 생성
7. 분석 JSON의 Atomic 저장

이 모듈에서는 실제 이미지 파일을 열지 않는다.
이미지 로딩과 Figure 생성은 misclassification_visualization.py가 담당한다.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

NORMAL_LABEL = 0
DEFECT_LABEL = 1

FALSE_POSITIVE = "FALSE_POSITIVE"
FALSE_NEGATIVE = "FALSE_NEGATIVE"

# Day 4 JSON의 표본 배열을 담을 수 있는 제한된 Key 목록이다.
# 무제한 재귀 탐색을 하지 않아 잘못된 List를 표본으로 인식하는 것을 방지한다.
SAMPLE_LIST_KEYS = (
    "samples",
    "predictions",
    "sample_predictions",
    "per_sample_results",
    "evaluation_samples",
    "sample_results",
)

REQUIRED_SAMPLE_KEYS = (
    "sample_index",
    "image_path",
    "ground_truth_label",
    "ground_truth_class_name",
    "raw_logit",
    "defect_probability",
    "prediction",
    "prediction_class_name",
    "correct",
)


class MisclassificationAnalysisError(ValueError):
    """오분류 분석 입력이나 Schema가 올바르지 않을 때 발생하는 예외."""


def _require_mapping(
    value: Any,
    *,
    context: str,
) -> Mapping[str, Any]:
    """값이 JSON Object 형태인지 검증한다."""

    if not isinstance(value, Mapping):
        raise MisclassificationAnalysisError(
            f"{context} must be a JSON object, "
            f"but received {type(value).__name__}."
        )

    return value


def _require_non_negative_integer(
    value: Any,
    *,
    field_name: str,
    sample_position: int,
) -> int:
    """bool을 제외한 0 이상의 정수인지 검증한다."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            f"'{field_name}' must be an integer."
        )

    if value < 0:
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            f"'{field_name}' must be greater than or equal to 0."
        )

    return value


def _require_binary_label(
    value: Any,
    *,
    field_name: str,
    sample_position: int,
) -> int:
    """Label 또는 Prediction이 0이나 1인지 검증한다."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            f"'{field_name}' must be integer 0 or 1."
        )

    if value not in {NORMAL_LABEL, DEFECT_LABEL}:
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            f"'{field_name}' must be 0 or 1, but received {value}."
        )

    return value


def _require_finite_number(
    value: Any,
    *,
    field_name: str,
    sample_position: int,
) -> float:
    """bool을 제외한 유한한 실수인지 검증한다."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            f"'{field_name}' must be a finite number."
        )

    converted_value = float(value)

    if not math.isfinite(converted_value):
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            f"'{field_name}' must not be NaN or Infinity."
        )

    return converted_value


def _require_non_empty_string(
    value: Any,
    *,
    field_name: str,
    sample_position: int,
) -> str:
    """공백이 아닌 문자열인지 검증한다."""

    if not isinstance(value, str) or not value.strip():
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            f"'{field_name}' must be a non-empty string."
        )

    return value.strip()


def _extract_sample_list(payload: Any) -> list[Any]:
    """JSON 최상위 구조에서 평가 표본 배열을 추출한다.

    허용 구조:
    1. 최상위가 직접 List
    2. 최상위 Object 안에 정해진 이름의 List가 정확히 하나 존재
    """

    if isinstance(payload, list):
        return payload

    payload_mapping = _require_mapping(
        payload,
        context="Evaluation artifact root",
    )

    candidate_keys = [
        key
        for key in SAMPLE_LIST_KEYS
        if key in payload_mapping and isinstance(payload_mapping[key], list)
    ]

    if not candidate_keys:
        available_keys = ", ".join(sorted(str(key) for key in payload_mapping))
        expected_keys = ", ".join(SAMPLE_LIST_KEYS)

        raise MisclassificationAnalysisError(
            "Could not find the per-sample evaluation list. "
            f"Expected one of [{expected_keys}]. "
            f"Available root keys: [{available_keys}]."
        )

    if len(candidate_keys) > 1:
        raise MisclassificationAnalysisError(
            "Evaluation artifact contains multiple possible sample lists: "
            f"{candidate_keys}. Exactly one sample list is required."
        )

    return payload_mapping[candidate_keys[0]]


def _validate_sample(
    sample: Any,
    *,
    sample_position: int,
) -> dict[str, Any]:
    """평가 표본 하나의 필수 Key, Type, 값 범위를 검증한다."""

    sample_mapping = _require_mapping(
        sample,
        context=f"Sample position {sample_position}",
    )

    missing_keys = [
        key for key in REQUIRED_SAMPLE_KEYS if key not in sample_mapping
    ]

    if missing_keys:
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            f"missing required keys {missing_keys}."
        )

    sample_index = _require_non_negative_integer(
        sample_mapping["sample_index"],
        field_name="sample_index",
        sample_position=sample_position,
    )

    image_path = _require_non_empty_string(
        sample_mapping["image_path"],
        field_name="image_path",
        sample_position=sample_position,
    )

    ground_truth_label = _require_binary_label(
        sample_mapping["ground_truth_label"],
        field_name="ground_truth_label",
        sample_position=sample_position,
    )

    prediction = _require_binary_label(
        sample_mapping["prediction"],
        field_name="prediction",
        sample_position=sample_position,
    )

    ground_truth_class_name = _require_non_empty_string(
        sample_mapping["ground_truth_class_name"],
        field_name="ground_truth_class_name",
        sample_position=sample_position,
    )

    prediction_class_name = _require_non_empty_string(
        sample_mapping["prediction_class_name"],
        field_name="prediction_class_name",
        sample_position=sample_position,
    )

    raw_logit = _require_finite_number(
        sample_mapping["raw_logit"],
        field_name="raw_logit",
        sample_position=sample_position,
    )

    defect_probability = _require_finite_number(
        sample_mapping["defect_probability"],
        field_name="defect_probability",
        sample_position=sample_position,
    )

    if not 0.0 <= defect_probability <= 1.0:
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            "'defect_probability' must be between 0.0 and 1.0, "
            f"but received {defect_probability}."
        )

    correct = sample_mapping["correct"]

    # Python에서는 bool이 int의 하위 Type이므로 isinstance(True, int)가 True다.
    # 따라서 정확히 bool인지 별도로 확인한다.
    if type(correct) is not bool:
        raise MisclassificationAnalysisError(
            f"Sample position {sample_position}: "
            "'correct' must be a boolean."
        )

    expected_correct = ground_truth_label == prediction

    if correct != expected_correct:
        raise MisclassificationAnalysisError(
            f"Sample index {sample_index}: stored correct={correct} "
            "does not match "
            f"(ground_truth_label == prediction)={expected_correct}."
        )

    # 원본 Dictionary의 임의 필드에 의존하지 않고,
    # Day 5에서 사용할 검증 완료 필드만 반환한다.
    return {
        "sample_index": sample_index,
        "image_path": image_path,
        "ground_truth_label": ground_truth_label,
        "ground_truth_class_name": ground_truth_class_name,
        "raw_logit": raw_logit,
        "defect_probability": defect_probability,
        "prediction": prediction,
        "prediction_class_name": prediction_class_name,
        "correct": correct,
    }


def load_day4_evaluation_samples(
    artifact_path: str | Path,
) -> list[dict[str, Any]]:
    """Day 4 평가 JSON을 읽고 모든 표본을 검증한다."""

    resolved_path = Path(artifact_path)

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Day 4 evaluation artifact was not found: {resolved_path}"
        )

    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise MisclassificationAnalysisError(
            f"Invalid JSON artifact: {resolved_path}. "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error

    raw_samples = _extract_sample_list(payload)

    if not raw_samples:
        raise MisclassificationAnalysisError(
            "Evaluation artifact contains no samples."
        )

    validated_samples: list[dict[str, Any]] = []
    sample_indices: set[int] = set()

    for sample_position, raw_sample in enumerate(raw_samples):
        validated_sample = _validate_sample(
            raw_sample,
            sample_position=sample_position,
        )

        sample_index = validated_sample["sample_index"]

        if sample_index in sample_indices:
            raise MisclassificationAnalysisError(
                f"Duplicate sample_index detected: {sample_index}."
            )

        sample_indices.add(sample_index)
        validated_samples.append(validated_sample)

    return validated_samples


def _validate_threshold(threshold: float) -> float:
    """Classification Threshold가 유효한 확률 범위인지 검증한다."""

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise MisclassificationAnalysisError(
            "classification_threshold must be a number."
        )

    converted_threshold = float(threshold)

    if not math.isfinite(converted_threshold):
        raise MisclassificationAnalysisError(
            "classification_threshold must not be NaN or Infinity."
        )

    if not 0.0 < converted_threshold < 1.0:
        raise MisclassificationAnalysisError(
            "classification_threshold must be greater than 0.0 "
            "and less than 1.0."
        )

    return converted_threshold


def _portable_filename(image_path: str) -> str:
    """Windows와 POSIX 경로 구분자를 모두 고려해 파일명만 추출한다."""

    normalized_path = image_path.replace("\\", "/")
    return normalized_path.rsplit("/", maxsplit=1)[-1]


def _determine_error_type(sample: Mapping[str, Any]) -> str:
    """오분류 표본을 False Positive 또는 False Negative로 구분한다."""

    ground_truth_label = sample["ground_truth_label"]
    prediction = sample["prediction"]

    if sample["correct"]:
        raise MisclassificationAnalysisError(
            f"Sample index {sample['sample_index']} is correct and "
            "cannot be assigned an error type."
        )

    if ground_truth_label == NORMAL_LABEL and prediction == DEFECT_LABEL:
        return FALSE_POSITIVE

    if ground_truth_label == DEFECT_LABEL and prediction == NORMAL_LABEL:
        return FALSE_NEGATIVE

    raise MisclassificationAnalysisError(
        f"Sample index {sample['sample_index']} has an invalid "
        "misclassification combination: "
        f"ground_truth_label={ground_truth_label}, prediction={prediction}."
    )


def _enrich_misclassified_sample(
    sample: Mapping[str, Any],
    *,
    classification_threshold: float,
) -> dict[str, Any]:
    """오분류 표본에 오류 유형과 확신도 관련 값을 추가한다."""

    error_type = _determine_error_type(sample)
    defect_probability = float(sample["defect_probability"])
    prediction = int(sample["prediction"])

    threshold_distance = abs(
        defect_probability - classification_threshold
    )

    if prediction == DEFECT_LABEL:
        wrong_prediction_confidence = defect_probability
    else:
        wrong_prediction_confidence = 1.0 - defect_probability

    return {
        **dict(sample),
        "image_filename": _portable_filename(str(sample["image_path"])),
        "error_type": error_type,
        "classification_threshold": classification_threshold,
        "threshold_distance": threshold_distance,
        "wrong_prediction_confidence": wrong_prediction_confidence,
    }


def _build_numeric_summary(
    values: Sequence[float],
) -> dict[str, int | float | None]:
    """숫자 배열의 기본 통계를 JSON 저장 가능한 형태로 만든다."""

    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }

    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def build_misclassification_analysis(
    samples: Sequence[Mapping[str, Any]],
    *,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    source_artifact: str | Path | None = None,
    ranking_limit: int = 5,
) -> dict[str, Any]:
    """검증된 평가 표본으로 Day 5 오분류 분석 결과를 생성한다."""

    threshold = _validate_threshold(classification_threshold)

    if isinstance(ranking_limit, bool) or not isinstance(ranking_limit, int):
        raise MisclassificationAnalysisError(
            "ranking_limit must be an integer."
        )

    if ranking_limit <= 0:
        raise MisclassificationAnalysisError(
            "ranking_limit must be greater than 0."
        )

    if not samples:
        raise MisclassificationAnalysisError(
            "At least one evaluation sample is required."
        )

    validated_samples: list[dict[str, Any]] = []
    sample_indices: set[int] = set()

    # 파일에서 읽은 표본뿐 아니라 직접 전달된 표본도 동일하게 검증한다.
    for sample_position, sample in enumerate(samples):
        validated_sample = _validate_sample(
            sample,
            sample_position=sample_position,
        )

        sample_index = validated_sample["sample_index"]

        if sample_index in sample_indices:
            raise MisclassificationAnalysisError(
                f"Duplicate sample_index detected: {sample_index}."
            )

        sample_indices.add(sample_index)
        validated_samples.append(validated_sample)

    misclassifications = [
        _enrich_misclassified_sample(
            sample,
            classification_threshold=threshold,
        )
        for sample in validated_samples
        if not sample["correct"]
    ]

    # JSON에서 표본을 찾기 쉽게 원래 sample_index 순서로 저장한다.
    misclassifications.sort(key=lambda item: item["sample_index"])

    false_positives = [
        sample
        for sample in misclassifications
        if sample["error_type"] == FALSE_POSITIVE
    ]

    false_negatives = [
        sample
        for sample in misclassifications
        if sample["error_type"] == FALSE_NEGATIVE
    ]

    # 모델이 틀린 Class를 가장 강하게 확신한 순서다.
    high_confidence_errors = sorted(
        misclassifications,
        key=lambda item: (
            -item["wrong_prediction_confidence"],
            -item["threshold_distance"],
            item["sample_index"],
        ),
    )

    # Threshold에 가장 가까운, 즉 가장 애매한 오류 순서다.
    boundary_errors = sorted(
        misclassifications,
        key=lambda item: (
            item["threshold_distance"],
            item["sample_index"],
        ),
    )

    probability_values = [
        sample["defect_probability"] for sample in misclassifications
    ]

    distance_values = [
        sample["threshold_distance"] for sample in misclassifications
    ]

    confidence_values = [
        sample["wrong_prediction_confidence"]
        for sample in misclassifications
    ]

    total_samples = len(validated_samples)
    total_misclassified = len(misclassifications)
    correct_samples = total_samples - total_misclassified

    source_artifact_value = (
        str(Path(source_artifact))
        if source_artifact is not None
        else None
    )

    return {
        "analysis_name": (
            "Day 5 - ResNet18 Misclassified Image Analysis"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact": source_artifact_value,
        "classification_policy": {
            "normal_label": NORMAL_LABEL,
            "defect_label": DEFECT_LABEL,
            "positive_class": "DEFECT",
            "classification_threshold": threshold,
            "false_positive_definition": (
                "ground_truth_label=0 and prediction=1"
            ),
            "false_negative_definition": (
                "ground_truth_label=1 and prediction=0"
            ),
            "threshold_distance_formula": (
                "abs(defect_probability - classification_threshold)"
            ),
            "wrong_prediction_confidence_formula": (
                "defect_probability if prediction == 1 "
                "else 1 - defect_probability"
            ),
        },
        "summary": {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "misclassified_samples": total_misclassified,
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "error_rate": total_misclassified / total_samples,
            "defect_probability": _build_numeric_summary(
                probability_values
            ),
            "threshold_distance": _build_numeric_summary(
                distance_values
            ),
            "wrong_prediction_confidence": _build_numeric_summary(
                confidence_values
            ),
        },
        "rankings": {
            "most_confident_errors": high_confidence_errors[
                :ranking_limit
            ],
            "closest_boundary_errors": boundary_errors[:ranking_limit],
        },
        "misclassifications": misclassifications,
    }


def assert_expected_analysis_counts(
    analysis: Mapping[str, Any],
    *,
    expected_total_samples: int | None = None,
    expected_false_positive_count: int | None = None,
    expected_false_negative_count: int | None = None,
    expected_misclassified_count: int | None = None,
) -> None:
    """실제 Day 4 결과와 Day 5 분석 Count가 일치하는지 검증한다."""

    summary = _require_mapping(
        analysis.get("summary"),
        context="Analysis summary",
    )

    expected_values = {
        "total_samples": expected_total_samples,
        "false_positive_count": expected_false_positive_count,
        "false_negative_count": expected_false_negative_count,
        "misclassified_samples": expected_misclassified_count,
    }

    for summary_key, expected_value in expected_values.items():
        if expected_value is None:
            continue

        actual_value = summary.get(summary_key)

        if actual_value != expected_value:
            raise MisclassificationAnalysisError(
                f"Unexpected '{summary_key}': "
                f"expected {expected_value}, received {actual_value}."
            )


def save_json_atomic(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """분석 JSON을 임시 파일에 쓴 뒤 원자적으로 교체한다."""

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=resolved_output_path.parent,
            prefix=f".{resolved_output_path.name}.",
            suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )

            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, resolved_output_path)

    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

        raise

    return resolved_output_path