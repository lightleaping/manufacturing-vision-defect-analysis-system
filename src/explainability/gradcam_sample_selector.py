from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class GradCAMSampleSelectionError(ValueError):
    """Grad-CAM 표본 선택 입력이나 결과가 유효하지 않을 때 발생합니다."""


@dataclass(frozen=True)
class SelectedGradCAMSample:
    """Day 6 Grad-CAM 대상으로 선택된 평가 표본입니다."""

    selection_type: str
    selection_reason: str
    sample_index: int
    image_path: str
    ground_truth_label: int
    ground_truth_class_name: str
    prediction: int
    prediction_class_name: str
    defect_probability: float
    correct: bool
    threshold_distance: float
    wrong_prediction_confidence: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_type": self.selection_type,
            "selection_reason": self.selection_reason,
            "sample_index": self.sample_index,
            "image_path": self.image_path,
            "ground_truth_label": self.ground_truth_label,
            "ground_truth_class_name": self.ground_truth_class_name,
            "prediction": self.prediction,
            "prediction_class_name": self.prediction_class_name,
            "defect_probability": self.defect_probability,
            "correct": self.correct,
            "threshold_distance": self.threshold_distance,
            "wrong_prediction_confidence": self.wrong_prediction_confidence,
        }


_REQUIRED_SAMPLE_KEYS = {
    "sample_index",
    "image_path",
    "ground_truth_label",
    "prediction",
    "defect_probability",
    "correct",
}


def _class_name(label: int) -> str:
    return "DEFECT" if label == 1 else "NORMAL"


def _validate_and_normalize_samples(
    sample_results: Sequence[Mapping[str, object]],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    if not 0.0 < threshold < 1.0:
        raise GradCAMSampleSelectionError("threshold는 0과 1 사이여야 합니다.")

    if not sample_results:
        raise GradCAMSampleSelectionError("sample_results가 비어 있습니다.")

    normalized: list[dict[str, object]] = []
    seen_indices: set[int] = set()

    for position, sample in enumerate(sample_results):
        missing = _REQUIRED_SAMPLE_KEYS - set(sample.keys())
        if missing:
            raise GradCAMSampleSelectionError(
                f"sample_results[{position}]에 필수 Key가 없습니다: {sorted(missing)}"
            )

        sample_index = sample["sample_index"]
        image_path = sample["image_path"]
        ground_truth_label = sample["ground_truth_label"]
        prediction = sample["prediction"]
        defect_probability = sample["defect_probability"]
        correct = sample["correct"]

        if not isinstance(sample_index, int) or isinstance(sample_index, bool):
            raise GradCAMSampleSelectionError("sample_index는 정수여야 합니다.")
        if sample_index in seen_indices:
            raise GradCAMSampleSelectionError(
                f"중복 sample_index가 발견되었습니다: {sample_index}"
            )
        seen_indices.add(sample_index)

        if not isinstance(image_path, str) or not image_path.strip():
            raise GradCAMSampleSelectionError("image_path는 비어 있지 않은 문자열이어야 합니다.")

        if ground_truth_label not in (0, 1):
            raise GradCAMSampleSelectionError("ground_truth_label은 0 또는 1이어야 합니다.")
        if prediction not in (0, 1):
            raise GradCAMSampleSelectionError("prediction은 0 또는 1이어야 합니다.")

        if not isinstance(defect_probability, (int, float)) or isinstance(
            defect_probability, bool
        ):
            raise GradCAMSampleSelectionError("defect_probability는 숫자여야 합니다.")

        probability = float(defect_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise GradCAMSampleSelectionError(
                "defect_probability는 유한한 0~1 값이어야 합니다."
            )

        if not isinstance(correct, bool):
            raise GradCAMSampleSelectionError("correct는 bool이어야 합니다.")

        expected_correct = ground_truth_label == prediction
        if correct != expected_correct:
            raise GradCAMSampleSelectionError(
                f"sample_index={sample_index}의 correct 값이 Label/Prediction과 일치하지 않습니다."
            )

        threshold_distance = abs(probability - threshold)
        wrong_prediction_confidence: float | None = None
        if not correct:
            wrong_prediction_confidence = (
                probability if prediction == 1 else 1.0 - probability
            )

        normalized.append(
            {
                "sample_index": sample_index,
                "image_path": image_path,
                "ground_truth_label": int(ground_truth_label),
                "ground_truth_class_name": str(
                    sample.get("ground_truth_class_name", _class_name(int(ground_truth_label)))
                ),
                "prediction": int(prediction),
                "prediction_class_name": str(
                    sample.get("prediction_class_name", _class_name(int(prediction)))
                ),
                "defect_probability": probability,
                "correct": correct,
                "threshold_distance": threshold_distance,
                "wrong_prediction_confidence": wrong_prediction_confidence,
            }
        )

    return normalized


def _pick_first_unused(
    candidates: Iterable[dict[str, object]],
    *,
    used_sample_indices: set[int],
    selection_type: str,
    selection_reason: str,
) -> SelectedGradCAMSample:
    for candidate in candidates:
        sample_index = int(candidate["sample_index"])
        if sample_index in used_sample_indices:
            continue

        used_sample_indices.add(sample_index)
        return SelectedGradCAMSample(
            selection_type=selection_type,
            selection_reason=selection_reason,
            sample_index=sample_index,
            image_path=str(candidate["image_path"]),
            ground_truth_label=int(candidate["ground_truth_label"]),
            ground_truth_class_name=str(candidate["ground_truth_class_name"]),
            prediction=int(candidate["prediction"]),
            prediction_class_name=str(candidate["prediction_class_name"]),
            defect_probability=float(candidate["defect_probability"]),
            correct=bool(candidate["correct"]),
            threshold_distance=float(candidate["threshold_distance"]),
            wrong_prediction_confidence=(
                None
                if candidate["wrong_prediction_confidence"] is None
                else float(candidate["wrong_prediction_confidence"])
            ),
        )

    raise GradCAMSampleSelectionError(
        f"선택 조건을 만족하는 고유 표본이 없습니다: {selection_type}"
    )


def select_gradcam_samples(
    sample_results: Sequence[Mapping[str, object]],
    *,
    threshold: float = 0.5,
) -> list[SelectedGradCAMSample]:
    """정분류 2장과 대표 오분류 5장을 선택합니다.

    선택 순서:
    1. 고확신 True Negative
    2. 고확신 True Positive
    3. 고확신 False Positive 1
    4. 고확신 False Positive 2
    5. 고확신 False Negative
    6. 결정 경계 False Positive
    7. 결정 경계 False Negative

    동일한 표본이 둘 이상의 기준에 걸리면 먼저 선택된 기준을 유지하고,
    뒤 기준에서는 다음 후보를 선택해 중복을 제거합니다.
    """

    normalized = _validate_and_normalize_samples(sample_results, threshold=threshold)

    true_negatives = sorted(
        (
            sample
            for sample in normalized
            if sample["ground_truth_label"] == 0
            and sample["prediction"] == 0
            and sample["correct"] is True
        ),
        key=lambda sample: (
            float(sample["defect_probability"]),
            int(sample["sample_index"]),
        ),
    )
    true_positives = sorted(
        (
            sample
            for sample in normalized
            if sample["ground_truth_label"] == 1
            and sample["prediction"] == 1
            and sample["correct"] is True
        ),
        key=lambda sample: (
            -float(sample["defect_probability"]),
            int(sample["sample_index"]),
        ),
    )
    false_positives_by_confidence = sorted(
        (
            sample
            for sample in normalized
            if sample["ground_truth_label"] == 0
            and sample["prediction"] == 1
            and sample["correct"] is False
        ),
        key=lambda sample: (
            -float(sample["wrong_prediction_confidence"]),
            int(sample["sample_index"]),
        ),
    )
    false_negatives_by_confidence = sorted(
        (
            sample
            for sample in normalized
            if sample["ground_truth_label"] == 1
            and sample["prediction"] == 0
            and sample["correct"] is False
        ),
        key=lambda sample: (
            -float(sample["wrong_prediction_confidence"]),
            int(sample["sample_index"]),
        ),
    )
    false_positives_by_boundary = sorted(
        false_positives_by_confidence,
        key=lambda sample: (
            float(sample["threshold_distance"]),
            int(sample["sample_index"]),
        ),
    )
    false_negatives_by_boundary = sorted(
        false_negatives_by_confidence,
        key=lambda sample: (
            float(sample["threshold_distance"]),
            int(sample["sample_index"]),
        ),
    )

    used_sample_indices: set[int] = set()
    selected: list[SelectedGradCAMSample] = []

    selected.append(
        _pick_first_unused(
            true_negatives,
            used_sample_indices=used_sample_indices,
            selection_type="HIGH_CONFIDENCE_TRUE_NEGATIVE",
            selection_reason="정상 정분류 중 P(DEFECT)가 가장 낮은 표본",
        )
    )
    selected.append(
        _pick_first_unused(
            true_positives,
            used_sample_indices=used_sample_indices,
            selection_type="HIGH_CONFIDENCE_TRUE_POSITIVE",
            selection_reason="불량 정분류 중 P(DEFECT)가 가장 높은 표본",
        )
    )
    selected.append(
        _pick_first_unused(
            false_positives_by_confidence,
            used_sample_indices=used_sample_indices,
            selection_type="HIGH_CONFIDENCE_FALSE_POSITIVE_1",
            selection_reason="False Positive 중 잘못된 예측 확신도가 가장 높은 표본",
        )
    )
    selected.append(
        _pick_first_unused(
            false_positives_by_confidence,
            used_sample_indices=used_sample_indices,
            selection_type="HIGH_CONFIDENCE_FALSE_POSITIVE_2",
            selection_reason="False Positive 중 잘못된 예측 확신도가 두 번째로 높은 고유 표본",
        )
    )
    selected.append(
        _pick_first_unused(
            false_negatives_by_confidence,
            used_sample_indices=used_sample_indices,
            selection_type="HIGH_CONFIDENCE_FALSE_NEGATIVE",
            selection_reason="False Negative 중 잘못된 예측 확신도가 가장 높은 표본",
        )
    )
    selected.append(
        _pick_first_unused(
            false_positives_by_boundary,
            used_sample_indices=used_sample_indices,
            selection_type="BOUNDARY_FALSE_POSITIVE",
            selection_reason="False Positive 중 Threshold 0.5에 가장 가까운 고유 표본",
        )
    )
    selected.append(
        _pick_first_unused(
            false_negatives_by_boundary,
            used_sample_indices=used_sample_indices,
            selection_type="BOUNDARY_FALSE_NEGATIVE",
            selection_reason="False Negative 중 Threshold 0.5에 가장 가까운 고유 표본",
        )
    )

    if len({sample.sample_index for sample in selected}) != len(selected):
        raise GradCAMSampleSelectionError("표본 선택 결과에 중복 sample_index가 있습니다.")

    return selected


def extract_sample_results(evaluation_artifact: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Day 4 평가 Artifact에서 sample_results 배열을 안전하게 추출합니다."""

    sample_results = evaluation_artifact.get("sample_results")
    if not isinstance(sample_results, list):
        raise GradCAMSampleSelectionError(
            "평가 Artifact의 sample_results는 list여야 합니다."
        )

    if not all(isinstance(sample, Mapping) for sample in sample_results):
        raise GradCAMSampleSelectionError(
            "sample_results의 각 항목은 Mapping이어야 합니다."
        )

    return sample_results
