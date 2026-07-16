from __future__ import annotations

import pytest

from src.explainability.gradcam_sample_selector import (
    GradCAMSampleSelectionError,
    extract_sample_results,
    select_gradcam_samples,
)


def _sample(
    sample_index: int,
    *,
    ground_truth_label: int,
    prediction: int,
    defect_probability: float,
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "image_path": f"sample_{sample_index}.jpeg",
        "ground_truth_label": ground_truth_label,
        "ground_truth_class_name": "DEFECT" if ground_truth_label == 1 else "NORMAL",
        "prediction": prediction,
        "prediction_class_name": "DEFECT" if prediction == 1 else "NORMAL",
        "defect_probability": defect_probability,
        "correct": ground_truth_label == prediction,
    }


def _complete_fixture() -> list[dict[str, object]]:
    return [
        _sample(1, ground_truth_label=0, prediction=0, defect_probability=0.02),
        _sample(2, ground_truth_label=0, prediction=0, defect_probability=0.10),
        _sample(3, ground_truth_label=1, prediction=1, defect_probability=0.99),
        _sample(4, ground_truth_label=1, prediction=1, defect_probability=0.90),
        _sample(5, ground_truth_label=0, prediction=1, defect_probability=0.90),
        _sample(6, ground_truth_label=0, prediction=1, defect_probability=0.80),
        _sample(7, ground_truth_label=0, prediction=1, defect_probability=0.51),
        _sample(8, ground_truth_label=1, prediction=0, defect_probability=0.10),
        _sample(9, ground_truth_label=1, prediction=0, defect_probability=0.49),
    ]


def test_select_gradcam_samples_returns_expected_seven_unique_samples() -> None:
    selected = select_gradcam_samples(_complete_fixture())

    assert [sample.selection_type for sample in selected] == [
        "HIGH_CONFIDENCE_TRUE_NEGATIVE",
        "HIGH_CONFIDENCE_TRUE_POSITIVE",
        "HIGH_CONFIDENCE_FALSE_POSITIVE_1",
        "HIGH_CONFIDENCE_FALSE_POSITIVE_2",
        "HIGH_CONFIDENCE_FALSE_NEGATIVE",
        "BOUNDARY_FALSE_POSITIVE",
        "BOUNDARY_FALSE_NEGATIVE",
    ]
    assert [sample.sample_index for sample in selected] == [1, 3, 5, 6, 8, 7, 9]
    assert len({sample.sample_index for sample in selected}) == 7


def test_selected_wrong_prediction_confidence_matches_prediction_class() -> None:
    selected = select_gradcam_samples(_complete_fixture())
    by_type = {sample.selection_type: sample for sample in selected}

    assert by_type["HIGH_CONFIDENCE_FALSE_POSITIVE_1"].wrong_prediction_confidence == pytest.approx(0.90)
    assert by_type["HIGH_CONFIDENCE_FALSE_NEGATIVE"].wrong_prediction_confidence == pytest.approx(0.90)
    assert by_type["BOUNDARY_FALSE_NEGATIVE"].threshold_distance == pytest.approx(0.01)


def test_select_gradcam_samples_rejects_duplicate_source_indices() -> None:
    samples = _complete_fixture()
    samples.append(dict(samples[0]))

    with pytest.raises(GradCAMSampleSelectionError, match="중복 sample_index"):
        select_gradcam_samples(samples)


def test_select_gradcam_samples_raises_when_required_type_is_missing() -> None:
    samples = [
        sample
        for sample in _complete_fixture()
        if not (
            sample["ground_truth_label"] == 1
            and sample["prediction"] == 0
        )
    ]

    with pytest.raises(GradCAMSampleSelectionError, match="HIGH_CONFIDENCE_FALSE_NEGATIVE"):
        select_gradcam_samples(samples)


def test_select_gradcam_samples_cross_validates_correct_field() -> None:
    samples = _complete_fixture()
    samples[0]["correct"] = False

    with pytest.raises(GradCAMSampleSelectionError, match="correct 값"):
        select_gradcam_samples(samples)


def test_extract_sample_results_reads_day4_artifact_key() -> None:
    samples = _complete_fixture()

    extracted = extract_sample_results({"sample_results": samples})

    assert extracted == samples


def test_extract_sample_results_rejects_missing_key() -> None:
    with pytest.raises(GradCAMSampleSelectionError, match="sample_results"):
        extract_sample_results({})
