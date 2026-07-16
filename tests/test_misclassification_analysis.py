"""오분류 분석 모듈 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.misclassification_analysis import (
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    MisclassificationAnalysisError,
    assert_expected_analysis_counts,
    build_misclassification_analysis,
    load_day4_evaluation_samples,
    save_json_atomic,
)


def _make_sample(
    *,
    sample_index: int,
    ground_truth_label: int,
    prediction: int,
    defect_probability: float,
    image_path: str | None = None,
) -> dict:
    """테스트에 사용할 Day 4 평가 표본을 만든다."""

    ground_truth_class_name = (
        "DEFECT" if ground_truth_label == 1 else "NORMAL"
    )

    prediction_class_name = (
        "DEFECT" if prediction == 1 else "NORMAL"
    )

    return {
        "sample_index": sample_index,
        "image_path": (
            image_path
            or f"data/test/image_{sample_index}.jpeg"
        ),
        "ground_truth_label": ground_truth_label,
        "ground_truth_class_name": ground_truth_class_name,
        "raw_logit": 1.2 if prediction == 1 else -1.2,
        "defect_probability": defect_probability,
        "prediction": prediction,
        "prediction_class_name": prediction_class_name,
        "correct": ground_truth_label == prediction,
    }


def _write_json(
    path: Path,
    payload: object,
) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def test_load_day4_evaluation_samples_validates_samples_key(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "evaluation.json"

    samples = [
        _make_sample(
            sample_index=0,
            ground_truth_label=0,
            prediction=0,
            defect_probability=0.1,
        ),
        _make_sample(
            sample_index=1,
            ground_truth_label=1,
            prediction=0,
            defect_probability=0.2,
        ),
    ]

    _write_json(
        artifact_path,
        {
            "metrics": {"accuracy": 0.5},
            "samples": samples,
        },
    )

    loaded_samples = load_day4_evaluation_samples(
        artifact_path
    )

    assert loaded_samples == samples


def test_load_day4_evaluation_samples_accepts_root_list(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "evaluation.json"

    samples = [
        _make_sample(
            sample_index=10,
            ground_truth_label=0,
            prediction=1,
            defect_probability=0.8,
        )
    ]

    _write_json(artifact_path, samples)

    loaded_samples = load_day4_evaluation_samples(
        artifact_path
    )

    assert loaded_samples[0]["sample_index"] == 10


def test_load_day4_evaluation_samples_rejects_missing_key(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "evaluation.json"

    sample = _make_sample(
        sample_index=0,
        ground_truth_label=0,
        prediction=1,
        defect_probability=0.8,
    )
    del sample["raw_logit"]

    _write_json(
        artifact_path,
        {"samples": [sample]},
    )

    with pytest.raises(
        MisclassificationAnalysisError,
        match="missing required keys",
    ):
        load_day4_evaluation_samples(artifact_path)


def test_load_day4_evaluation_samples_cross_checks_correct(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "evaluation.json"

    sample = _make_sample(
        sample_index=0,
        ground_truth_label=0,
        prediction=1,
        defect_probability=0.8,
    )
    sample["correct"] = True

    _write_json(
        artifact_path,
        {"samples": [sample]},
    )

    with pytest.raises(
        MisclassificationAnalysisError,
        match="does not match",
    ):
        load_day4_evaluation_samples(artifact_path)


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_load_day4_evaluation_samples_rejects_invalid_probability(
    tmp_path: Path,
    invalid_probability: float,
) -> None:
    artifact_path = tmp_path / "evaluation.json"

    sample = _make_sample(
        sample_index=0,
        ground_truth_label=0,
        prediction=1,
        defect_probability=invalid_probability,
    )

    _write_json(
        artifact_path,
        {"samples": [sample]},
    )

    with pytest.raises(MisclassificationAnalysisError):
        load_day4_evaluation_samples(artifact_path)


def test_load_day4_evaluation_samples_rejects_duplicate_index(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "evaluation.json"

    samples = [
        _make_sample(
            sample_index=3,
            ground_truth_label=0,
            prediction=0,
            defect_probability=0.1,
        ),
        _make_sample(
            sample_index=3,
            ground_truth_label=1,
            prediction=1,
            defect_probability=0.9,
        ),
    ]

    _write_json(
        artifact_path,
        {"samples": samples},
    )

    with pytest.raises(
        MisclassificationAnalysisError,
        match="Duplicate sample_index",
    ):
        load_day4_evaluation_samples(artifact_path)


def test_build_misclassification_analysis_separates_fp_and_fn(
) -> None:
    samples = [
        _make_sample(
            sample_index=0,
            ground_truth_label=0,
            prediction=0,
            defect_probability=0.1,
        ),
        _make_sample(
            sample_index=1,
            ground_truth_label=0,
            prediction=1,
            defect_probability=0.82,
        ),
        _make_sample(
            sample_index=2,
            ground_truth_label=1,
            prediction=0,
            defect_probability=0.18,
        ),
        _make_sample(
            sample_index=3,
            ground_truth_label=1,
            prediction=1,
            defect_probability=0.95,
        ),
    ]

    analysis = build_misclassification_analysis(
        samples,
        classification_threshold=0.5,
    )

    summary = analysis["summary"]
    errors = analysis["misclassifications"]

    assert summary["total_samples"] == 4
    assert summary["correct_samples"] == 2
    assert summary["misclassified_samples"] == 2
    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 1

    false_positive = next(
        error
        for error in errors
        if error["error_type"] == FALSE_POSITIVE
    )

    false_negative = next(
        error
        for error in errors
        if error["error_type"] == FALSE_NEGATIVE
    )

    assert false_positive["threshold_distance"] == pytest.approx(
        0.32
    )
    assert false_positive[
        "wrong_prediction_confidence"
    ] == pytest.approx(0.82)

    assert false_negative["threshold_distance"] == pytest.approx(
        0.32
    )
    assert false_negative[
        "wrong_prediction_confidence"
    ] == pytest.approx(0.82)


def test_build_misclassification_analysis_creates_rankings(
) -> None:
    samples = [
        _make_sample(
            sample_index=1,
            ground_truth_label=0,
            prediction=1,
            defect_probability=0.51,
        ),
        _make_sample(
            sample_index=2,
            ground_truth_label=0,
            prediction=1,
            defect_probability=0.99,
        ),
        _make_sample(
            sample_index=3,
            ground_truth_label=1,
            prediction=0,
            defect_probability=0.45,
        ),
    ]

    analysis = build_misclassification_analysis(samples)

    most_confident = analysis["rankings"][
        "most_confident_errors"
    ]
    closest_boundary = analysis["rankings"][
        "closest_boundary_errors"
    ]

    assert most_confident[0]["sample_index"] == 2
    assert closest_boundary[0]["sample_index"] == 1


def test_assert_expected_analysis_counts_passes_for_expected_values(
) -> None:
    samples = [
        _make_sample(
            sample_index=0,
            ground_truth_label=0,
            prediction=1,
            defect_probability=0.8,
        ),
        _make_sample(
            sample_index=1,
            ground_truth_label=1,
            prediction=0,
            defect_probability=0.2,
        ),
    ]

    analysis = build_misclassification_analysis(samples)

    assert_expected_analysis_counts(
        analysis,
        expected_total_samples=2,
        expected_false_positive_count=1,
        expected_false_negative_count=1,
        expected_misclassified_count=2,
    )


def test_assert_expected_analysis_counts_rejects_mismatch(
) -> None:
    sample = _make_sample(
        sample_index=0,
        ground_truth_label=0,
        prediction=1,
        defect_probability=0.8,
    )

    analysis = build_misclassification_analysis([sample])

    with pytest.raises(
        MisclassificationAnalysisError,
        match="Unexpected 'false_positive_count'",
    ):
        assert_expected_analysis_counts(
            analysis,
            expected_false_positive_count=13,
        )


def test_save_json_atomic_writes_valid_json(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "analysis.json"
    payload = {
        "summary": {
            "misclassified_samples": 2,
        }
    }

    saved_path = save_json_atomic(
        payload,
        output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()

    loaded_payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert loaded_payload == payload

def test_load_day4_evaluation_samples_accepts_sample_results_key(
    tmp_path: Path,
) -> None:
    """?? Day 4 Artifact? sample_results Key? ???? ??."""

    artifact_path = tmp_path / "evaluation.json"

    samples = [
        _make_sample(
            sample_index=0,
            ground_truth_label=0,
            prediction=0,
            defect_probability=0.1,
        ),
        _make_sample(
            sample_index=1,
            ground_truth_label=1,
            prediction=0,
            defect_probability=0.2,
        ),
    ]

    _write_json(
        artifact_path,
        {
            "metrics": {
                "accuracy": 0.5,
            },
            "sample_results": samples,
        },
    )

    loaded_samples = load_day4_evaluation_samples(
        artifact_path
    )

    assert loaded_samples == samples

