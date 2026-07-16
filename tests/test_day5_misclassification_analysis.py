"""Day 5 실행 Script 통합 테스트."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.run_day5_misclassification_analysis import (
    run_day5_misclassification_analysis,
)


def _create_image(
    path: Path,
    *,
    value: int,
) -> None:
    image = Image.new(
        mode="RGB",
        size=(48, 48),
        color=(value, value, value),
    )
    image.save(path)
    image.close()


def _make_sample(
    *,
    sample_index: int,
    image_path: str,
    ground_truth_label: int,
    prediction: int,
    defect_probability: float,
) -> dict:
    return {
        "sample_index": sample_index,
        "image_path": image_path,
        "ground_truth_label": ground_truth_label,
        "ground_truth_class_name": (
            "DEFECT" if ground_truth_label == 1 else "NORMAL"
        ),
        "raw_logit": 1.0 if prediction == 1 else -1.0,
        "defect_probability": defect_probability,
        "prediction": prediction,
        "prediction_class_name": (
            "DEFECT" if prediction == 1 else "NORMAL"
        ),
        "correct": ground_truth_label == prediction,
    }


def test_run_day5_misclassification_analysis_generates_artifacts(
    tmp_path: Path,
) -> None:
    fp_image_path = tmp_path / "fp.png"
    fn_image_path = tmp_path / "fn.png"
    normal_image_path = tmp_path / "normal.png"

    _create_image(fp_image_path, value=60)
    _create_image(fn_image_path, value=180)
    _create_image(normal_image_path, value=110)

    evaluation_path = tmp_path / "day4_evaluation.json"

    samples = [
        _make_sample(
            sample_index=0,
            image_path=fp_image_path.name,
            ground_truth_label=0,
            prediction=1,
            defect_probability=0.84,
        ),
        _make_sample(
            sample_index=1,
            image_path=fn_image_path.name,
            ground_truth_label=1,
            prediction=0,
            defect_probability=0.16,
        ),
        _make_sample(
            sample_index=2,
            image_path=normal_image_path.name,
            ground_truth_label=0,
            prediction=0,
            defect_probability=0.08,
        ),
    ]

    evaluation_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "accuracy": 1 / 3,
                },
                "predictions": samples,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    analysis_output_path = tmp_path / "analysis.json"
    fp_output_path = tmp_path / "fp_grid.png"
    fn_output_path = tmp_path / "fn_grid.png"
    all_output_path = tmp_path / "all_grid.png"

    result = run_day5_misclassification_analysis(
        input_artifact_path=evaluation_path,
        analysis_output_path=analysis_output_path,
        false_positive_figure_path=fp_output_path,
        false_negative_figure_path=fn_output_path,
        all_misclassifications_figure_path=all_output_path,
        project_root=tmp_path,
        expected_total_samples=3,
        expected_false_positive_count=1,
        expected_false_negative_count=1,
        expected_misclassified_count=2,
    )

    assert analysis_output_path.is_file()
    assert fp_output_path.is_file()
    assert fn_output_path.is_file()
    assert all_output_path.is_file()

    assert result["analysis"]["summary"][
        "false_positive_count"
    ] == 1

    assert result["analysis"]["summary"][
        "false_negative_count"
    ] == 1

    saved_analysis = json.loads(
        analysis_output_path.read_text(encoding="utf-8")
    )

    assert saved_analysis["summary"][
        "misclassified_samples"
    ] == 2