"""오분류 시각화 모듈 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.evaluation.misclassification_visualization import (
    MisclassificationVisualizationError,
    create_day5_misclassification_figures,
    save_misclassification_grid,
)


def _create_test_image(
    path: Path,
    *,
    value: int,
) -> None:
    """단순한 RGB 테스트 이미지를 생성한다."""

    image = Image.new(
        mode="RGB",
        size=(48, 48),
        color=(value, value, value),
    )

    image.save(path)
    image.close()


def _make_error_record(
    *,
    sample_index: int,
    image_path: str,
    error_type: str,
    defect_probability: float,
) -> dict:
    """시각화 테스트용 오분류 Record를 만든다."""

    if error_type == "FALSE_POSITIVE":
        ground_truth_label = 0
        ground_truth_class_name = "NORMAL"
        prediction = 1
        prediction_class_name = "DEFECT"
        wrong_prediction_confidence = defect_probability
    else:
        ground_truth_label = 1
        ground_truth_class_name = "DEFECT"
        prediction = 0
        prediction_class_name = "NORMAL"
        wrong_prediction_confidence = 1.0 - defect_probability

    return {
        "sample_index": sample_index,
        "image_path": image_path,
        "image_filename": Path(image_path).name,
        "ground_truth_label": ground_truth_label,
        "ground_truth_class_name": ground_truth_class_name,
        "raw_logit": 1.0 if prediction == 1 else -1.0,
        "defect_probability": defect_probability,
        "prediction": prediction,
        "prediction_class_name": prediction_class_name,
        "correct": False,
        "error_type": error_type,
        "classification_threshold": 0.5,
        "threshold_distance": abs(
            defect_probability - 0.5
        ),
        "wrong_prediction_confidence": (
            wrong_prediction_confidence
        ),
    }


def test_save_misclassification_grid_creates_png(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    output_path = tmp_path / "grid.png"

    _create_test_image(
        image_path,
        value=100,
    )

    record = _make_error_record(
        sample_index=0,
        image_path=image_path.name,
        error_type="FALSE_POSITIVE",
        defect_probability=0.8,
    )

    saved_path = save_misclassification_grid(
        [record],
        output_path=output_path,
        project_root=tmp_path,
        figure_title="Test Grid",
        max_columns=4,
        dpi=100,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    with Image.open(output_path) as generated_image:
        assert generated_image.format == "PNG"


def test_create_day5_misclassification_figures_creates_three_pngs(
    tmp_path: Path,
) -> None:
    fp_image_path = tmp_path / "fp.png"
    fn_image_path = tmp_path / "fn.png"

    _create_test_image(fp_image_path, value=70)
    _create_test_image(fn_image_path, value=180)

    analysis = {
        "misclassifications": [
            _make_error_record(
                sample_index=0,
                image_path=fp_image_path.name,
                error_type="FALSE_POSITIVE",
                defect_probability=0.88,
            ),
            _make_error_record(
                sample_index=1,
                image_path=fn_image_path.name,
                error_type="FALSE_NEGATIVE",
                defect_probability=0.12,
            ),
        ]
    }

    output_paths = create_day5_misclassification_figures(
        analysis,
        project_root=tmp_path,
        false_positive_output_path=tmp_path / "fp_grid.png",
        false_negative_output_path=tmp_path / "fn_grid.png",
        all_misclassifications_output_path=(
            tmp_path / "all_grid.png"
        ),
        max_columns=2,
        dpi=100,
    )

    assert output_paths["false_positives"].is_file()
    assert output_paths["false_negatives"].is_file()
    assert output_paths["all_misclassifications"].is_file()


def test_save_misclassification_grid_rejects_missing_image(
    tmp_path: Path,
) -> None:
    record = _make_error_record(
        sample_index=7,
        image_path="missing.png",
        error_type="FALSE_NEGATIVE",
        defect_probability=0.2,
    )

    with pytest.raises(
        FileNotFoundError,
        match="sample index 7",
    ):
        save_misclassification_grid(
            [record],
            output_path=tmp_path / "grid.png",
            project_root=tmp_path,
            figure_title="Missing Image Test",
        )


def test_save_misclassification_grid_rejects_corrupt_image(
    tmp_path: Path,
) -> None:
    corrupt_image_path = tmp_path / "corrupt.png"
    corrupt_image_path.write_text(
        "this is not an image",
        encoding="utf-8",
    )

    record = _make_error_record(
        sample_index=8,
        image_path=corrupt_image_path.name,
        error_type="FALSE_POSITIVE",
        defect_probability=0.7,
    )

    with pytest.raises(
        MisclassificationVisualizationError,
        match="Failed to open image",
    ):
        save_misclassification_grid(
            [record],
            output_path=tmp_path / "grid.png",
            project_root=tmp_path,
            figure_title="Corrupt Image Test",
        )


def test_save_misclassification_grid_rejects_empty_records(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        MisclassificationVisualizationError,
        match="At least one",
    ):
        save_misclassification_grid(
            [],
            output_path=tmp_path / "grid.png",
            project_root=tmp_path,
            figure_title="Empty Test",
        )


def test_visualization_rejects_error_type_label_mismatch(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    _create_test_image(image_path, value=120)

    record = _make_error_record(
        sample_index=0,
        image_path=image_path.name,
        error_type="FALSE_POSITIVE",
        defect_probability=0.8,
    )

    # FP인데 실제 Label을 DEFECT로 바꾸어 잘못된 조합을 만든다.
    record["ground_truth_label"] = 1

    with pytest.raises(
        MisclassificationVisualizationError,
        match="does not match",
    ):
        save_misclassification_grid(
            [record],
            output_path=tmp_path / "grid.png",
            project_root=tmp_path,
            figure_title="Mismatch Test",
        )