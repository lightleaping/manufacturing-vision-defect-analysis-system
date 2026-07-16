"""ResNet18 오분류 이미지 Grid 생성 모듈.

분석 모듈에서 생성한 오분류 Record를 사용하여 다음 Figure를 생성한다.

1. False Positive Grid
2. False Negative Grid
3. 전체 오분류 Grid

matplotlib의 Agg Backend를 사용하므로 GUI가 없는 Headless 환경에서도
plt.show() 없이 PNG 파일을 생성할 수 있다.
"""

from __future__ import annotations

import math
import os
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# pyplot를 Import하기 전에 Backend를 지정해야 한다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from src.evaluation.misclassification_analysis import (
    DEFECT_LABEL,
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    NORMAL_LABEL,
)


class MisclassificationVisualizationError(RuntimeError):
    """오분류 Figure 생성 중 문제가 발생했을 때 사용하는 예외."""


def _resolve_image_path(
    image_path: str,
    *,
    project_root: str | Path,
) -> Path:
    """JSON의 이미지 경로가 상대 경로라면 프로젝트 Root와 결합한다."""

    candidate_path = Path(image_path)

    if candidate_path.is_absolute():
        return candidate_path

    return Path(project_root) / candidate_path


def _load_rgb_image(
    image_path: Path,
    *,
    sample_index: int,
) -> Image.Image:
    """이미지를 열고 RGB 복사본을 반환한다."""

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file was not found for sample index "
            f"{sample_index}: {image_path}"
        )

    try:
        with Image.open(image_path) as source_image:
            rgb_image = source_image.convert("RGB")
            rgb_image.load()
            return rgb_image.copy()

    except (UnidentifiedImageError, OSError) as error:
        raise MisclassificationVisualizationError(
            f"Failed to open image for sample index "
            f"{sample_index}: {image_path}"
        ) from error


def _validate_visualization_record(
    record: Mapping[str, Any],
) -> None:
    """Figure에 사용할 오분류 Record의 핵심 필드를 검증한다."""

    required_keys = (
        "sample_index",
        "image_path",
        "ground_truth_label",
        "ground_truth_class_name",
        "defect_probability",
        "prediction",
        "prediction_class_name",
        "error_type",
        "threshold_distance",
        "wrong_prediction_confidence",
    )

    missing_keys = [
        key for key in required_keys if key not in record
    ]

    if missing_keys:
        raise MisclassificationVisualizationError(
            f"Visualization record is missing keys: {missing_keys}."
        )

    ground_truth_label = record["ground_truth_label"]
    prediction = record["prediction"]
    error_type = record["error_type"]

    if error_type == FALSE_POSITIVE:
        valid_combination = (
            ground_truth_label == NORMAL_LABEL
            and prediction == DEFECT_LABEL
        )
    elif error_type == FALSE_NEGATIVE:
        valid_combination = (
            ground_truth_label == DEFECT_LABEL
            and prediction == NORMAL_LABEL
        )
    else:
        raise MisclassificationVisualizationError(
            f"Unsupported error_type: {error_type}."
        )

    if not valid_combination:
        raise MisclassificationVisualizationError(
            "The error_type does not match ground truth and prediction: "
            f"sample_index={record['sample_index']}, "
            f"error_type={error_type}, "
            f"ground_truth_label={ground_truth_label}, "
            f"prediction={prediction}."
        )


def _shorten_filename(
    filename: str,
    *,
    maximum_width: int = 34,
) -> str:
    """Figure 제목에서 파일명이 지나치게 길어지는 것을 방지한다."""

    return textwrap.shorten(
        filename,
        width=maximum_width,
        placeholder="...",
    )


def _build_subplot_title(
    record: Mapping[str, Any],
) -> str:
    """오분류 이미지 하나에 표시할 제목을 만든다."""

    filename = str(
        record.get("image_filename")
        or Path(str(record["image_path"])).name
    )

    return (
        f"{record['error_type']} | "
        f"Sample #{record['sample_index']}\n"
        f"{_shorten_filename(filename)}\n"
        f"GT={record['ground_truth_class_name']} → "
        f"Pred={record['prediction_class_name']}\n"
        f"P(DEFECT)={record['defect_probability']:.4f} | "
        f"Wrong Conf.={record['wrong_prediction_confidence']:.4f}\n"
        f"Threshold Distance={record['threshold_distance']:.4f}"
    )


def save_misclassification_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path,
    project_root: str | Path,
    figure_title: str,
    max_columns: int = 4,
    dpi: int = 180,
) -> Path:
    """오분류 표본 배열을 이미지 Grid로 저장한다."""

    if not records:
        raise MisclassificationVisualizationError(
            "At least one misclassification record is required."
        )

    if isinstance(max_columns, bool) or not isinstance(max_columns, int):
        raise MisclassificationVisualizationError(
            "max_columns must be an integer."
        )

    if max_columns <= 0:
        raise MisclassificationVisualizationError(
            "max_columns must be greater than 0."
        )

    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise MisclassificationVisualizationError(
            "dpi must be a positive integer."
        )

    validated_records = []

    for record in records:
        _validate_visualization_record(record)
        validated_records.append(record)

    image_count = len(validated_records)
    column_count = min(max_columns, image_count)
    row_count = math.ceil(image_count / column_count)

    # 각 이미지 제목이 최대 5줄이므로 일반적인 Grid보다 높이를 넉넉하게 잡는다.
    figure_width = max(8.0, column_count * 4.2)
    figure_height = max(6.0, row_count * 5.8 + 1.5)

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    figure.suptitle(
        figure_title,
        fontsize=16,
        fontweight="bold",
    )

    loaded_images: list[Image.Image] = []

    try:
        for axis_index, axis in enumerate(axes.flat):
            if axis_index >= image_count:
                axis.axis("off")
                continue

            record = validated_records[axis_index]

            resolved_image_path = _resolve_image_path(
                str(record["image_path"]),
                project_root=project_root,
            )

            image = _load_rgb_image(
                resolved_image_path,
                sample_index=int(record["sample_index"]),
            )

            loaded_images.append(image)

            axis.imshow(image)
            axis.set_title(
                _build_subplot_title(record),
                fontsize=8,
                pad=14,
                linespacing=1.25,
            )
            axis.axis("off")

        figure.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.95),
            h_pad=4.5,
            w_pad=1.8,
        )

        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=resolved_output_path.parent,
                prefix=f".{resolved_output_path.name}.",
                suffix=".png",
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            figure.savefig(
                temporary_path,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
            )

            os.replace(temporary_path, resolved_output_path)

        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

            raise

        return resolved_output_path

    finally:
        for image in loaded_images:
            image.close()

        plt.close(figure)


def create_day5_misclassification_figures(
    analysis: Mapping[str, Any],
    *,
    project_root: str | Path,
    false_positive_output_path: str | Path,
    false_negative_output_path: str | Path,
    all_misclassifications_output_path: str | Path,
    max_columns: int = 4,
    dpi: int = 180,
) -> dict[str, Path]:
    """Day 5의 FP, FN, 전체 오분류 Figure를 생성한다."""

    raw_records = analysis.get("misclassifications")

    if not isinstance(raw_records, list):
        raise MisclassificationVisualizationError(
            "Analysis must contain a 'misclassifications' list."
        )

    if not raw_records:
        raise MisclassificationVisualizationError(
            "Analysis contains no misclassified samples."
        )

    false_positive_records = [
        record
        for record in raw_records
        if record.get("error_type") == FALSE_POSITIVE
    ]

    false_negative_records = [
        record
        for record in raw_records
        if record.get("error_type") == FALSE_NEGATIVE
    ]

    if not false_positive_records:
        raise MisclassificationVisualizationError(
            "No False Positive records were found."
        )

    if not false_negative_records:
        raise MisclassificationVisualizationError(
            "No False Negative records were found."
        )

    # FP와 FN Grid는 잘못된 예측 확신도가 높은 순서로 배치한다.
    false_positive_records.sort(
        key=lambda record: (
            -record["wrong_prediction_confidence"],
            record["sample_index"],
        )
    )

    false_negative_records.sort(
        key=lambda record: (
            -record["wrong_prediction_confidence"],
            record["sample_index"],
        )
    )

    # 전체 Grid에서는 FP를 먼저, FN을 다음에 배치한다.
    all_records = sorted(
        raw_records,
        key=lambda record: (
            0 if record["error_type"] == FALSE_POSITIVE else 1,
            -record["wrong_prediction_confidence"],
            record["sample_index"],
        ),
    )

    false_positive_path = save_misclassification_grid(
        false_positive_records,
        output_path=false_positive_output_path,
        project_root=project_root,
        figure_title=(
            "ResNet18 False Positives "
            f"(NORMAL → DEFECT, n={len(false_positive_records)})"
        ),
        max_columns=max_columns,
        dpi=dpi,
    )

    false_negative_path = save_misclassification_grid(
        false_negative_records,
        output_path=false_negative_output_path,
        project_root=project_root,
        figure_title=(
            "ResNet18 False Negatives "
            f"(DEFECT → NORMAL, n={len(false_negative_records)})"
        ),
        max_columns=max_columns,
        dpi=dpi,
    )

    all_misclassifications_path = save_misclassification_grid(
        all_records,
        output_path=all_misclassifications_output_path,
        project_root=project_root,
        figure_title=(
            "ResNet18 Test Misclassifications "
            f"(n={len(all_records)})"
        ),
        max_columns=max_columns,
        dpi=dpi,
    )

    return {
        "false_positives": false_positive_path,
        "false_negatives": false_negative_path,
        "all_misclassifications": all_misclassifications_path,
    }