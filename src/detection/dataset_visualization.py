"""Day 9 Dataset 통계와 Bounding Box Overlay Figure 생성."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from .dataset_analysis import DatasetAnalysisResult, ImageAnnotationRecord


def _prepare_output(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def create_class_distribution_figure(
    result: DatasetAnalysisResult,
    output_path: Path,
) -> Path:
    """Class별 이미지 수와 Box 수를 한 Figure에 비교한다."""
    output_path = _prepare_output(output_path)
    image_counts = result.summary["class_image_counts"]
    box_counts = result.summary["class_box_counts"]
    assert isinstance(image_counts, dict)
    assert isinstance(box_counts, dict)

    classes = list(image_counts)
    positions = list(range(len(classes)))
    width = 0.38

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        [position - width / 2 for position in positions],
        [image_counts[name] for name in classes],
        width=width,
        label="Images",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [box_counts[name] for name in classes],
        width=width,
        label="Bounding Boxes",
    )
    axis.set_title("NEU-DET Class Distribution")
    axis.set_xlabel("Defect Class")
    axis.set_ylabel("Count")
    axis.set_xticks(positions)
    axis.set_xticklabels(classes, rotation=25, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def create_box_statistics_figure(
    result: DatasetAnalysisResult,
    output_path: Path,
) -> Path:
    """레코드의 Box 원자료로 주요 분포 Figure를 생성한다."""
    output_path = _prepare_output(output_path)
    boxes_per_image: list[int] = []
    widths: list[float] = []
    heights: list[float] = []
    area_ratios: list[float] = []
    aspect_ratios: list[float] = []

    for record in result.records:
        boxes_per_image.append(len(record.boxes))
        image_area = record.image_width * record.image_height
        for x_min, y_min, x_max, y_max in record.boxes:
            width = x_max - x_min
            height = y_max - y_min
            widths.append(width)
            heights.append(height)
            if image_area > 0:
                area_ratios.append((width * height) / image_area)
            if height > 0:
                aspect_ratios.append(width / height)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(boxes_per_image, bins="auto")
    axes[0, 0].set_title("Bounding Boxes per Image")
    axes[0, 0].set_xlabel("Box Count")

    axes[0, 1].hist(area_ratios, bins=30)
    axes[0, 1].set_title("Bounding Box Area Ratio")
    axes[0, 1].set_xlabel("Box Area / Image Area")

    axes[1, 0].hist(aspect_ratios, bins=30)
    axes[1, 0].set_title("Bounding Box Aspect Ratio")
    axes[1, 0].set_xlabel("Width / Height")

    axes[1, 1].scatter(widths, heights, alpha=0.5)
    axes[1, 1].set_title("Bounding Box Width vs Height")
    axes[1, 1].set_xlabel("Width")
    axes[1, 1].set_ylabel("Height")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _resolve_dataset_path(dataset_root: Path, stored_path: str) -> Path:
    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate
    return dataset_root / candidate


def create_annotation_overview_figure(
    result: DatasetAnalysisResult,
    *,
    dataset_root: Path,
    output_path: Path,
    max_samples: int = 6,
) -> Path:
    """실제 이미지 위에 Class Label과 Bounding Box를 Overlay한다."""
    if max_samples <= 0:
        raise ValueError("max_samples는 1 이상이어야 합니다.")
    if not result.records:
        raise ValueError("시각화할 유효 Annotation Record가 없습니다.")

    output_path = _prepare_output(output_path)

    # Class가 다양하게 보이도록 첫 등장 Class 기준으로 우선 선택한다.
    selected: list[ImageAnnotationRecord] = []
    seen_classes: set[str] = set()
    for record in result.records:
        record_classes = set(record.class_names)
        if record_classes - seen_classes:
            selected.append(record)
            seen_classes.update(record_classes)
        if len(selected) >= max_samples:
            break
    if len(selected) < max_samples:
        for record in result.records:
            if record not in selected:
                selected.append(record)
            if len(selected) >= max_samples:
                break

    columns = min(3, len(selected))
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for axis, record in zip(axes.flat, selected):
        image_path = _resolve_dataset_path(dataset_root, record.image_path)
        with Image.open(image_path) as image:
            image.load()
            axis.imshow(image, cmap="gray" if image.mode in {"1", "L", "I"} else None)

        for class_name, box_values in zip(record.class_names, record.boxes):
            x_min, y_min, x_max, y_max = box_values
            rectangle = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                linewidth=1.8,
            )
            axis.add_patch(rectangle)
            axis.text(
                x_min,
                max(0, y_min - 3),
                class_name,
                fontsize=8,
                bbox={"alpha": 0.65, "pad": 1},
            )
        axis.set_title(record.key)
        axis.axis("off")

    figure.suptitle("NEU-DET Annotation Overview")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
