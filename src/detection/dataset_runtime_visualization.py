"""Day 11 Detection Dataset Target을 성능 주장 없이 시각 검증한다.

이 Figure는 Ground Truth와 Dataset 좌표 변환을 확인하기 위한 것이다.
모델 Prediction 또는 Detection 성능 결과가 아니다.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor

from src.detection.data_loader import DetectionDataLoaders
from src.detection.detection_dataset import NeuDetDetectionDataset


_GRID_BACKGROUND = (245, 245, 245)
_BOX_COLOR = (220, 30, 30)
_TEXT_COLOR = (20, 20, 20)
_DUPLICATE_COLOR = (20, 80, 220)


def _tensor_to_pil(image: Tensor) -> Image.Image:
    if image.dtype != torch.float32 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be FloatTensor[3, H, W].")
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _draw_overlay(
    *,
    image: Tensor,
    target: Mapping[str, Tensor],
    index_to_class: Mapping[int, str],
    group_exact_duplicates: bool,
) -> Image.Image:
    canvas = _tensor_to_pil(image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    boxes = target["boxes"].detach().cpu().tolist()
    labels = target["labels"].detach().cpu().tolist()
    signatures = [
        (int(label), tuple(float(value) for value in box))
        for label, box in zip(labels, boxes)
    ]
    counts = Counter(signatures)
    drawn: set[tuple[int, tuple[float, float, float, float]]] = set()

    for signature in signatures:
        label, box = signature
        if group_exact_duplicates and signature in drawn:
            continue
        drawn.add(signature)
        occurrence_count = counts[signature]
        color = _DUPLICATE_COLOR if occurrence_count > 1 else _BOX_COLOR
        xmin, ymin, xmax, ymax = box
        draw.rectangle((xmin, ymin, xmax - 1, ymax - 1), outline=color, width=2)
        class_name = index_to_class.get(label, f"label_{label}")
        suffix = f" x{occurrence_count}" if occurrence_count > 1 else ""
        text = f"{class_name}{suffix}"
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0, int(ymin) - text_height - 2)
        draw.rectangle(
            (int(xmin), text_y, int(xmin) + text_width + 3, text_y + text_height + 2),
            fill=color,
        )
        draw.text((int(xmin) + 1, text_y + 1), text, fill=(255, 255, 255), font=font)
    return canvas


def _paste_cell(
    *,
    grid: Image.Image,
    cell_index: int,
    columns: int,
    cell_width: int,
    cell_height: int,
    title: str,
    overlay: Image.Image,
) -> None:
    row = cell_index // columns
    column = cell_index % columns
    left = column * cell_width
    top = row * cell_height
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    draw.text((left + 8, top + 7), title, fill=_TEXT_COLOR, font=font)

    available_width = cell_width - 16
    available_height = cell_height - 34
    scale = min(
        available_width / overlay.width,
        available_height / overlay.height,
    )
    resized = overlay.resize(
        (
            max(1, int(round(overlay.width * scale))),
            max(1, int(round(overlay.height * scale))),
        ),
        Image.Resampling.NEAREST,
    )
    image_left = left + (cell_width - resized.width) // 2
    image_top = top + 28 + (available_height - resized.height) // 2
    grid.paste(resized, (image_left, image_top))


def _figure_metadata(path: Path, semantic_role: str) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
    return {
        "path": path.as_posix(),
        "semantic_role": semantic_role,
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "decode_valid": True,
        "interpretation": (
            "Dataset ground-truth target and coordinate-conversion validation; "
            "not model predictions or performance results."
        ),
    }


def save_detection_dataset_batch_figure(
    *,
    dataset: NeuDetDetectionDataset,
    output_path: Path,
) -> dict[str, Any]:
    """Train Dataset에서 Class별 대표 Sample 한 장씩을 그린다."""
    selected: dict[int, tuple[int, Tensor, Mapping[str, Tensor]]] = {}
    desired_labels = set(range(1, len(dataset.class_names) + 1))

    for index in range(len(dataset)):
        image, target = dataset[index]
        for label in target["labels"].tolist():
            label = int(label)
            if label in desired_labels and label not in selected:
                selected[label] = (index, image, target)
        if set(selected) == desired_labels:
            break

    missing = desired_labels - set(selected)
    if missing:
        raise ValueError(f"Could not find representative samples for labels: {missing}.")

    columns = 3
    rows = 2
    cell_width = 300
    cell_height = 260
    grid = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        color=_GRID_BACKGROUND,
    )
    for cell_index, label in enumerate(sorted(selected)):
        index, image, target = selected[label]
        overlay = _draw_overlay(
            image=image,
            target=target,
            index_to_class=dataset.index_to_class,
            group_exact_duplicates=True,
        )
        _paste_cell(
            grid=grid,
            cell_index=cell_index,
            columns=columns,
            cell_width=cell_width,
            cell_height=cell_height,
            title=(
                f"{dataset.index_to_class[label]} | index={index} | "
                f"boxes={len(target['boxes'])}"
            ),
            overlay=overlay,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, format="PNG")
    return _figure_metadata(output_path, "class_representative_dataset_batch")


def save_duplicate_target_overlay_figure(
    *,
    loaders: DetectionDataLoaders,
    duplicate_records: list[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Duplicate Box가 포함된 Record를 occurrence count와 함께 그린다."""
    if not duplicate_records:
        raise ValueError("No duplicate records were found in the Manifest.")

    datasets = {
        "train": loaders.train_dataset,
        "validation": loaders.validation_dataset,
        "test": loaders.test_dataset,
    }
    cells: list[tuple[str, Image.Image]] = []
    for duplicate_record in duplicate_records:
        split = duplicate_record.get("split")
        record_id = duplicate_record.get("record_id")
        if split not in datasets or not isinstance(record_id, str):
            raise ValueError("Duplicate record has invalid split or record_id.")
        dataset = datasets[split]
        matching_indexes = [
            index
            for index, sample in enumerate(dataset.samples)
            if sample.record_id == record_id
        ]
        if len(matching_indexes) != 1:
            raise ValueError(
                f"Expected one Dataset record for {record_id!r}, "
                f"found {len(matching_indexes)}."
            )
        index = matching_indexes[0]
        image, target = dataset[index]
        overlay = _draw_overlay(
            image=image,
            target=target,
            index_to_class=dataset.index_to_class,
            group_exact_duplicates=True,
        )
        duplicate_count = sum(
            int(item["duplicate_count"])
            for item in duplicate_record.get("duplicates", [])
        )
        cells.append(
            (
                f"{record_id} | boxes={len(target['boxes'])} | "
                f"exact duplicates={duplicate_count}",
                overlay,
            )
        )

    columns = min(3, len(cells))
    rows = (len(cells) + columns - 1) // columns
    cell_width = 320
    cell_height = 280
    grid = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        color=_GRID_BACKGROUND,
    )
    for cell_index, (title, overlay) in enumerate(cells):
        _paste_cell(
            grid=grid,
            cell_index=cell_index,
            columns=columns,
            cell_width=cell_width,
            cell_height=cell_height,
            title=title,
            overlay=overlay,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, format="PNG")
    return _figure_metadata(output_path, "duplicate_box_target_overlay")
