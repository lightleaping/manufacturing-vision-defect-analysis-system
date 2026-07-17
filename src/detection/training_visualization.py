"""Day 12 Detection 학습·평가·실패 분석 Figure 생성.

[변경 이유]
기존 Montage는 200×200 원본 위에 긴 Class 이름을 직접 그린 뒤 확대해
Box·Label이 겹치고 글자가 작게 보이는 문제가 있었다.

[개선 정책]
- Ground Truth와 Prediction을 좌우 Panel로 분리한다.
- Box 위에는 G1·P1 같은 짧은 Tag만 표시한다.
- 전체 Class 이름·Score·판정은 이미지 밖 설명 영역에 표시한다.
- Tag 위치 충돌을 피하기 위해 후보 위치를 순서대로 탐색한다.
- Class Metric은 긴 Class 이름이 잘리지 않도록 가로 막대그래프로 그린다.
- Figure는 원자적으로 저장한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor

from src.detection.metrics import match_predictions_to_ground_truth


_BOX_COLORS = {
    "ground_truth": "deepskyblue",
    "correct_prediction": "limegreen",
    "false_positive": "red",
    "missed_ground_truth": "orange",
    "failure_prediction": "magenta",
}

_PREDICTION_PANEL_SIZE = 520
_FAILURE_PANEL_SIZE = 390
_CELL_PADDING = 18
_PANEL_GAP = 18
_HEADER_HEIGHT = 72
_PREDICTION_FOOTER_HEIGHT = 190
_FAILURE_FOOTER_HEIGHT = 125


def _validate_output_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("output_path must be pathlib.Path.")
    if path.suffix.lower() != ".png":
        raise ValueError("Figure output_path must use .png extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_save_figure(figure: Any, path: Path) -> Path:
    temporary = path.with_name(f".{path.stem}.tmp.png")
    try:
        figure.savefig(temporary, dpi=170, bbox_inches="tight")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
        plt.close(figure)
    return path


def _atomic_save_image(image: Image.Image, path: Path) -> Path:
    temporary = path.with_name(f".{path.stem}.tmp.png")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    if not isinstance(size, int) or size <= 0:
        raise ValueError("font size must be a positive int.")
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return int(left), int(top), int(right), int(bottom)


def _rectangles_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _place_tag_rectangle(
    *,
    preferred_x: int,
    preferred_y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    occupied: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    candidate_offsets = (
        (0, 0),
        (0, height + 3),
        (width + 3, 0),
        (0, -(height + 3)),
        (width + 3, height + 3),
        (-(width + 3), 0),
        (0, 2 * (height + 3)),
        (2 * (width + 3), 0),
    )
    for offset_x, offset_y in candidate_offsets:
        x1 = min(max(preferred_x + offset_x, 0), max(image_width - width, 0))
        y1 = min(max(preferred_y + offset_y, 0), max(image_height - height, 0))
        candidate = (x1, y1, x1 + width, y1 + height)
        if not any(_rectangles_overlap(candidate, item) for item in occupied):
            return candidate
    x1 = min(max(preferred_x, 0), max(image_width - width, 0))
    y1 = min(max(preferred_y, 0), max(image_height - height, 0))
    return (x1, y1, x1 + width, y1 + height)


def _draw_tagged_box(
    image: Image.Image,
    *,
    box: Sequence[float],
    source_width: int,
    source_height: int,
    color: str,
    tag: str,
    occupied_tags: list[tuple[int, int, int, int]],
    width: int = 5,
) -> None:
    draw = ImageDraw.Draw(image)
    scale_x = image.width / float(source_width)
    scale_y = image.height / float(source_height)
    x1, y1, x2, y2 = (float(value) for value in box)
    scaled = (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )
    draw.rectangle(scaled, outline=color, width=width)

    font = _load_font(19, bold=True)
    _, _, text_right, text_bottom = _text_bbox(draw, tag, font=font)
    padding_x = 6
    padding_y = 4
    tag_width = text_right + 2 * padding_x
    tag_height = text_bottom + 2 * padding_y
    rectangle = _place_tag_rectangle(
        preferred_x=scaled[0],
        preferred_y=scaled[1],
        width=tag_width,
        height=tag_height,
        image_width=image.width,
        image_height=image.height,
        occupied=occupied_tags,
    )
    occupied_tags.append(rectangle)
    draw.rectangle(rectangle, fill=color)
    draw.text(
        (rectangle[0] + padding_x, rectangle[1] + padding_y - 1),
        tag,
        fill="white",
        font=font,
    )


def _draw_panel_title(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    title: str,
) -> None:
    font = _load_font(22, bold=True)
    draw.rectangle((x, y, x + width, y + 42), fill="whitesmoke")
    draw.text((x + 10, y + 8), title, fill="black", font=font)


def _draw_text_lines(
    draw: ImageDraw.ImageDraw,
    *,
    lines: Sequence[str],
    x: int,
    y: int,
    max_width_chars: int,
    max_lines: int,
    font_size: int = 17,
) -> int:
    font = _load_font(font_size)
    wrapped: list[str] = []
    for line in lines:
        pieces = textwrap.wrap(
            str(line),
            width=max_width_chars,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        wrapped.extend(pieces)
    clipped = wrapped[:max_lines]
    if len(wrapped) > max_lines and clipped:
        clipped[-1] = clipped[-1].rstrip(" .") + " ..."
    cursor_y = y
    line_height = font_size + 6
    for line in clipped:
        draw.text((x, cursor_y), line, fill="black", font=font)
        cursor_y += line_height
    return cursor_y


def plot_detection_training_history(
    *,
    history: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Epoch별 Train Loss·Validation mAP@0.50·F1을 큰 Figure에 표시한다."""
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence.")
    if not history:
        raise ValueError("history must not be empty.")

    epochs: list[int] = []
    losses: list[float] = []
    map_50_values: list[float] = []
    f1_values: list[float] = []
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise TypeError(f"history[{index}] must be a mapping.")
        epoch = int(item["epoch"]) + 1
        train = item["train"]
        validation = item["validation"]
        loss = float(train["average_losses"]["total_loss"])
        overall = validation["metrics"]["overall"]
        map_value = overall["map_50"]
        if map_value is None:
            raise ValueError("Every history entry must contain validation map_50.")
        epochs.append(epoch)
        losses.append(loss)
        map_50_values.append(float(map_value))
        f1_values.append(float(overall["f1"]))

    path = _validate_output_path(output_path)
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(epochs, losses, marker="o", linewidth=2.2, label="Train total loss")
    axis.plot(
        epochs,
        map_50_values,
        marker="o",
        linewidth=2.2,
        label="Validation mAP@0.50",
    )
    axis.plot(epochs, f1_values, marker="o", linewidth=2.2, label="Validation F1")

    for values in (losses, map_50_values, f1_values):
        for epoch, value in zip(epochs, values):
            axis.annotate(
                f"{value:.3f}",
                (epoch, value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=10,
            )

    maximum = max(losses + map_50_values + f1_values)
    axis.set_ylim(0.0, max(1.1, maximum + 0.18))
    axis.set_xlabel("Epoch", fontsize=12)
    axis.set_ylabel("Value", fontsize=12)
    axis.set_title("Day 12 Detection Training History", fontsize=16, pad=16)
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", frameon=True)
    figure.tight_layout(pad=2.0)
    return _atomic_save_figure(figure, path)


def plot_detection_class_metrics(
    *,
    class_metrics: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Class별 Precision·Recall·F1·AP@0.50을 가로 막대로 표시한다."""
    if not isinstance(class_metrics, Mapping) or not class_metrics:
        raise ValueError("class_metrics must be a non-empty mapping.")

    names = list(class_metrics)
    precision = [float(class_metrics[name]["precision"]) for name in names]
    recall = [float(class_metrics[name]["recall"]) for name in names]
    f1 = [float(class_metrics[name]["f1"]) for name in names]
    ap_50 = [
        0.0 if class_metrics[name]["ap_50"] is None else float(class_metrics[name]["ap_50"])
        for name in names
    ]

    path = _validate_output_path(output_path)
    positions = list(range(len(names)))
    height = 0.18
    figure, axis = plt.subplots(figsize=(12, 8))
    groups = (
        (precision, -1.5 * height, "Precision"),
        (recall, -0.5 * height, "Recall"),
        (f1, 0.5 * height, "F1"),
        (ap_50, 1.5 * height, "AP@0.50"),
    )
    for values, offset, label in groups:
        bars = axis.barh(
            [position + offset for position in positions],
            values,
            height,
            label=label,
        )
        for bar, value in zip(bars, values):
            axis.text(
                min(value + 0.012, 0.985),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left" if value < 0.94 else "right",
                fontsize=9,
            )

    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.03)
    axis.set_xlabel("Score", fontsize=12)
    axis.set_title("Day 12 Test Class Metrics", fontsize=16, pad=16)
    axis.grid(True, axis="x", alpha=0.3)
    axis.legend(loc="lower right", frameon=True)
    figure.subplots_adjust(left=0.22, right=0.97, top=0.90, bottom=0.10)
    return _atomic_save_figure(figure, path)


def _tensor_to_pil(image: Tensor) -> Image.Image:
    if not isinstance(image, Tensor) or image.dtype != torch.float32:
        raise TypeError("image must be float32 Tensor.")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W].")
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), resample=Image.Resampling.LANCZOS)


def _sample_id(dataset: Any, image_index: int) -> str:
    if hasattr(dataset, "samples"):
        return str(dataset.samples[image_index].record_id)
    return f"image_{image_index}"


def _prediction_cell(
    *,
    image: Image.Image,
    target: Mapping[str, Tensor],
    prediction: Mapping[str, Tensor],
    matching: Any,
    index_to_class: Mapping[int, str],
    sample_id: str,
    max_predictions_per_image: int,
) -> Image.Image:
    source_width, source_height = image.size
    panel_size = _PREDICTION_PANEL_SIZE
    cell_width = 2 * panel_size + _PANEL_GAP + 2 * _CELL_PADDING
    cell_height = (
        _HEADER_HEIGHT
        + 42
        + panel_size
        + _PREDICTION_FOOTER_HEIGHT
        + 2 * _CELL_PADDING
    )
    cell = Image.new("RGB", (cell_width, cell_height), "white")
    draw = ImageDraw.Draw(cell)
    title_font = _load_font(23, bold=True)
    subtitle_font = _load_font(17)
    draw.text(
        (_CELL_PADDING, _CELL_PADDING),
        sample_id,
        fill="black",
        font=title_font,
    )
    summary = (
        f"TP={len(matching.matches)}  "
        f"FP={len(matching.false_positive_prediction_indexes)}  "
        f"FN={len(matching.false_negative_ground_truth_indexes)}"
    )
    draw.text(
        (_CELL_PADDING, _CELL_PADDING + 34),
        summary,
        fill="dimgray",
        font=subtitle_font,
    )

    left_x = _CELL_PADDING
    right_x = _CELL_PADDING + panel_size + _PANEL_GAP
    panel_y = _CELL_PADDING + _HEADER_HEIGHT
    ground_truth_panel = _resize_square(image, panel_size)
    prediction_panel = _resize_square(image, panel_size)
    _draw_panel_title(
        draw,
        x=left_x,
        y=panel_y,
        width=panel_size,
        title="Ground truth",
    )
    _draw_panel_title(
        draw,
        x=right_x,
        y=panel_y,
        width=panel_size,
        title="Predictions",
    )

    gt_image_y = panel_y + 42
    matched_gt = {pair.ground_truth_index for pair in matching.matches}
    gt_occupied: list[tuple[int, int, int, int]] = []
    gt_lines: list[str] = []
    for gt_index, (box, label_value) in enumerate(
        zip(target["boxes"].tolist(), target["labels"].tolist()),
        start=1,
    ):
        target_index = gt_index - 1
        category = "ground_truth" if target_index in matched_gt else "missed_ground_truth"
        class_name = index_to_class[int(label_value)]
        _draw_tagged_box(
            ground_truth_panel,
            box=box,
            source_width=source_width,
            source_height=source_height,
            color=_BOX_COLORS[category],
            tag=f"G{gt_index}",
            occupied_tags=gt_occupied,
        )
        status = "matched" if target_index in matched_gt else "missed"
        gt_lines.append(f"G{gt_index}: {class_name} [{status}]")

    true_positive = set(matching.true_positive_prediction_indexes)
    kept_indexes = list(matching.kept_prediction_indexes)
    kept_indexes.sort(
        key=lambda index: float(prediction["scores"][index].item()),
        reverse=True,
    )
    displayed_indexes = kept_indexes[:max_predictions_per_image]
    pred_occupied: list[tuple[int, int, int, int]] = []
    pred_lines: list[str] = []
    for display_index, prediction_index in enumerate(displayed_indexes, start=1):
        box = prediction["boxes"][prediction_index].tolist()
        label_value = int(prediction["labels"][prediction_index].item())
        score = float(prediction["scores"][prediction_index].item())
        category = (
            "correct_prediction"
            if prediction_index in true_positive
            else "false_positive"
        )
        status = "TP" if prediction_index in true_positive else "FP"
        _draw_tagged_box(
            prediction_panel,
            box=box,
            source_width=source_width,
            source_height=source_height,
            color=_BOX_COLORS[category],
            tag=f"P{display_index}",
            occupied_tags=pred_occupied,
        )
        pred_lines.append(
            f"P{display_index}: {index_to_class[label_value]} "
            f"score={score:.2f} [{status}]"
        )
    hidden_count = len(kept_indexes) - len(displayed_indexes)
    if hidden_count > 0:
        pred_lines.append(f"+ {hidden_count} lower-score predictions not drawn")

    cell.paste(ground_truth_panel, (left_x, gt_image_y))
    cell.paste(prediction_panel, (right_x, gt_image_y))
    footer_y = gt_image_y + panel_size + 10
    _draw_text_lines(
        draw,
        lines=gt_lines or ["No ground-truth boxes"],
        x=left_x,
        y=footer_y,
        max_width_chars=45,
        max_lines=7,
    )
    _draw_text_lines(
        draw,
        lines=pred_lines or ["No predictions above score threshold"],
        x=right_x,
        y=footer_y,
        max_width_chars=48,
        max_lines=7,
    )
    return cell


def _compose_grid(
    *,
    cells: Sequence[Image.Image],
    output_path: Path,
    columns: int,
    gap: int = 18,
) -> Path:
    if not cells:
        raise ValueError("cells must not be empty.")
    if not isinstance(columns, int) or columns <= 0:
        raise ValueError("columns must be a positive int.")
    cell_width = max(image.width for image in cells)
    cell_height = max(image.height for image in cells)
    rows = (len(cells) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            cell_width * columns + gap * (columns - 1),
            cell_height * rows + gap * (rows - 1),
        ),
        "gainsboro",
    )
    for index, image in enumerate(cells):
        row, column = divmod(index, columns)
        x = column * (cell_width + gap)
        y = row * (cell_height + gap)
        canvas.paste(image, (x, y))
    return _atomic_save_image(canvas, _validate_output_path(output_path))


def create_detection_prediction_montage(
    *,
    dataset: Any,
    predictions: Sequence[Mapping[str, Tensor]],
    targets: Sequence[Mapping[str, Tensor]],
    index_to_class: Mapping[int, str],
    output_path: Path,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    max_images: int = 4,
    max_predictions_per_image: int = 8,
) -> Path:
    """대표 Test Sample을 GT·Prediction 분리 Panel로 표시한다."""
    if len(predictions) != len(targets) or len(predictions) != len(dataset):
        raise ValueError("dataset, predictions, targets lengths must match.")
    if max_images <= 0 or max_predictions_per_image <= 0:
        raise ValueError("max_images and max_predictions_per_image must be positive.")

    ranked: list[tuple[int, int, int]] = []
    matchings = []
    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        matching = match_predictions_to_ground_truth(
            prediction=prediction,
            target=target,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        matchings.append(matching)
        ranked.append(
            (
                len(matching.matches),
                -len(matching.false_positive_prediction_indexes),
                image_index,
            )
        )
    selected_indexes = [
        item[2] for item in sorted(ranked, reverse=True)[:max_images]
    ]
    cells: list[Image.Image] = []
    for image_index in selected_indexes:
        image_tensor, _ = dataset[image_index]
        cells.append(
            _prediction_cell(
                image=_tensor_to_pil(image_tensor),
                target=targets[image_index],
                prediction=predictions[image_index],
                matching=matchings[image_index],
                index_to_class=index_to_class,
                sample_id=_sample_id(dataset, image_index),
                max_predictions_per_image=max_predictions_per_image,
            )
        )
    return _compose_grid(cells=cells, output_path=output_path, columns=1)


def _safe_value(value: Any, *, decimals: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{decimals}f}"


def _failure_cell(
    *,
    image: Image.Image,
    event: Mapping[str, Any],
    target: Mapping[str, Tensor],
    prediction: Mapping[str, Tensor],
    index_to_class: Mapping[int, str],
) -> Image.Image:
    source_width, source_height = image.size
    panel_size = _FAILURE_PANEL_SIZE
    cell_width = 2 * panel_size + _PANEL_GAP + 2 * _CELL_PADDING
    cell_height = (
        _HEADER_HEIGHT
        + 42
        + panel_size
        + _FAILURE_FOOTER_HEIGHT
        + 2 * _CELL_PADDING
    )
    cell = Image.new("RGB", (cell_width, cell_height), "white")
    draw = ImageDraw.Draw(cell)
    title_font = _load_font(22, bold=True)
    subtitle_font = _load_font(16)
    category = str(event["category"])
    sample_id = str(event["sample_id"])
    draw.text(
        (_CELL_PADDING, _CELL_PADDING),
        category.replace("_", " ").title(),
        fill="black",
        font=title_font,
    )
    draw.text(
        (_CELL_PADDING, _CELL_PADDING + 34),
        sample_id,
        fill="dimgray",
        font=subtitle_font,
    )

    left_x = _CELL_PADDING
    right_x = _CELL_PADDING + panel_size + _PANEL_GAP
    panel_y = _CELL_PADDING + _HEADER_HEIGHT
    left_panel = _resize_square(image, panel_size)
    right_panel = _resize_square(image, panel_size)
    _draw_panel_title(draw, x=left_x, y=panel_y, width=panel_size, title="Ground truth")
    _draw_panel_title(draw, x=right_x, y=panel_y, width=panel_size, title="Prediction")
    image_y = panel_y + 42

    ground_truth_index = event.get("ground_truth_index")
    prediction_index = event.get("prediction_index")
    if ground_truth_index is not None:
        gt_index = int(ground_truth_index)
        _draw_tagged_box(
            left_panel,
            box=target["boxes"][gt_index].tolist(),
            source_width=source_width,
            source_height=source_height,
            color=_BOX_COLORS["missed_ground_truth"],
            tag="GT",
            occupied_tags=[],
            width=6,
        )
    if prediction_index is not None:
        pred_index = int(prediction_index)
        _draw_tagged_box(
            right_panel,
            box=prediction["boxes"][pred_index].tolist(),
            source_width=source_width,
            source_height=source_height,
            color=_BOX_COLORS["failure_prediction"],
            tag="P",
            occupied_tags=[],
            width=6,
        )

    cell.paste(left_panel, (left_x, image_y))
    cell.paste(right_panel, (right_x, image_y))
    footer_y = image_y + panel_size + 8
    gt_class = event.get("ground_truth_class") or "-"
    pred_class = event.get("predicted_class") or "-"
    lines = [
        f"GT class: {gt_class}",
        f"Pred class: {pred_class}",
        f"Score: {_safe_value(event.get('score'))}",
        f"IoU: {_safe_value(event.get('iou'))}",
    ]
    _draw_text_lines(
        draw,
        lines=lines,
        x=left_x,
        y=footer_y,
        max_width_chars=38,
        max_lines=5,
        font_size=17,
    )
    legend_lines = [
        "Orange: associated/missed ground truth",
        "Magenta: failure prediction",
    ]
    _draw_text_lines(
        draw,
        lines=legend_lines,
        x=right_x,
        y=footer_y,
        max_width_chars=40,
        max_lines=4,
        font_size=17,
    )
    return cell


def create_detection_failure_montage(
    *,
    dataset: Any,
    predictions: Sequence[Mapping[str, Tensor]],
    targets: Sequence[Mapping[str, Tensor]],
    failure_analysis: Mapping[str, Any],
    index_to_class: Mapping[int, str],
    output_path: Path,
    max_images: int = 6,
) -> Path:
    """대표 실패 Event를 GT·Prediction 분리 Panel로 저장한다."""
    representatives = failure_analysis["representative_samples"]
    selected: list[Mapping[str, Any]] = []
    for category in (
        "wrong_class",
        "low_iou_localization",
        "duplicate_prediction",
        "low_confidence_correct_detection",
        "false_negative",
        "false_positive",
    ):
        values = representatives.get(category, [])
        if values:
            selected.append(values[0])
        if len(selected) >= max_images:
            break
    if not selected:
        raise ValueError("failure_analysis contains no representative failures.")

    cells: list[Image.Image] = []
    for event in selected:
        image_index = int(event["image_index"])
        image_tensor, _ = dataset[image_index]
        cells.append(
            _failure_cell(
                image=_tensor_to_pil(image_tensor),
                event=event,
                target=targets[image_index],
                prediction=predictions[image_index],
                index_to_class=index_to_class,
            )
        )
    return _compose_grid(cells=cells, output_path=output_path, columns=2)
