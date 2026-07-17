"""Day 11 Detection Model Smoke Prediction을 성능 주장 없이 시각화한다.

[신규 구현]
- 실제 NEU-DET 이미지의 Ground Truth와 Random Initialization Prediction을 비교한다.
- Prediction Figure가 학습 성능으로 오해되지 않도록 해석 경고를 Metadata에 기록한다.
- Model Mode와 입력 Tensor를 변경하지 않는 별도 Evaluation Forward를 수행한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor, nn


_REQUIRED_PREDICTION_KEYS = frozenset({"boxes", "labels", "scores"})
_BACKGROUND = (245, 245, 245)
_GT_COLOR = (210, 35, 35)
_PREDICTION_COLOR = (30, 90, 210)
_TEXT_COLOR = (25, 25, 25)


def _tensor_to_pil(image: Tensor) -> Image.Image:
    if not isinstance(image, Tensor):
        raise TypeError("image must be torch.Tensor.")
    if image.dtype != torch.float32 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be FloatTensor[3, H, W].")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image contains non-finite values.")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError("image values must be in [0, 1].")

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


def _clone_prediction(prediction: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not isinstance(prediction, Mapping):
        raise TypeError("prediction must be a mapping.")
    missing = _REQUIRED_PREDICTION_KEYS - set(prediction)
    if missing:
        raise KeyError(f"prediction is missing keys: {sorted(missing)}.")

    output: dict[str, Tensor] = {}
    for key in _REQUIRED_PREDICTION_KEYS:
        value = prediction[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"prediction[{key!r}] must be torch.Tensor.")
        output[key] = value.detach().cpu().clone()

    boxes = output["boxes"]
    labels = output["labels"]
    scores = output["scores"]
    if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("prediction boxes must be FloatTensor[N, 4].")
    if labels.dtype != torch.int64 or labels.ndim != 1:
        raise ValueError("prediction labels must be Int64Tensor[N].")
    if scores.dtype != torch.float32 or scores.ndim != 1:
        raise ValueError("prediction scores must be FloatTensor[N].")
    if not (boxes.shape[0] == labels.shape[0] == scores.shape[0]):
        raise ValueError("prediction boxes, labels and scores counts must match.")
    if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
        raise ValueError("prediction contains non-finite values.")
    return output


def capture_model_smoke_prediction(
    *,
    model: nn.Module,
    image: Tensor,
    device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Model Evaluation Forward 결과 한 개를 CPU Tensor로 복사해 반환한다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise ValueError("Day 11 Smoke Prediction capture supports CPU only.")

    original_mode = bool(model.training)
    original_image = image.clone()
    try:
        model.eval()
        with torch.no_grad():
            predictions = model([image.to(resolved_device)])
    finally:
        model.train(mode=original_mode)

    if not isinstance(predictions, (list, tuple)) or len(predictions) != 1:
        raise ValueError("model evaluation must return one prediction mapping.")
    if not torch.equal(image, original_image):
        raise RuntimeError("model evaluation modified the source image tensor.")
    return _clone_prediction(predictions[0])


def _draw_box_label(
    *,
    draw: ImageDraw.ImageDraw,
    box: list[float],
    text: str,
    color: tuple[int, int, int],
    image_width: int,
    image_height: int,
) -> None:
    font = ImageFont.load_default()
    xmin, ymin, xmax, ymax = box
    xmin = min(max(float(xmin), 0.0), float(image_width))
    ymin = min(max(float(ymin), 0.0), float(image_height))
    xmax = min(max(float(xmax), 0.0), float(image_width))
    ymax = min(max(float(ymax), 0.0), float(image_height))
    if xmax <= xmin or ymax <= ymin:
        return

    draw.rectangle(
        (xmin, ymin, max(xmin, xmax - 1), max(ymin, ymax - 1)),
        outline=color,
        width=2,
    )
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_y = max(0, int(ymin) - text_height - 3)
    draw.rectangle(
        (
            int(xmin),
            text_y,
            min(image_width - 1, int(xmin) + text_width + 4),
            min(image_height - 1, text_y + text_height + 3),
        ),
        fill=color,
    )
    draw.text((int(xmin) + 2, text_y + 1), text, fill=(255, 255, 255), font=font)


def _draw_ground_truth(
    *,
    image: Tensor,
    target: Mapping[str, Tensor],
    index_to_class: Mapping[int, str],
) -> Image.Image:
    canvas = _tensor_to_pil(image)
    boxes = target.get("boxes")
    labels = target.get("labels")
    if not isinstance(boxes, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("target must contain Tensor boxes and labels.")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.ndim != 1:
        raise ValueError("target boxes or labels have invalid shapes.")
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError("target box and label counts must match.")

    draw = ImageDraw.Draw(canvas)
    for box, label in zip(boxes.detach().cpu().tolist(), labels.detach().cpu().tolist()):
        label_index = int(label)
        _draw_box_label(
            draw=draw,
            box=[float(value) for value in box],
            text=index_to_class.get(label_index, f"label_{label_index}"),
            color=_GT_COLOR,
            image_width=canvas.width,
            image_height=canvas.height,
        )
    return canvas


def _draw_predictions(
    *,
    image: Tensor,
    prediction: Mapping[str, Tensor],
    index_to_class: Mapping[int, str],
    max_predictions: int,
) -> tuple[Image.Image, int]:
    if not isinstance(max_predictions, int) or isinstance(max_predictions, bool):
        raise TypeError("max_predictions must be int.")
    if max_predictions <= 0:
        raise ValueError("max_predictions must be positive.")

    validated = _clone_prediction(prediction)
    canvas = _tensor_to_pil(image)
    draw = ImageDraw.Draw(canvas)
    box_count = int(validated["boxes"].shape[0])
    selected_count = min(box_count, max_predictions)
    if selected_count:
        order = torch.argsort(validated["scores"], descending=True)[:selected_count]
        for prediction_index in order.tolist():
            box = validated["boxes"][prediction_index].tolist()
            label_index = int(validated["labels"][prediction_index].item())
            score = float(validated["scores"][prediction_index].item())
            class_name = index_to_class.get(label_index, f"label_{label_index}")
            _draw_box_label(
                draw=draw,
                box=[float(value) for value in box],
                text=f"{class_name} {score:.3f}",
                color=_PREDICTION_COLOR,
                image_width=canvas.width,
                image_height=canvas.height,
            )
    return canvas, selected_count


def _paste_panel(
    *,
    figure: Image.Image,
    panel_index: int,
    panel_width: int,
    panel_height: int,
    title: str,
    content: Image.Image,
) -> None:
    left = panel_index * panel_width
    draw = ImageDraw.Draw(figure)
    font = ImageFont.load_default()
    draw.text((left + 10, 8), title, fill=_TEXT_COLOR, font=font)

    available_width = panel_width - 20
    available_height = panel_height - 48
    scale = min(available_width / content.width, available_height / content.height)
    resized = content.resize(
        (
            max(1, int(round(content.width * scale))),
            max(1, int(round(content.height * scale))),
        ),
        Image.Resampling.NEAREST,
    )
    x = left + (panel_width - resized.width) // 2
    y = 34 + (available_height - resized.height) // 2
    figure.paste(resized, (x, y))


def save_model_smoke_prediction_figure(
    *,
    image: Tensor,
    target: Mapping[str, Tensor],
    prediction: Mapping[str, Tensor],
    index_to_class: Mapping[int, str],
    output_path: Path,
    max_predictions: int = 10,
) -> dict[str, Any]:
    """Original·Ground Truth·Random-init Prediction 비교 Figure를 저장한다."""
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path.")
    original = _tensor_to_pil(image)
    ground_truth = _draw_ground_truth(
        image=image,
        target=target,
        index_to_class=index_to_class,
    )
    prediction_overlay, displayed_prediction_count = _draw_predictions(
        image=image,
        prediction=prediction,
        index_to_class=index_to_class,
        max_predictions=max_predictions,
    )

    panel_width = 360
    panel_height = 300
    footer_height = 34
    figure = Image.new(
        "RGB",
        (panel_width * 3, panel_height + footer_height),
        color=_BACKGROUND,
    )
    _paste_panel(
        figure=figure,
        panel_index=0,
        panel_width=panel_width,
        panel_height=panel_height,
        title="Original NEU-DET sample",
        content=original,
    )
    _paste_panel(
        figure=figure,
        panel_index=1,
        panel_width=panel_width,
        panel_height=panel_height,
        title="Ground truth target",
        content=ground_truth,
    )
    _paste_panel(
        figure=figure,
        panel_index=2,
        panel_width=panel_width,
        panel_height=panel_height,
        title="Random-init model smoke prediction",
        content=prediction_overlay,
    )
    footer = ImageDraw.Draw(figure)
    footer.text(
        (10, panel_height + 10),
        "Structure validation only; not trained detection performance.",
        fill=_TEXT_COLOR,
        font=ImageFont.load_default(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output_path, format="PNG")
    with Image.open(output_path) as saved:
        width, height = saved.size
        image_format = saved.format

    return {
        "path": output_path.as_posix(),
        "semantic_role": "random_initialization_detection_smoke_prediction",
        "size_bytes": output_path.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "decode_valid": True,
        "displayed_prediction_count": displayed_prediction_count,
        "interpretation": (
            "Random-initialization model forward structure validation; "
            "not trained detection performance or a Day 12 evaluation result."
        ),
    }
