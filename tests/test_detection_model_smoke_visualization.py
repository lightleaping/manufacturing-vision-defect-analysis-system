"""Day 11 Random-init Detection Smoke Prediction Figure를 검증한다."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import nn

from src.detection.model_smoke_visualization import (
    capture_model_smoke_prediction,
    save_model_smoke_prediction_figure,
)


class FakeEvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            raise AssertionError("capture must switch the model to eval mode")
        return [
            {
                "boxes": torch.tensor(
                    [[1.0, 2.0, 12.0, 14.0], [3.0, 3.0, 9.0, 10.0]],
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([1, 2], dtype=torch.int64),
                "scores": torch.tensor([0.8, 0.4], dtype=torch.float32),
            }
            for _ in images
        ]


def create_inputs():
    image = torch.zeros((3, 16, 16), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor([0], dtype=torch.int64),
        "area": torch.tensor([64.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }
    return image, target


def test_capture_returns_cpu_copy_and_restores_training_mode() -> None:
    model = FakeEvaluationModel()
    model.train()
    image, _ = create_inputs()
    original = image.clone()

    prediction = capture_model_smoke_prediction(model=model, image=image)

    assert model.training is True
    assert prediction["boxes"].device.type == "cpu"
    assert prediction["boxes"].shape == (2, 4)
    assert torch.equal(image, original)


def test_save_prediction_figure_creates_decodable_png(tmp_path: Path) -> None:
    image, target = create_inputs()
    prediction = capture_model_smoke_prediction(
        model=FakeEvaluationModel().eval(),
        image=image,
    )
    output_path = tmp_path / "prediction.png"

    metadata = save_model_smoke_prediction_figure(
        image=image,
        target=target,
        prediction=prediction,
        index_to_class={0: "background", 1: "crazing", 2: "inclusion"},
        output_path=output_path,
        max_predictions=1,
    )

    assert output_path.is_file()
    assert metadata["decode_valid"] is True
    assert metadata["displayed_prediction_count"] == 1
    assert "not trained detection performance" in metadata["interpretation"]
    with Image.open(output_path) as decoded:
        assert decoded.format == "PNG"
        assert decoded.width > decoded.height


def test_visualization_rejects_non_positive_max_predictions(tmp_path: Path) -> None:
    image, target = create_inputs()
    prediction = capture_model_smoke_prediction(
        model=FakeEvaluationModel().eval(),
        image=image,
    )
    with pytest.raises(ValueError, match="positive"):
        save_model_smoke_prediction_figure(
            image=image,
            target=target,
            prediction=prediction,
            index_to_class={1: "crazing", 2: "inclusion"},
            output_path=tmp_path / "prediction.png",
            max_predictions=0,
        )
