"""가벼운 Fake Detection Model로 Forward 검증 로직을 테스트한다."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.detection.model_validation import run_detection_model_smoke_validation


class FakeDetectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            assert targets is not None
            zero = self.anchor * 0.0
            return {
                "loss_classifier": zero + 1.0,
                "loss_box_reg": zero + 0.5,
                "loss_objectness": zero + 0.25,
                "loss_rpn_box_reg": zero + 0.125,
            }
        outputs = []
        for image in images:
            outputs.append(
                {
                    "boxes": torch.tensor(
                        [[0.0, 0.0, float(image.shape[2]), float(image.shape[1])]],
                        dtype=torch.float32,
                    ),
                    "labels": torch.tensor([1], dtype=torch.int64),
                    "scores": torch.tensor([0.75], dtype=torch.float32),
                }
            )
        return outputs


def create_input() -> tuple[Tensor, dict[str, Tensor]]:
    image = torch.zeros((3, 16, 16), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor([0], dtype=torch.int64),
        "area": torch.tensor([64.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }
    return image, target


def test_smoke_validation_checks_train_and_eval_and_restores_mode() -> None:
    model = FakeDetectionModel()
    model.eval()
    image, target = create_input()
    original_image = image.clone()
    original_target = {key: value.clone() for key, value in target.items()}

    result = run_detection_model_smoke_validation(
        model=model,
        images=[image],
        targets=[target],
        num_classes=7,
    )

    assert result.payload["validation_passed"] is True
    assert result.payload["training_forward"]["loss_keys"] == [
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    ]
    assert result.payload["evaluation_forward"]["predictions"][0]["box_count"] == 1
    assert model.training is False
    assert torch.equal(image, original_image)
    assert all(torch.equal(target[key], original_target[key]) for key in target)


def test_smoke_validation_rejects_background_target_label() -> None:
    model = FakeDetectionModel()
    image, target = create_input()
    target["labels"] = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(ValueError, match="Target labels"):
        run_detection_model_smoke_validation(
            model=model,
            images=[image],
            targets=[target],
            num_classes=7,
        )


def test_smoke_validation_rejects_invalid_box() -> None:
    model = FakeDetectionModel()
    image, target = create_input()
    target["boxes"] = torch.tensor([[8.0, 0.0, 8.0, 8.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="xmax"):
        run_detection_model_smoke_validation(
            model=model,
            images=[image],
            targets=[target],
            num_classes=7,
        )
