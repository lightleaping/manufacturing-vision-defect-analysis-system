from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.api.config import ApiSettings
from src.api.inference_service import ImageInferenceService
from src.api.model_loader import create_production_inference_service


class _DummyModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((inputs.shape[0],), dtype=torch.float32)


def test_production_loader_reuses_existing_checkpoint_and_test_transform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "resnet18_transfer_best.pt"
    checkpoint_path.write_bytes(b"checkpoint-placeholder")

    calls: dict[str, object] = {}
    dummy_model = _DummyModel()

    def fake_restore_best_checkpoint(*, checkpoint_path: Path, device: torch.device):
        calls["checkpoint_path"] = checkpoint_path
        calls["device"] = device
        return dummy_model

    def fake_create_test_transform():
        calls["transform_created"] = True

        def transform(_: Image.Image) -> torch.Tensor:
            return torch.zeros((3, 224, 224), dtype=torch.float32)

        return transform

    monkeypatch.setattr(
        "src.api.model_loader.restore_best_checkpoint",
        fake_restore_best_checkpoint,
    )
    monkeypatch.setattr(
        "src.api.model_loader.create_test_transform",
        fake_create_test_transform,
    )

    service = create_production_inference_service(
        settings=ApiSettings(checkpoint_path=checkpoint_path),
    )

    assert isinstance(service, ImageInferenceService)
    assert calls["checkpoint_path"] == checkpoint_path.resolve()
    assert calls["device"] == torch.device("cpu")
    assert calls["transform_created"] is True
    assert service.model is dummy_model
    assert service.model.training is False


def test_production_loader_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Best Checkpoint"):
        create_production_inference_service(
            settings=ApiSettings(
                checkpoint_path=tmp_path / "missing.pt",
            )
        )
