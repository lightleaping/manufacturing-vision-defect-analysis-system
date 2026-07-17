from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.api.detection_model_loader as loader
from src.api.detection_config import DetectionApiSettings
from src.api.detection_inference_service import DetectionInferenceService


CLASS_MAPPING = {
    "BACKGROUND": 0,
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
    "rolled_in_scale": 5,
    "scratches": 6,
}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, images):
        return []


def checkpoint_payload(model: nn.Module) -> dict:
    return {
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "training_config": {
            "min_size": 320,
            "max_size": 320,
            "score_threshold": 0.5,
            "iou_threshold": 0.5,
        },
        "class_mapping": CLASS_MAPPING,
        "best_metric": 0.677418,
    }


def test_loader_builds_random_architecture_then_restores_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    saved_model = TinyModel()
    payload = checkpoint_payload(saved_model)
    built_model = TinyModel()
    built_model.weight.data.zero_()
    captured_config = {}

    monkeypatch.setattr(
        loader,
        "load_detection_checkpoint_payload",
        lambda *args, **kwargs: payload,
    )

    def fake_create_detection_model(**kwargs):
        captured_config["config"] = kwargs["config"]
        return SimpleNamespace(
            model=built_model,
            metadata={"network_download_requested": False},
        )

    monkeypatch.setattr(loader, "create_detection_model", fake_create_detection_model)

    service = loader.create_production_detection_inference_service(
        settings=DetectionApiSettings(checkpoint_path=checkpoint)
    )

    assert isinstance(service, DetectionInferenceService)
    assert service.checkpoint_epoch == 3
    assert built_model.training is False
    assert torch.equal(built_model.weight, saved_model.weight)
    config = captured_config["config"]
    assert config.use_pretrained_weights is False
    assert config.use_pretrained_backbone is False
    assert config.num_classes == 7


def test_loader_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        loader.create_production_detection_inference_service(
            settings=DetectionApiSettings(
                checkpoint_path=tmp_path / "missing.pt"
            )
        )


def test_loader_rejects_class_mapping_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = TinyModel()
    payload = checkpoint_payload(model)
    payload["class_mapping"] = {"BACKGROUND": 0, "scratches": 1}

    monkeypatch.setattr(
        loader,
        "load_detection_checkpoint_payload",
        lambda *args, **kwargs: payload,
    )

    with pytest.raises(loader.DetectionModelLoadingError) as exc_info:
        loader.create_production_detection_inference_service(
            settings=DetectionApiSettings(checkpoint_path=checkpoint)
        )

    assert "class_mapping" in str(exc_info.value.__cause__)


def test_loader_rejects_state_dict_architecture_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = checkpoint_payload(TinyModel())
    payload["model_state_dict"] = {"unexpected": torch.tensor([1.0])}

    monkeypatch.setattr(
        loader,
        "load_detection_checkpoint_payload",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(
        loader,
        "create_detection_model",
        lambda **kwargs: SimpleNamespace(
            model=TinyModel(),
            metadata={"network_download_requested": False},
        ),
    )

    with pytest.raises(loader.DetectionModelLoadingError) as exc_info:
        loader.create_production_detection_inference_service(
            settings=DetectionApiSettings(checkpoint_path=checkpoint)
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
