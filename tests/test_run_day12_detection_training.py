from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import SGD

import scripts.run_day12_detection_training as script
from src.detection.checkpoint import (
    build_detection_checkpoint_payload,
    load_detection_checkpoint_payload,
    save_detection_checkpoint,
)
from src.detection.epoch_runner import (
    build_detection_checkpoint_class_mapping,
)
from src.detection.model_config import DetectionModelConfig


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, split: str) -> None:
        self.split = split

    def __len__(self) -> int:
        return 2 if self.split == "train" else 1

    def __getitem__(self, index: int):
        image = torch.zeros((3, 16, 16), dtype=torch.float32)
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        return image, target


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.head = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            base = (self.head - 0.25).pow(2)
            return {
                "loss_classifier": base,
                "loss_box_reg": base * 0.5,
                "loss_objectness": base * 0.25,
                "loss_rpn_box_reg": base * 0.125,
            }
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9]),
            }
            for _ in images
        ]


def _write_seed_checkpoint(project_root: Path) -> None:
    model = TinyModel()
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
    class_mapping = build_detection_checkpoint_class_mapping(
        DetectionModelConfig().index_to_class
    )
    payload = build_detection_checkpoint_payload(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        training_config={"epochs": 3, "learning_rate": 0.005},
        class_mapping=class_mapping,
        best_metric=0.0,
        history=[{"epoch": 0, "stage": "seed"}],
    )
    save_detection_checkpoint(
        payload=payload,
        latest_path=project_root / script.DEFAULT_LATEST_CHECKPOINT,
        best_path=project_root / script.DEFAULT_BEST_CHECKPOINT,
        is_best=True,
    )


def test_resumed_training_writes_epoch_one_checkpoint_and_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    cached_weight = tmp_path / "weight.pth"
    cached_weight.write_bytes(b"weight")
    _write_seed_checkpoint(tmp_path)

    monkeypatch.setattr(script, "_expected_weight_path", lambda: cached_weight)
    monkeypatch.setattr(
        script.NeuDetDetectionDataset,
        "from_manifest",
        lambda **kwargs: TinyDataset(kwargs["split"]),
    )
    monkeypatch.setattr(
        script,
        "create_detection_model",
        lambda **kwargs: SimpleNamespace(
            model=TinyModel(),
            metadata={
                "architecture": "tiny",
                "predictor_output_classes": 7,
            },
        ),
    )

    result = script.run_day12_detection_training(
        project_root=tmp_path,
        target_total_epochs=2,
        unfreeze_learning_rate=0.001,
        torch_num_threads=1,
        log_interval=1,
    )

    assert result["validation_passed"] is True
    assert result["execution_policy"]["completed_epoch_indexes"] == [1]
    assert result["execution_policy"]["test_split_used"] is False
    assert result["unfreeze"]["backbone_trainable"] is True
    assert (tmp_path / script.DEFAULT_ARTIFACT).is_file()

    latest = load_detection_checkpoint_payload(
        tmp_path / script.DEFAULT_LATEST_CHECKPOINT
    )
    assert latest["epoch"] == 1
    assert len(latest["history"]) == 2
    assert latest["training_config"]["learning_rate"] == 0.001
