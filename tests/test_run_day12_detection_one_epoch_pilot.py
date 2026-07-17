from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import scripts.run_day12_detection_one_epoch_pilot as script


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


def test_one_epoch_pilot_writes_latest_best_and_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    cached_weight = tmp_path / "weight.pth"
    cached_weight.write_bytes(b"weight")

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

    payload = script.run_day12_detection_one_epoch_pilot(
        project_root=tmp_path,
        torch_num_threads=1,
        log_interval=1,
    )

    assert payload["validation_passed"] is True
    assert payload["execution_policy"]["test_split_used"] is False
    assert (tmp_path / script.DEFAULT_ARTIFACT).is_file()
    assert (tmp_path / script.DEFAULT_LATEST_CHECKPOINT).is_file()
    assert (tmp_path / script.DEFAULT_BEST_CHECKPOINT).is_file()
    assert payload["validation"]["metrics"]["overall"]["map_50"] == 1.0
