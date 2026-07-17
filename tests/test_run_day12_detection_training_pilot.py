"""실제 Weight 다운로드 없이 Day 12 Pilot Script 계약을 테스트한다."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import scripts.run_day12_detection_training_pilot as script


class TinyDataset:
    def __init__(self, split: str) -> None:
        self.split = split
        self.samples = [SimpleNamespace(record_id=f"{split}/sample_1")]
        self.index_to_class = {0: "BACKGROUND", 1: "crazing"}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return (
            torch.zeros((3, 16, 16), dtype=torch.float32),
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "image_id": torch.tensor([index], dtype=torch.int64),
                "area": torch.tensor([64.0], dtype=torch.float32),
                "iscrowd": torch.tensor([0], dtype=torch.int64),
            },
        )


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            base = self.value.square()
            return {
                "loss_classifier": base,
                "loss_box_reg": base * 0.5,
                "loss_objectness": base * 0.25,
                "loss_rpn_box_reg": base * 0.125,
            }
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
            for _ in images
        ]


def test_pilot_requires_explicit_download_permission_when_cache_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(script, "_expected_weight_path", lambda: tmp_path / "missing.pth")

    try:
        script.run_day12_detection_training_pilot(
            project_root=tmp_path,
            allow_pretrained_download=False,
        )
    except RuntimeError as error:
        assert "--allow-pretrained-download" in str(error)
    else:
        raise AssertionError("Pilot should require explicit download permission.")


def test_pilot_writes_artifact_and_checkpoint_without_real_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    fake_weight = tmp_path / "cached.pth"
    fake_weight.write_bytes(b"weight")
    monkeypatch.setattr(script, "_expected_weight_path", lambda: fake_weight)
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
                "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "predictor_output_classes": 7,
                "network_download_requested": True,
            },
        ),
    )

    payload = script.run_day12_detection_training_pilot(
        project_root=tmp_path,
        allow_pretrained_download=False,
        train_samples=1,
        validation_samples=1,
        overfit_steps=2,
        torch_num_threads=1,
    )

    assert payload["validation_passed"] is True
    assert payload["execution_policy"]["full_training_executed"] is False
    assert (tmp_path / script.DEFAULT_PILOT_ARTIFACT).is_file()
    assert (tmp_path / script.DEFAULT_PILOT_CHECKPOINT).is_file()
