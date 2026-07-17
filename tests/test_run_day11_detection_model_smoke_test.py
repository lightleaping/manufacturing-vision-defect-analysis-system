"""Day 11 Model Smoke Script의 Artifact와 Prediction Figure 연결을 검증한다."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from scripts import run_day11_detection_model_smoke_test as script
from src.detection.model_factory import DetectionModelBuildResult
from src.detection.model_validation import DetectionModelSmokeResult


class TinyDataset:
    def __init__(self) -> None:
        self.samples = [
            SimpleNamespace(
                record_id="train/crazing_1",
                image_path=Path("image.jpg").resolve(),
                annotation_path=Path("annotation.xml").resolve(),
            )
        ]
        self.index_to_class = {0: "background", 1: "crazing"}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        return (
            torch.zeros((3, 16, 16), dtype=torch.float32),
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "image_id": torch.tensor([0], dtype=torch.int64),
                "area": torch.tensor([64.0], dtype=torch.float32),
                "iscrowd": torch.tensor([0], dtype=torch.int64),
            },
        )


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))


def test_script_writes_pass_artifact_and_prediction_figure_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        script.NeuDetDetectionDataset,
        "from_manifest",
        lambda **kwargs: TinyDataset(),
    )
    monkeypatch.setattr(
        script,
        "create_detection_model",
        lambda **kwargs: DetectionModelBuildResult(
            model=DummyModel(),
            metadata={
                "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "device": "cpu",
                "network_download_requested": False,
                "predictor_output_classes": 7,
            },
        ),
    )
    monkeypatch.setattr(
        script,
        "run_detection_model_smoke_validation",
        lambda **kwargs: DetectionModelSmokeResult(
            payload={
                "training_forward": {
                    "losses": {
                        "loss_classifier": 1.0,
                        "loss_box_reg": 0.5,
                        "loss_objectness": 0.25,
                        "loss_rpn_box_reg": 0.125,
                    },
                    "total_loss": 1.875,
                    "elapsed_seconds": 0.01,
                },
                "evaluation_forward": {
                    "elapsed_seconds": 0.02,
                    "predictions": [{"box_count": 1}],
                },
                "validation_passed": True,
            }
        ),
    )
    monkeypatch.setattr(
        script,
        "capture_model_smoke_prediction",
        lambda **kwargs: {
            "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "scores": torch.tensor([0.5], dtype=torch.float32),
        },
    )

    def fake_save(*, output_path: Path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake png")
        return {
            "path": output_path.as_posix(),
            "decode_valid": True,
            "displayed_prediction_count": 1,
            "interpretation": "not trained detection performance",
        }

    monkeypatch.setattr(script, "save_model_smoke_prediction_figure", fake_save)

    payload = script.run_day11_detection_model_smoke_test(
        project_root=tmp_path,
        smoke_size=64,
        torch_num_threads=1,
    )

    artifact_path = (
        tmp_path
        / "reports"
        / "artifacts"
        / "day11_detection_model_smoke_test.json"
    )
    figure_path = (
        tmp_path
        / "reports"
        / "figures"
        / "day11_detection_model_predictions_smoke_test.png"
    )
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["validation_passed"] is True
    assert saved["validation_passed"] is True
    assert saved["execution_policy"]["pretrained_weight_download_executed"] is False
    assert saved["source_sample"]["record_id"] == "train/crazing_1"
    assert saved["artifacts"]["prediction_figure"]["decode_valid"] is True
    assert figure_path.is_file()
