from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset

import scripts.run_day12_detection_evaluation as script
from src.detection.metrics import calculate_detection_metrics


INDEX_TO_CLASS = {0: "background", 1: "crazing", 2: "inclusion", 3: "patches", 4: "pitted_surface", 5: "rolled_in_scale", 6: "scratches"}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class TinyDataset(Dataset):
    def __init__(self, split: str) -> None:
        self.split = split
        self.samples = [SimpleNamespace(record_id=f"{split}/sample_1")]
        self.image = torch.zeros((3, 20, 20), dtype=torch.float32)
        self.target = {
            "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return self.image.clone(), {
            key: value.clone() for key, value in self.target.items()
        }


class FakeEvaluation:
    def __init__(self, split: str) -> None:
        self.split = split
        self.predictions = (
            {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            },
        )
        self.targets = (
            {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
        )
        self.metrics = calculate_detection_metrics(
            predictions=self.predictions,
            targets=self.targets,
            index_to_class=INDEX_TO_CLASS,
        )

    def summary(self):
        return {
            "split": self.split,
            "batch_count": 1,
            "sample_count": 1,
            "prediction_box_count": 1,
            "elapsed_seconds": 0.01,
            "average_batch_seconds": 0.01,
            "all_inputs_unchanged": True,
            "metrics": self.metrics,
        }


def test_final_evaluation_writes_artifacts_without_real_model(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    best = tmp_path / "models" / "detection" / "day12_detection_best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"checkpoint")
    cached = tmp_path / "cached.pth"
    cached.write_bytes(b"weight")

    model = TinyModel()
    checkpoint_payload = {
        "epoch": 2,
        "best_metric": 0.67,
        "class_mapping": {
            "BACKGROUND": 0,
            "crazing": 1,
            "inclusion": 2,
            "patches": 3,
            "pitted_surface": 4,
            "rolled_in_scale": 5,
            "scratches": 6,
        },
        "model_state_dict": model.state_dict(),
        "history": [
            {
                "epoch": 0,
                "train": {"average_losses": {"total_loss": 0.7}},
                "validation": {"metrics": {"overall": {"map_50": 0.4, "f1": 0.3}}},
            }
        ],
    }

    monkeypatch.setattr(script, "_expected_weight_path", lambda: cached)
    monkeypatch.setattr(
        script,
        "load_detection_checkpoint_payload",
        lambda *args, **kwargs: checkpoint_payload,
    )
    monkeypatch.setattr(
        script.NeuDetDetectionDataset,
        "from_manifest",
        classmethod(lambda cls, **kwargs: TinyDataset(kwargs["split"])),
    )
    monkeypatch.setattr(
        script,
        "create_detection_model",
        lambda **kwargs: SimpleNamespace(model=TinyModel(), metadata={"architecture": "tiny"}),
    )
    monkeypatch.setattr(
        script,
        "run_detection_evaluation_epoch",
        lambda **kwargs: FakeEvaluation(kwargs["split"]),
    )

    def touch_plot(*, output_path: Path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return output_path

    monkeypatch.setattr(script, "plot_detection_training_history", touch_plot)
    monkeypatch.setattr(script, "plot_detection_class_metrics", touch_plot)
    monkeypatch.setattr(script, "create_detection_prediction_montage", touch_plot)
    monkeypatch.setattr(script, "create_detection_failure_montage", touch_plot)

    payload = script.run_day12_detection_evaluation(
        project_root=tmp_path,
        torch_num_threads=1,
        log_interval=1,
    )

    assert payload["evaluation_policy"]["test_split_used"] is True
    assert payload["evaluation_policy"]["test_result_used_for_model_selection"] is False
    assert payload["test"]["metrics"]["overall"]["map_50"] == 1.0
    assert (tmp_path / script.DEFAULT_EVALUATION_ARTIFACT).is_file()
    assert (tmp_path / script.DEFAULT_FAILURE_ARTIFACT).is_file()
