from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch
from torch.utils.data import Dataset

import scripts.regenerate_day12_detection_figures as script


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [SimpleNamespace(record_id="test/crazing_1")]
        self.images = [torch.zeros((3, 40, 40), dtype=torch.float32)]
        self.targets = [
            {
                "boxes": torch.tensor([[5, 5, 25, 25]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        ]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return self.images[index].clone(), {
            key: value.clone() for key, value in self.targets[index].items()
        }


class TinyModel:
    def load_state_dict(self, state_dict, strict=True):
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def eval(self):
        return self


def _write_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), "white").save(path)
    return path


def test_regeneration_preserves_json_and_writes_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    checkpoint = tmp_path / "models" / "detection" / "day12_detection_best.pt"
    artifact_dir = tmp_path / "reports" / "artifacts"
    manifest.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")

    overall = {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "map_50": 1.0,
    }
    evaluation = {
        "evaluation_policy": {"score_threshold": 0.5, "iou_threshold": 0.5},
        "test": {
            "metrics": {
                "overall": overall,
                "class_metrics": {
                    "crazing": {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "ap_50": 1.0,
                    }
                },
            }
        },
    }
    failure = {"analysis": {"summary": {"event_count": 0}}}
    evaluation_path = artifact_dir / "day12_detection_evaluation.json"
    failure_path = artifact_dir / "day12_detection_failure_analysis.json"
    evaluation_text = json.dumps(evaluation)
    failure_text = json.dumps(failure)
    evaluation_path.write_text(evaluation_text, encoding="utf-8")
    failure_path.write_text(failure_text, encoding="utf-8")

    fake_weight = tmp_path / "weight.pth"
    fake_weight.write_bytes(b"weight")
    dataset = TinyDataset()
    predictions = [
        {
            "boxes": torch.tensor([[5, 5, 25, 25]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "scores": torch.tensor([0.9], dtype=torch.float32),
        }
    ]
    fake_summary = {
        "metrics": {
            "overall": overall,
            "class_metrics": evaluation["test"]["metrics"]["class_metrics"],
        }
    }
    fake_result = SimpleNamespace(
        predictions=predictions,
        targets=dataset.targets,
        summary=lambda: fake_summary,
    )

    monkeypatch.setattr(script, "_expected_weight_path", lambda: fake_weight)
    monkeypatch.setattr(
        script.NeuDetDetectionDataset,
        "from_manifest",
        lambda **kwargs: dataset,
    )
    monkeypatch.setattr(
        script,
        "load_detection_checkpoint_payload",
        lambda *args, **kwargs: {
            "epoch": 2,
            "model_state_dict": {},
            "history": [
                {
                    "epoch": 0,
                    "train": {"average_losses": {"total_loss": 1.0}},
                    "validation": {
                        "metrics": {"overall": {"map_50": 0.5, "f1": 0.4}}
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(
        script,
        "create_detection_model",
        lambda **kwargs: SimpleNamespace(model=TinyModel()),
    )
    monkeypatch.setattr(
        script,
        "run_detection_evaluation_epoch",
        lambda **kwargs: fake_result,
    )
    monkeypatch.setattr(
        script,
        "analyze_detection_failures",
        lambda **kwargs: {
            "summary": {"event_count": 0},
            "representative_samples": {},
        },
    )
    monkeypatch.setattr(
        script,
        "plot_detection_training_history",
        lambda **kwargs: _write_png(kwargs["output_path"]),
    )
    monkeypatch.setattr(
        script,
        "plot_detection_class_metrics",
        lambda **kwargs: _write_png(kwargs["output_path"]),
    )
    monkeypatch.setattr(
        script,
        "create_detection_prediction_montage",
        lambda **kwargs: _write_png(kwargs["output_path"]),
    )
    monkeypatch.setattr(
        script,
        "create_detection_failure_montage",
        lambda **kwargs: _write_png(kwargs["output_path"]),
    )

    payload = script.regenerate_day12_detection_figures(
        project_root=tmp_path,
        torch_num_threads=1,
        log_interval=1,
    )

    assert payload["result"] == "PASS"
    assert evaluation_path.read_text(encoding="utf-8") == evaluation_text
    assert failure_path.read_text(encoding="utf-8") == failure_text
    for relative in script.FIGURE_PATHS.values():
        assert (tmp_path / relative).is_file()
