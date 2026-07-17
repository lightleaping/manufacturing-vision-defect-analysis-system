from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.detection.failure_analysis import analyze_detection_failures
from src.detection.training_visualization import (
    create_detection_failure_montage,
    create_detection_prediction_montage,
    plot_detection_class_metrics,
    plot_detection_training_history,
)


INDEX_TO_CLASS = {0: "background", 1: "crazing", 2: "inclusion"}


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [
            SimpleNamespace(record_id="test/crazing_1"),
            SimpleNamespace(record_id="test/inclusion_1"),
        ]
        self.images = [
            torch.zeros((3, 40, 40), dtype=torch.float32),
            torch.ones((3, 40, 40), dtype=torch.float32),
        ]
        self.targets = [
            {
                "boxes": torch.tensor([[5, 5, 25, 25]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
            {
                "boxes": torch.tensor([[10, 10, 30, 30]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.int64),
            },
        ]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return self.images[index].clone(), {
            key: value.clone() for key, value in self.targets[index].items()
        }


def _predictions():
    return [
        {
            "boxes": torch.tensor(
                [[5, 5, 25, 25], [7, 7, 22, 22]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1, 1], dtype=torch.int64),
            "scores": torch.tensor([0.95, 0.70], dtype=torch.float32),
        },
        {
            "boxes": torch.tensor([[10, 10, 30, 30]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "scores": torch.tensor([0.90], dtype=torch.float32),
        },
    ]


def test_training_history_and_class_metric_figures_are_large_and_written(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.png"
    class_path = tmp_path / "classes.png"
    history = [
        {
            "epoch": 0,
            "train": {"average_losses": {"total_loss": 0.9}},
            "validation": {
                "metrics": {"overall": {"map_50": 0.4, "f1": 0.3}}
            },
        },
        {
            "epoch": 1,
            "train": {"average_losses": {"total_loss": 0.8}},
            "validation": {
                "metrics": {"overall": {"map_50": 0.6, "f1": 0.5}}
            },
        },
    ]
    class_metrics = {
        "rolled_in_scale": {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "ap_50": 0.75,
        },
        "pitted_surface": {
            "precision": 0.6,
            "recall": 0.5,
            "f1": 0.55,
            "ap_50": 0.55,
        },
    }

    plot_detection_training_history(history=history, output_path=history_path)
    plot_detection_class_metrics(
        class_metrics=class_metrics,
        output_path=class_path,
    )

    with Image.open(history_path) as history_image:
        assert history_image.width >= 1200
        assert history_image.height >= 800
    with Image.open(class_path) as class_image:
        assert class_image.width >= 1400
        assert class_image.height >= 900


def test_prediction_and_failure_montages_use_readable_large_layout(
    tmp_path: Path,
) -> None:
    dataset = TinyDataset()
    predictions = _predictions()
    targets = dataset.targets
    analysis = analyze_detection_failures(
        predictions=predictions,
        targets=targets,
        index_to_class=INDEX_TO_CLASS,
        sample_ids=[sample.record_id for sample in dataset.samples],
    )
    prediction_path = tmp_path / "predictions.png"
    failure_path = tmp_path / "failures.png"

    create_detection_prediction_montage(
        dataset=dataset,
        predictions=predictions,
        targets=targets,
        index_to_class=INDEX_TO_CLASS,
        output_path=prediction_path,
    )
    create_detection_failure_montage(
        dataset=dataset,
        predictions=predictions,
        targets=targets,
        failure_analysis=analysis,
        index_to_class=INDEX_TO_CLASS,
        output_path=failure_path,
    )

    with Image.open(prediction_path) as prediction_image:
        assert prediction_image.width >= 1000
        assert prediction_image.height >= 1400
    with Image.open(failure_path) as failure_image:
        assert failure_image.width >= 1500
        assert failure_image.height >= 600
