"""Detection collate_fn과 Windows CPU DataLoader 설정 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.detection.data_loader import (
    DetectionDataLoaderConfig,
    create_detection_data_loader,
    detection_collate_fn,
)
from src.detection.detection_dataset import NeuDetDetectionDataset
from src.detection.transforms import create_detection_transform
from tests.detection_test_helpers import create_sample


def test_collate_fn_preserves_variable_box_counts() -> None:
    batch = [
        (
            torch.zeros((3, 8, 10), dtype=torch.float32),
            {"boxes": torch.zeros((1, 4), dtype=torch.float32)},
        ),
        (
            torch.ones((3, 8, 10), dtype=torch.float32),
            {"boxes": torch.zeros((3, 4), dtype=torch.float32)},
        ),
    ]

    images, targets = detection_collate_fn(batch)  # type: ignore[arg-type]

    assert len(images) == 2
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (3, 4)


def create_two_sample_dataset(tmp_path: Path) -> NeuDetDetectionDataset:
    sample_a = create_sample(
        tmp_path,
        record_id="sample_a",
        boxes=(("crazing", (1, 1, 10, 8)),),
    )
    sample_b = create_sample(
        tmp_path,
        record_id="sample_b",
        boxes=(
            ("inclusion", (1, 1, 4, 4)),
            ("scratches", (5, 2, 10, 8)),
        ),
    )
    return NeuDetDetectionDataset(
        samples=[sample_a, sample_b],
        transform=create_detection_transform(training=False),
    )


def test_cpu_data_loader_returns_image_and_target_tuples(tmp_path: Path) -> None:
    dataset = create_two_sample_dataset(tmp_path)
    config = DetectionDataLoaderConfig(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    loader = create_detection_data_loader(
        dataset=dataset,
        shuffle=False,
        config=config,
    )

    images, targets = next(iter(loader))

    assert isinstance(images, tuple)
    assert isinstance(targets, tuple)
    assert len(images) == 2
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (2, 4)
    assert loader.num_workers == 0
    assert loader.pin_memory is False
    assert loader.drop_last is False


def test_loader_without_shuffle_preserves_dataset_order(tmp_path: Path) -> None:
    dataset = create_two_sample_dataset(tmp_path)
    loader = create_detection_data_loader(
        dataset=dataset,
        shuffle=False,
        config=DetectionDataLoaderConfig(batch_size=1),
    )

    observed_image_ids = [
        int(targets[0]["image_id"].item())
        for _, targets in loader
    ]
    assert observed_image_ids == [0, 1]


def test_persistent_workers_requires_worker_process() -> None:
    with pytest.raises(ValueError, match="num_workers > 0"):
        DetectionDataLoaderConfig(
            num_workers=0,
            persistent_workers=True,
        )
