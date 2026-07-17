"""가변 Bounding Box 개수를 보존하는 Detection DataLoader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.detection.detection_dataset import NeuDetDetectionDataset
from src.detection.model_config import DEFAULT_DEFECT_CLASS_NAMES
from src.detection.transforms import DetectionTarget, create_detection_transform


def detection_collate_fn(
    batch: list[tuple[Tensor, DetectionTarget]],
) -> tuple[tuple[Tensor, ...], tuple[DetectionTarget, ...]]:
    """이미지별 Box 수가 달라도 Stack하지 않고 Tuple로 묶는다."""
    if not isinstance(batch, list):
        raise TypeError("batch must be list.")
    if not batch:
        raise ValueError("batch must not be empty.")
    if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in batch):
        raise TypeError("Every batch item must be (image, target).")

    images, targets = zip(*batch)
    if any(not isinstance(image, Tensor) for image in images):
        raise TypeError("Every collated image must be torch.Tensor.")
    if any(not isinstance(target, dict) for target in targets):
        raise TypeError("Every collated target must be dict.")
    return tuple(images), tuple(targets)


@dataclass(frozen=True, slots=True)
class DetectionDataLoaderConfig:
    """현재 Windows CPU 환경의 보수적인 DataLoader 기본값."""

    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise TypeError("batch_size must be int.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not isinstance(self.num_workers, int) or isinstance(
            self.num_workers, bool
        ):
            raise TypeError("num_workers must be int.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed, bool
        ):
            raise TypeError("random_seed must be int.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")

        for field_name in ("pin_memory", "drop_last", "persistent_workers"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool.")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers=True requires num_workers > 0."
            )


def create_detection_data_loader(
    *,
    dataset: Dataset[Any],
    shuffle: bool,
    config: DetectionDataLoaderConfig | None = None,
) -> DataLoader[Any]:
    """재현 가능한 Detection DataLoader를 만든다."""
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must inherit torch.utils.data.Dataset.")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be bool.")
    resolved_config = config or DetectionDataLoaderConfig()
    if not isinstance(resolved_config, DetectionDataLoaderConfig):
        raise TypeError("config must be DetectionDataLoaderConfig.")

    generator = torch.Generator()
    generator.manual_seed(resolved_config.random_seed)

    return DataLoader(
        dataset,
        batch_size=resolved_config.batch_size,
        shuffle=shuffle,
        num_workers=resolved_config.num_workers,
        pin_memory=resolved_config.pin_memory,
        drop_last=resolved_config.drop_last,
        persistent_workers=resolved_config.persistent_workers,
        collate_fn=detection_collate_fn,
        generator=generator,
    )


@dataclass(frozen=True, slots=True)
class DetectionDataLoaders:
    train_dataset: NeuDetDetectionDataset
    validation_dataset: NeuDetDetectionDataset
    test_dataset: NeuDetDetectionDataset
    train_loader: DataLoader[Any]
    validation_loader: DataLoader[Any]
    test_loader: DataLoader[Any]


def create_neu_det_detection_data_loaders(
    *,
    project_root: Path,
    manifest_path: Path,
    config: DetectionDataLoaderConfig | None = None,
    duplicate_box_policy: str = "preserve",
    training_horizontal_flip_probability: float = 0.0,
) -> DetectionDataLoaders:
    """Day 9 Manifest로 Train·Validation·Test Dataset과 Loader를 만든다."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    if not isinstance(manifest_path, Path):
        raise TypeError("manifest_path must be pathlib.Path.")

    resolved_config = config or DetectionDataLoaderConfig()
    train_dataset = NeuDetDetectionDataset.from_manifest(
        manifest_path=manifest_path,
        project_root=project_root,
        split="train",
        transform=create_detection_transform(
            training=True,
            horizontal_flip_probability=training_horizontal_flip_probability,
        ),
        class_names=DEFAULT_DEFECT_CLASS_NAMES,
        duplicate_box_policy=duplicate_box_policy,
    )
    validation_dataset = NeuDetDetectionDataset.from_manifest(
        manifest_path=manifest_path,
        project_root=project_root,
        split="validation",
        transform=create_detection_transform(training=False),
        class_names=DEFAULT_DEFECT_CLASS_NAMES,
        duplicate_box_policy=duplicate_box_policy,
    )
    test_dataset = NeuDetDetectionDataset.from_manifest(
        manifest_path=manifest_path,
        project_root=project_root,
        split="test",
        transform=create_detection_transform(training=False),
        class_names=DEFAULT_DEFECT_CLASS_NAMES,
        duplicate_box_policy=duplicate_box_policy,
    )

    return DetectionDataLoaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=create_detection_data_loader(
            dataset=train_dataset,
            shuffle=True,
            config=resolved_config,
        ),
        validation_loader=create_detection_data_loader(
            dataset=validation_dataset,
            shuffle=False,
            config=resolved_config,
        ),
        test_loader=create_detection_data_loader(
            dataset=test_dataset,
            shuffle=False,
            config=resolved_config,
        ),
    )
