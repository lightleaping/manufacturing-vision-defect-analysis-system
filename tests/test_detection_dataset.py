"""합성 Pascal VOC와 실제 Day 9 Manifest Schema 기반 Dataset 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.detection.detection_dataset import (
    NeuDetDetectionDataset,
    convert_voc_box_to_torchvision,
    load_detection_samples_from_manifest,
)
from src.detection.transforms import create_detection_transform
from tests.detection_test_helpers import create_sample


def create_dataset(
    tmp_path: Path,
    *,
    boxes=(("crazing", (1, 1, 10, 8)),),
    duplicate_box_policy: str = "preserve",
) -> NeuDetDetectionDataset:
    sample = create_sample(tmp_path, record_id="sample_1", boxes=boxes)
    return NeuDetDetectionDataset(
        samples=[sample],
        transform=create_detection_transform(training=False),
        duplicate_box_policy=duplicate_box_policy,
    )


def test_boundary_voc_box_converts_to_full_image_exclusive_box() -> None:
    assert convert_voc_box_to_torchvision(
        (1, 1, 10, 8),
        image_width=10,
        image_height=8,
    ) == (0.0, 0.0, 10.0, 8.0)


def test_single_pixel_inclusive_voc_box_remains_positive() -> None:
    assert convert_voc_box_to_torchvision(
        (3, 4, 3, 4),
        image_width=10,
        image_height=8,
    ) == (2.0, 3.0, 3.0, 4.0)


def test_dataset_returns_torchvision_detection_contract(tmp_path: Path) -> None:
    dataset = create_dataset(
        tmp_path,
        boxes=(
            ("crazing", (1, 1, 10, 8)),
            ("scratches", (3, 2, 7, 5)),
        ),
    )

    image, target = dataset[0]

    assert image.shape == (3, 8, 10)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min().item() <= image.max().item() <= 1.0
    assert set(target) == {"boxes", "labels", "image_id", "area", "iscrowd"}
    assert target["boxes"].shape == (2, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64
    assert target["labels"].tolist() == [1, 6]
    assert target["image_id"].tolist() == [0]
    assert target["iscrowd"].tolist() == [0, 0]
    assert target["area"].tolist() == pytest.approx([80.0, 20.0])


def test_duplicate_boxes_are_preserved_by_default(tmp_path: Path) -> None:
    duplicate = ("patches", (2, 2, 6, 6))
    dataset = create_dataset(
        tmp_path,
        boxes=(duplicate, duplicate),
        duplicate_box_policy="preserve",
    )

    _, target = dataset[0]

    assert len(target["boxes"]) == 2
    assert target["labels"].tolist() == [3, 3]


def test_remove_exact_only_removes_same_class_and_coordinates(
    tmp_path: Path,
) -> None:
    dataset = create_dataset(
        tmp_path,
        boxes=(
            ("patches", (2, 2, 6, 6)),
            ("patches", (2, 2, 6, 6)),
            ("scratches", (2, 2, 6, 6)),
        ),
        duplicate_box_policy="remove_exact",
    )

    _, target = dataset[0]

    assert len(target["boxes"]) == 2
    assert target["labels"].tolist() == [3, 6]


def test_xml_filename_mismatch_does_not_break_manifest_pair(tmp_path: Path) -> None:
    dataset = create_dataset(tmp_path)
    image, target = dataset[0]

    assert image.shape == (3, 8, 10)
    assert len(target["boxes"]) == 1


def test_day9_class_alias_is_normalized_before_label_mapping(
    tmp_path: Path,
) -> None:
    dataset = create_dataset(
        tmp_path,
        boxes=(("rolled-in_scale", (1, 1, 10, 8)),),
    )

    _, target = dataset[0]

    assert target["labels"].tolist() == [5]


def test_unsupported_class_is_rejected(tmp_path: Path) -> None:
    dataset = create_dataset(
        tmp_path,
        boxes=(("unknown_defect", (1, 1, 3, 3)),),
    )

    with pytest.raises(ValueError, match="Unsupported class"):
        _ = dataset[0]


def test_invalid_box_is_rejected(tmp_path: Path) -> None:
    dataset = create_dataset(
        tmp_path,
        boxes=(("crazing", (0, 1, 3, 3)),),
    )

    with pytest.raises(ValueError, match="at least 1"):
        _ = dataset[0]


def test_manifest_resolves_actual_neu_det_windows_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "raw" / "neu_det"
    sample = create_sample(
        dataset_root / "NEU-DET" / "train",
        record_id="crazing_1",
    )
    manifest_path = tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "NEU-DET",
                "splits": {
                    "train": [
                        {
                            "key": "train/crazing_1",
                            "image_path": (
                                "NEU-DET\\train\\images\\crazing_1.jpg"
                            ),
                            "annotation_path": (
                                "NEU-DET\\train\\annotations\\crazing_1.xml"
                            ),
                            "class_names": ["crazing"],
                            "boxes": [[1, 1, 10, 8]],
                        }
                    ],
                    "validation": [],
                    "test": [],
                },
            }
        ),
        encoding="utf-8",
    )

    samples = load_detection_samples_from_manifest(
        manifest_path=manifest_path,
        project_root=tmp_path,
        split="train",
    )

    assert samples[0].record_id == "train/crazing_1"
    assert samples[0].image_path == sample.image_path.resolve()
    assert samples[0].annotation_path == sample.annotation_path.resolve()
