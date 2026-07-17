"""합성 세 Split으로 Day 11 Dataset Runtime Validation을 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

from src.detection.data_loader import DetectionDataLoaderConfig
from src.detection.dataset_validation import (
    build_day11_detection_dataset_validation,
    find_manifest_exact_duplicate_records,
    write_validation_artifact,
)
from tests.detection_test_helpers import write_pascal_voc_xml, write_rgb_image


def create_three_split_project(project_root: Path) -> Path:
    dataset_root = project_root / "data" / "raw" / "neu_det" / "NEU-DET"
    manifest_splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    specifications = {
        "train": [
            (
                "crazing_1",
                [
                    ("crazing", (1, 1, 10, 8)),
                    ("crazing", (1, 1, 10, 8)),
                ],
            ),
            ("scratches_1", [("scratches", (3, 2, 7, 5))]),
        ],
        "validation": [
            ("patches_1", [("patches", (2, 2, 8, 7))]),
        ],
        "test": [
            ("inclusion_1", [("inclusion", (1, 3, 4, 8))]),
        ],
    }
    for split, split_specs in specifications.items():
        for record_name, annotations in split_specs:
            image_path = dataset_root / split / "images" / f"{record_name}.jpg"
            xml_path = dataset_root / split / "annotations" / f"{record_name}.xml"
            write_rgb_image(image_path, width=10, height=8)
            write_pascal_voc_xml(
                xml_path,
                image_filename=f"metadata_{record_name}.jpg",
                width=10,
                height=8,
                boxes=annotations,
            )
            manifest_splits[split].append(
                {
                    "key": f"{split}/{record_name}",
                    "image_path": (
                        f"NEU-DET\\{split}\\images\\{record_name}.jpg"
                    ),
                    "annotation_path": (
                        f"NEU-DET\\{split}\\annotations\\{record_name}.xml"
                    ),
                    "image_width": 10,
                    "image_height": 8,
                    "image_mode": "RGB",
                    "class_names": [item[0] for item in annotations],
                    "boxes": [list(item[1]) for item in annotations],
                    "source_split": split,
                }
            )

    manifest_path = project_root / "data" / "processed" / "neu_det" / "splits.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"dataset_name": "NEU-DET", "splits": manifest_splits}),
        encoding="utf-8",
    )
    return manifest_path


def test_find_manifest_exact_duplicate_records(tmp_path: Path) -> None:
    manifest_path = create_three_split_project(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    duplicates = find_manifest_exact_duplicate_records(payload)

    assert len(duplicates) == 1
    assert duplicates[0]["record_id"] == "train/crazing_1"
    assert duplicates[0]["duplicates"][0]["occurrence_count"] == 2


def test_full_validation_preserves_duplicate_and_validates_loaders(
    tmp_path: Path,
) -> None:
    manifest_path = create_three_split_project(tmp_path)

    payload, loaders = build_day11_detection_dataset_validation(
        project_root=tmp_path,
        manifest_path=manifest_path,
        loader_config=DetectionDataLoaderConfig(batch_size=2),
        duplicate_box_policy="preserve",
    )

    assert payload["validation_passed"] is True
    assert payload["totals"] == {
        "sample_count": 4,
        "box_count": 5,
        "raw_exact_duplicate_count": 1,
    }
    assert payload["splits"]["train"]["dataset_box_count"] == 3
    assert payload["splits"]["train"]["class_box_counts"]["crazing"] == 2
    assert payload["data_loader_validation"]["train"]["first_batch_size"] == 2
    assert len(loaders.train_dataset) == 2


def test_remove_exact_policy_reduces_effective_box_count(tmp_path: Path) -> None:
    manifest_path = create_three_split_project(tmp_path)

    payload, _ = build_day11_detection_dataset_validation(
        project_root=tmp_path,
        manifest_path=manifest_path,
        duplicate_box_policy="remove_exact",
    )

    assert payload["validation_passed"] is True
    assert payload["totals"]["box_count"] == 4
    assert payload["totals"]["raw_exact_duplicate_count"] == 1
    assert (
        payload["splits"]["train"]["duplicate_box_audit"]
        ["effective_exact_duplicate_count"]
        == 0
    )


def test_validation_artifact_is_written_as_utf8_json(tmp_path: Path) -> None:
    manifest_path = create_three_split_project(tmp_path)
    payload, _ = build_day11_detection_dataset_validation(
        project_root=tmp_path,
        manifest_path=manifest_path,
    )
    output_path = tmp_path / "reports" / "artifacts" / "validation.json"

    write_validation_artifact(payload, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["project_name_ko"] == "제조 비전 결함 분석 시스템"
    assert loaded["validation_passed"] is True
