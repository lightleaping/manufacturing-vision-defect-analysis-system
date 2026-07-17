from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image

from scripts import run_day9_detection_dataset_analysis as runner
from src.detection.dataset_config import NEU_DET_CANONICAL_CLASSES


def _write_pair(
    dataset_root: Path,
    *,
    split_name: str,
    class_name: str,
    index: int,
    pixel_value: int,
) -> None:
    images_dir = dataset_root / "NEU-DET" / split_name / "images" / class_name
    annotations_dir = dataset_root / "NEU-DET" / split_name / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{class_name}_{split_name}_{index}"
    Image.new("L", (200, 200), pixel_value).save(images_dir / f"{stem}.jpg")
    (annotations_dir / f"{stem}.xml").write_text(
        f"""
<annotation>
  <folder>{class_name}</folder>
  <filename>{stem}.jpg</filename>
  <source><database>NEU-DET</database></source>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  <segmented>0</segmented>
  <object>
    <name>{class_name}</name>
    <pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>200</xmax><ymax>200</ymax></bndbox>
  </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )


def test_run_day9_analysis_with_kaggle_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "data" / "raw" / "neu_det"
    pixel_value = 1
    for split_name in ("train", "validation"):
        for class_name in NEU_DET_CANONICAL_CLASSES:
            for index in range(2):
                _write_pair(
                    dataset_root,
                    split_name=split_name,
                    class_name=class_name,
                    index=index,
                    pixel_value=pixel_value,
                )
                pixel_value += 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_day9_detection_dataset_analysis.py",
            "--project-root",
            str(tmp_path),
            "--dataset-root",
            str(dataset_root),
        ],
    )

    assert runner.main() == 0

    analysis_path = (
        tmp_path
        / "reports"
        / "artifacts"
        / "day9_object_detection_dataset_analysis.json"
    )
    split_path = (
        tmp_path
        / "reports"
        / "artifacts"
        / "day9_object_detection_dataset_split.json"
    )
    processed_split_path = (
        tmp_path / "data" / "processed" / "neu_det" / "splits.json"
    )
    assert analysis_path.exists()
    assert split_path.exists()
    assert processed_split_path.exists()

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["summary"]["total_image_files"] == 24
    assert analysis["summary"]["total_annotation_files"] == 24
    assert analysis["summary"]["error_issue_count"] == 0
    assert analysis["summary"]["source_split_image_counts"] == {
        "train": 12,
        "validation": 12,
    }

    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert len(split["splits"]["train"]) == 12
    assert len(split["splits"]["validation"]) == 6
    assert len(split["splits"]["test"]) == 6

    for filename in (
        "day9_detection_class_distribution.png",
        "day9_detection_box_statistics.png",
        "day9_detection_annotation_overview.png",
    ):
        path = tmp_path / "reports" / "figures" / filename
        assert path.exists()
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            image.verify()


def test_run_day9_reconciles_cross_partition_pair_and_groups_duplicates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "data" / "raw" / "neu_det"
    pixel_value = 1
    for split_name in ("train", "validation"):
        for class_name in NEU_DET_CANONICAL_CLASSES:
            for index in range(2):
                _write_pair(
                    dataset_root,
                    split_name=split_name,
                    class_name=class_name,
                    index=index,
                    pixel_value=pixel_value,
                )
                pixel_value += 1

    # 실제 Kaggle 미러의 crazing_240처럼 이미지와 XML이 서로 다른
    # 원본 Partition에 있는 Pair를 재현한다.
    cross_images = (
        dataset_root / "NEU-DET" / "train" / "images" / "crazing"
    )
    cross_annotations = (
        dataset_root / "NEU-DET" / "validation" / "annotations"
    )
    cross_images.mkdir(parents=True, exist_ok=True)
    cross_annotations.mkdir(parents=True, exist_ok=True)
    Image.new("L", (200, 200), 200).save(
        cross_images / "crazing_cross.jpg"
    )
    (cross_annotations / "crazing_cross.xml").write_text(
        """
<annotation>
  <filename>crazing_cross.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  <object><name>crazing</name><bndbox>
    <xmin>1</xmin><ymin>1</ymin><xmax>200</xmax><ymax>200</ymax>
  </bndbox></object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    # 동일 이미지 Hash지만 Annotation은 별도인 두 레코드를 재현한다.
    train_images = (
        dataset_root / "NEU-DET" / "train" / "images" / "patches"
    )
    train_annotations = (
        dataset_root / "NEU-DET" / "train" / "annotations"
    )
    for stem, box in (
        ("patches_duplicate_a", (10, 10, 80, 90)),
        ("patches_duplicate_b", (12, 10, 82, 90)),
    ):
        Image.new("L", (200, 200), 222).save(train_images / f"{stem}.jpg")
        x_min, y_min, x_max, y_max = box
        (train_annotations / f"{stem}.xml").write_text(
            f"""
<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  <object><name>patches</name><bndbox>
    <xmin>{x_min}</xmin><ymin>{y_min}</ymin>
    <xmax>{x_max}</xmax><ymax>{y_max}</ymax>
  </bndbox></object>
</annotation>
""".strip(),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_day9_detection_dataset_analysis.py",
            "--project-root",
            str(tmp_path),
            "--dataset-root",
            str(dataset_root),
        ],
    )

    assert runner.main() == 0

    analysis_path = (
        tmp_path
        / "reports"
        / "artifacts"
        / "day9_object_detection_dataset_analysis.json"
    )
    split_path = (
        tmp_path
        / "reports"
        / "artifacts"
        / "day9_object_detection_dataset_split.json"
    )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    summary = analysis["summary"]
    assert summary["valid_record_count"] == 27
    assert summary["raw_missing_annotation_count"] == 1
    assert summary["raw_missing_image_count"] == 1
    assert summary["missing_annotation_count"] == 0
    assert summary["missing_image_count"] == 0
    assert summary["reconciled_cross_partition_pair_count"] == 1
    assert summary["duplicate_image_hash_group_count"] == 1
    assert summary["error_issue_count"] == 0

    split = json.loads(split_path.read_text(encoding="utf-8"))
    duplicate_hash = next(
        record["image_sha256"]
        for record in analysis["records"]
        if record["key"] == "train/patches_duplicate_a"
    )
    containing_splits = [
        split_name
        for split_name, records in split["splits"].items()
        if any(record["image_sha256"] == duplicate_hash for record in records)
    ]
    assert containing_splits == ["train"]
    assert sum(
        record["image_sha256"] == duplicate_hash
        for record in split["splits"]["train"]
    ) == 2
    assert split["statistics"]["train"][
        "duplicate_image_hash_group_count"
    ] == 1
