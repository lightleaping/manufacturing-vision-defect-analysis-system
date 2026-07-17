from pathlib import Path

from PIL import Image

from src.detection.dataset_analysis import (
    analyze_detection_dataset,
    combine_analysis_results,
    save_analysis_json,
)
from src.detection.dataset_config import DetectionDatasetConfig


def _config(
    tmp_path: Path,
    *,
    source_split: str = "unspecified",
) -> DetectionDatasetConfig:
    partition_root = tmp_path / source_split
    images_dir = partition_root / "IMAGES"
    annotations_dir = partition_root / "ANNOTATIONS"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    return DetectionDatasetConfig(
        project_root=tmp_path,
        dataset_root=tmp_path,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
        figures_dir=tmp_path / "figures",
        source_split=source_split,
    )


def _write_pair(
    config: DetectionDatasetConfig,
    stem: str,
    *,
    class_name: str,
    box: tuple[int, int, int, int] = (10, 20, 100, 120),
    value: int = 127,
) -> None:
    image_path = config.images_dir / f"{stem}.jpg"
    Image.new("L", (200, 200), value).save(image_path)
    x_min, y_min, x_max, y_max = box
    (config.annotations_dir / f"{stem}.xml").write_text(
        f"""
<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  <object>
    <name>{class_name}</name>
    <bndbox>
      <xmin>{x_min}</xmin><ymin>{y_min}</ymin>
      <xmax>{x_max}</xmax><ymax>{y_max}</ymax>
    </bndbox>
  </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )


def test_analyze_detection_dataset_counts_and_statistics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_pair(config, "a", class_name="crazing", value=10)
    _write_pair(
        config,
        "b",
        class_name="scratches",
        box=(20, 30, 80, 100),
        value=20,
    )

    result = analyze_detection_dataset(config)
    summary = result.summary

    assert summary["total_image_files"] == 2
    assert summary["total_annotation_files"] == 2
    assert summary["valid_record_count"] == 2
    assert summary["total_valid_bounding_boxes"] == 2
    assert summary["class_image_counts"]["crazing"] == 1
    assert summary["class_box_counts"]["scratches"] == 1
    assert summary["boxes_per_image"]["mean"] == 1.0
    assert summary["box_area_ratio"]["count"] == 2
    assert summary["corrupted_image_count"] == 0


def test_analyze_tracks_source_split_and_coordinate_policy(tmp_path: Path) -> None:
    config = _config(tmp_path, source_split="train")
    _write_pair(
        config,
        "sample",
        class_name="crazing",
        box=(1, 1, 200, 200),
    )
    result = analyze_detection_dataset(config)

    assert result.records[0].source_split == "train"
    assert result.records[0].key == "train/sample"
    assert result.summary["source_split_image_counts"] == {"train": 1}
    coordinate = result.summary["coordinate_statistics"]
    assert coordinate["x_max_at_image_width_count"] == 1
    assert coordinate["y_max_at_image_height_count"] == 1
    assert coordinate["inferred_source_coordinate_policy"] == (
        "pascal_voc_one_based_inclusive_likely"
    )


def test_analyze_reports_missing_pair_invalid_box_and_corruption(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_pair(
        config,
        "invalid_box",
        class_name="crazing",
        box=(100, 20, 90, 120),
    )

    Image.new("L", (200, 200), 0).save(config.images_dir / "missing_xml.jpg")
    (config.annotations_dir / "missing_image.xml").write_text(
        """
<annotation>
  <filename>missing_image.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
</annotation>
""".strip(),
        encoding="utf-8",
    )
    (config.images_dir / "broken.jpg").write_bytes(b"not-an-image")
    (config.annotations_dir / "broken.xml").write_text(
        """
<annotation>
  <filename>broken.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  <object><name>crazing</name><bndbox>
  <xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax>
  </bndbox></object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    result = analyze_detection_dataset(config)
    summary = result.summary
    codes = {issue["code"] for issue in result.issues}

    assert summary["missing_annotation_count"] == 1
    assert summary["missing_image_count"] == 1
    assert summary["invalid_box_count"] == 1
    assert summary["corrupted_image_count"] == 1
    assert "missing_annotation" in codes
    assert "missing_image" in codes
    assert "invalid_x_order" in codes
    assert "corrupted_image" in codes


def test_duplicate_image_hash_is_reported_as_warning(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_pair(config, "first", class_name="crazing", value=50)
    _write_pair(config, "second", class_name="crazing", value=50)

    result = analyze_detection_dataset(config)
    assert result.summary["duplicate_image_hash_group_count"] == 1
    assert result.summary["duplicate_image_count"] == 2
    duplicate_issue = next(
        issue
        for issue in result.issues
        if issue["code"] == "duplicate_image_hash"
    )
    assert duplicate_issue["severity"] == "warning"
    assert result.summary["error_issue_count"] == 0


def test_combine_partition_results(tmp_path: Path) -> None:
    train = _config(tmp_path, source_split="train")
    validation = _config(tmp_path, source_split="validation")
    _write_pair(train, "train_a", class_name="crazing", value=10)
    _write_pair(validation, "val_a", class_name="scratches", value=20)

    combined = combine_analysis_results(
        [
            analyze_detection_dataset(train),
            analyze_detection_dataset(validation),
        ],
        dataset_root=tmp_path,
    )

    assert combined.summary["total_image_files"] == 2
    assert combined.summary["source_partition_count"] == 2
    assert combined.summary["source_split_image_counts"] == {
        "train": 1,
        "validation": 1,
    }
    assert combined.summary["cross_source_split_duplicate_hash_count"] == 0


def test_save_analysis_json(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_pair(config, "sample", class_name="patches")
    result = analyze_detection_dataset(config)

    output = save_analysis_json(result, tmp_path / "artifact.json")
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert '"total_image_files": 1' in text
    assert '"patches": 1' in text



def test_combine_reconciles_unique_cross_partition_pair(tmp_path: Path) -> None:
    train = _config(tmp_path, source_split="train")
    validation = _config(tmp_path, source_split="validation")

    Image.new("L", (200, 200), 77).save(
        train.images_dir / "crazing_240.jpg"
    )
    (validation.annotations_dir / "crazing_240.xml").write_text(
        """
<annotation>
  <filename>crazing_240.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  <object>
    <name>crazing</name>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>200</xmax><ymax>200</ymax></bndbox>
  </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    combined = combine_analysis_results(
        [
            analyze_detection_dataset(train),
            analyze_detection_dataset(validation),
        ],
        dataset_root=tmp_path,
    )

    assert combined.summary["raw_missing_annotation_count"] == 1
    assert combined.summary["raw_missing_image_count"] == 1
    assert combined.summary["missing_annotation_count"] == 0
    assert combined.summary["missing_image_count"] == 0
    assert combined.summary["reconciled_cross_partition_pair_count"] == 1
    assert combined.summary["valid_record_count"] == 1
    assert combined.summary["error_issue_count"] == 0
    assert combined.records[0].source_split == "train"
    assert combined.records[0].image_path.replace("\\", "/").endswith("train/IMAGES/crazing_240.jpg")
    assert combined.records[0].annotation_path.replace("\\", "/").endswith(
        "validation/ANNOTATIONS/crazing_240.xml"
    )
    assert combined.summary["issue_counts_by_code"] == {
        "cross_partition_pair_reconciled": 1
    }



