from pathlib import Path

import pytest

from src.detection.dataset_config import (
    DETECTION_MODEL_CLASS_TO_INDEX,
    DETECTION_SOURCE_CLASS_TO_INDEX,
    SplitRatios,
    build_default_config,
    discover_neu_det_layout,
    discover_neu_det_partitions,
    normalize_annotation_class_name,
)


def test_detection_class_mapping_separates_background() -> None:
    assert "background" not in DETECTION_SOURCE_CLASS_TO_INDEX
    assert DETECTION_MODEL_CLASS_TO_INDEX["background"] == 0
    assert DETECTION_MODEL_CLASS_TO_INDEX["crazing"] == 1
    assert len(DETECTION_SOURCE_CLASS_TO_INDEX) == 6
    assert len(DETECTION_MODEL_CLASS_TO_INDEX) == 7


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Crazing", "crazing"),
        ("pitted surface", "pitted_surface"),
        ("rolled-in scale", "rolled_in_scale"),
        ("rolled-in_scale", "rolled_in_scale"),
        ("rolled_in_scale", "rolled_in_scale"),
        ("Sc", "scratches"),
    ],
)
def test_normalize_annotation_class_name(
    raw_name: str,
    expected: str,
) -> None:
    assert normalize_annotation_class_name(raw_name) == expected


def test_unknown_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="알 수 없는 Detection Class"):
        normalize_annotation_class_name("mystery_defect")


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="합은 1.0"):
        SplitRatios(train=0.8, validation=0.15, test=0.1)


def test_build_default_config_without_dataset(tmp_path: Path) -> None:
    config = build_default_config(
        project_root=tmp_path,
        dataset_root=tmp_path / "data" / "raw" / "neu_det",
        discover_layout=False,
    )
    assert config.images_dir.name == "IMAGES"
    assert config.annotations_dir.name == "ANNOTATIONS"
    assert config.random_seed == 42
    assert config.processed_dir == tmp_path / "data" / "processed" / "neu_det"


def test_discover_neu_det_layout(tmp_path: Path) -> None:
    images_dir = tmp_path / "NEU-DET" / "IMAGES"
    annotations_dir = tmp_path / "NEU-DET" / "ANNOTATIONS"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    (images_dir / "sample.jpg").write_bytes(b"placeholder")
    (annotations_dir / "sample.xml").write_text("<annotation />", encoding="utf-8")

    layout = discover_neu_det_layout(tmp_path)
    assert layout.images_dir == images_dir.resolve()
    assert layout.annotations_dir == annotations_dir.resolve()


def test_discover_kaggle_train_validation_partitions(tmp_path: Path) -> None:
    for split_name in ("train", "validation"):
        images_dir = tmp_path / "NEU-DET" / split_name / "images" / "crazing"
        annotations_dir = tmp_path / "NEU-DET" / split_name / "annotations"
        images_dir.mkdir(parents=True)
        annotations_dir.mkdir(parents=True)
        (images_dir / f"{split_name}.jpg").write_bytes(b"image")
        (annotations_dir / f"{split_name}.xml").write_text(
            "<annotation />",
            encoding="utf-8",
        )

    layout = discover_neu_det_partitions(tmp_path)
    assert layout.partition_names == ("train", "validation")
    assert layout.get("train").images_dir.name == "images"
    assert layout.get("validation").annotations_dir.name == "annotations"

    with pytest.raises(RuntimeError, match="여러 Partition"):
        discover_neu_det_layout(tmp_path)
