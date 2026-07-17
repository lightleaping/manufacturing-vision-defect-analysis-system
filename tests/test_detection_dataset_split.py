from pathlib import Path

import pytest

from src.detection.dataset_analysis import ImageAnnotationRecord
from src.detection.dataset_config import SplitRatios
from src.detection.dataset_split import (
    build_source_preserving_split_manifest,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
    validate_split_manifest,
)


def _records(
    count_per_class: int = 20,
    *,
    source_split: str = "all",
    start_index: int = 0,
) -> list[ImageAnnotationRecord]:
    records: list[ImageAnnotationRecord] = []
    for class_name in ("crazing", "scratches"):
        for index in range(start_index, start_index + count_per_class):
            key = f"{class_name}_{source_split}_{index}"
            records.append(
                ImageAnnotationRecord(
                    key=key,
                    image_path=f"{source_split}/images/{key}.jpg",
                    annotation_path=f"{source_split}/annotations/{key}.xml",
                    image_width=200,
                    image_height=200,
                    image_mode="L",
                    image_sha256=f"{class_name}-{source_split}-{index:04d}",
                    class_names=(class_name,),
                    boxes=((10, 20, 100, 120),),
                    source_split=source_split,
                )
            )
    return records


def test_split_is_reproducible_and_disjoint() -> None:
    records = _records()
    first = build_split_manifest(records, random_seed=42)
    second = build_split_manifest(records, random_seed=42)

    assert first.to_dict() == second.to_dict()
    validation = validate_split_manifest(
        first,
        expected_record_count=len(records),
    )
    assert validation["is_valid"] is True
    assert validation["split_overlap_count"] == 0
    assert validation["cross_split_duplicate_hash_count"] == 0


def test_split_preserves_all_records_and_approximate_ratio() -> None:
    records = _records(count_per_class=100)
    manifest = build_split_manifest(
        records,
        ratios=SplitRatios(train=0.7, validation=0.15, test=0.15),
        random_seed=42,
    )
    counts = {
        name: len(items)
        for name, items in manifest.splits.items()
    }
    assert counts == {"train": 140, "validation": 30, "test": 30}
    assert sum(counts.values()) == len(records)


def test_source_preserving_policy_keeps_train_and_splits_validation() -> None:
    train_records = _records(
        count_per_class=8,
        source_split="train",
    )
    validation_pool = _records(
        count_per_class=4,
        source_split="validation",
        start_index=100,
    )
    manifest = build_source_preserving_split_manifest(
        train_records,
        validation_pool,
        validation_fraction=0.5,
        random_seed=42,
    )

    assert len(manifest.splits["train"]) == 16
    assert len(manifest.splits["validation"]) == 4
    assert len(manifest.splits["test"]) == 4
    assert manifest.statistics["split_policy"] == (
        "preserve_source_train_and_hash_group_split_source_validation_pool"
    )
    assert {
        item["image_path"] for item in manifest.splits["train"]
    } == {record.image_path for record in train_records}
    for split_name in ("validation", "test"):
        assert all(
            item["source_split"] == "validation"
            for item in manifest.splits[split_name]
        )


def test_source_preserving_policy_is_reproducible() -> None:
    train_records = _records(8, source_split="train")
    validation_pool = _records(
        4,
        source_split="validation",
        start_index=100,
    )
    first = build_source_preserving_split_manifest(
        train_records,
        validation_pool,
        random_seed=42,
    )
    second = build_source_preserving_split_manifest(
        train_records,
        validation_pool,
        random_seed=42,
    )
    assert first.to_dict() == second.to_dict()


def test_duplicate_hash_records_are_kept_in_one_split() -> None:
    records = _records(count_per_class=2)
    duplicate = records[0]
    records.append(
        ImageAnnotationRecord(
            key="copy",
            image_path="images/copy.jpg",
            annotation_path="annotations/copy.xml",
            image_width=200,
            image_height=200,
            image_mode="L",
            image_sha256=duplicate.image_sha256,
            class_names=("crazing",),
            boxes=((1, 1, 10, 10),),
        )
    )

    manifest = build_split_manifest(records, random_seed=42)
    containing_splits = [
        split_name
        for split_name, items in manifest.splits.items()
        if sum(
            item["image_sha256"] == duplicate.image_sha256
            for item in items
        )
    ]
    assert len(containing_splits) == 1
    split_name = containing_splits[0]
    assert sum(
        item["image_sha256"] == duplicate.image_sha256
        for item in manifest.splits[split_name]
    ) == 2
    assert validate_split_manifest(manifest)[
        "cross_split_duplicate_hash_count"
    ] == 0


def test_duplicate_hash_can_be_explicitly_rejected() -> None:
    records = _records(count_per_class=2)
    duplicate = records[0]
    records.append(
        ImageAnnotationRecord(
            key="copy",
            image_path="images/copy.jpg",
            annotation_path="annotations/copy.xml",
            image_width=200,
            image_height=200,
            image_mode="L",
            image_sha256=duplicate.image_sha256,
            class_names=("crazing",),
            boxes=((1, 1, 10, 10),),
        )
    )
    with pytest.raises(ValueError, match="명시적 거부 정책"):
        build_split_manifest(records, reject_duplicate_hashes=True)


def test_manifest_save_and_load(tmp_path: Path) -> None:
    manifest = build_split_manifest(_records(count_per_class=5))
    path = save_split_manifest(manifest, tmp_path / "splits.json")
    restored = load_split_manifest(path)
    assert restored.to_dict() == manifest.to_dict()



def test_source_preserving_policy_keeps_validation_duplicates_together() -> None:
    train_records = _records(2, source_split="train")
    validation_pool = _records(
        3,
        source_split="validation",
        start_index=100,
    )
    original = validation_pool[0]
    validation_pool.append(
        ImageAnnotationRecord(
            key="validation_copy",
            image_path="validation/images/validation_copy.jpg",
            annotation_path="validation/annotations/validation_copy.xml",
            image_width=200,
            image_height=200,
            image_mode="L",
            image_sha256=original.image_sha256,
            class_names=original.class_names,
            boxes=((20, 20, 80, 80),),
            source_split="validation",
        )
    )

    manifest = build_source_preserving_split_manifest(
        train_records,
        validation_pool,
        validation_fraction=0.5,
        random_seed=42,
    )
    final_splits = [
        split_name
        for split_name in ("validation", "test")
        if sum(
            item["image_sha256"] == original.image_sha256
            for item in manifest.splits[split_name]
        )
    ]
    assert len(final_splits) == 1
    assert sum(
        item["image_sha256"] == original.image_sha256
        for item in manifest.splits[final_splits[0]]
    ) == 2
    assert validate_split_manifest(manifest)["is_valid"] is True


def test_source_preserving_rejects_duplicate_hash_across_source_pools() -> None:
    train_records = _records(2, source_split="train")
    validation_pool = _records(
        2,
        source_split="validation",
        start_index=100,
    )
    validation_pool[0] = ImageAnnotationRecord(
        key=validation_pool[0].key,
        image_path=validation_pool[0].image_path,
        annotation_path=validation_pool[0].annotation_path,
        image_width=200,
        image_height=200,
        image_mode="L",
        image_sha256=train_records[0].image_sha256,
        class_names=validation_pool[0].class_names,
        boxes=validation_pool[0].boxes,
        source_split="validation",
    )
    with pytest.raises(ValueError, match="원본 Train과 Validation Pool"):
        build_source_preserving_split_manifest(
            train_records,
            validation_pool,
            random_seed=42,
        )
