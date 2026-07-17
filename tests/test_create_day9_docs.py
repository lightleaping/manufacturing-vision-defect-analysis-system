from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_day9_docs import (
    README_END,
    README_START,
    create_day9_docs,
    replace_readme_section,
)


def _analysis_payload() -> dict[str, object]:
    return {
        "summary": {
            "total_image_files": 1800,
            "total_annotation_files": 1800,
            "valid_record_count": 1800,
            "total_valid_bounding_boxes": 4189,
            "class_count": 6,
            "class_image_counts": {
                "crazing": 300,
                "inclusion": 300,
                "patches": 300,
                "pitted_surface": 300,
                "rolled_in_scale": 300,
                "scratches": 300,
            },
            "class_box_counts": {
                "crazing": 724,
                "inclusion": 812,
                "patches": 610,
                "pitted_surface": 650,
                "rolled_in_scale": 700,
                "scratches": 693,
            },
            "image_mode_counts": {"RGB": 1800},
            "raw_missing_annotation_count": 1,
            "raw_missing_image_count": 1,
            "missing_annotation_count": 0,
            "missing_image_count": 0,
            "reconciled_cross_partition_pair_count": 1,
            "duplicate_image_hash_group_count": 1,
            "corrupted_image_count": 0,
            "invalid_annotation_count": 0,
            "invalid_box_count": 0,
            "error_issue_count": 0,
            "warning_issue_count": 179,
            "issue_counts_by_code": {
                "cross_partition_pair_reconciled": 1,
                "duplicate_box": 3,
                "duplicate_image_hash": 1,
                "filename_mismatch": 174,
            },
            "boxes_per_image": {"mean": 2.327, "median": 2, "min": 1, "max": 10},
            "box_width": {"mean": 71, "median": 65, "min": 4, "max": 199},
            "box_height": {"mean": 95, "median": 90, "min": 4, "max": 199},
            "box_area_ratio": {"mean": 0.2, "median": 0.15, "min": 0.001, "max": 0.95},
            "box_aspect_ratio": {"mean": 1.2, "median": 1.0, "min": 0.1, "max": 8.0},
            "coordinate_statistics": {
                "inferred_source_coordinate_policy": "pascal_voc_one_based_inclusive_likely",
                "zero_min_coordinate_count": 0,
                "one_min_coordinate_count": 1500,
                "x_max_at_image_width_count": 256,
                "y_max_at_image_height_count": 316,
            },
        }
    }


def _split_payload() -> dict[str, object]:
    def item(index: int, split_name: str) -> dict[str, object]:
        return {
            "image_path": f"{split_name}/image_{index}.jpg",
            "image_sha256": f"{split_name}-hash-{index}",
        }

    train = [item(index, "train") for index in range(4)]
    validation = [item(index, "validation") for index in range(2)]
    test = [item(index, "test") for index in range(2)]
    return {
        "random_seed": 42,
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "statistics": {
            "split_policy": "preserve_source_train_and_hash_group_split_source_validation_pool",
            "duplicate_hash_policy": "keep_duplicate_hash_group_inside_one_split",
            "train": {"image_count": 4, "box_count": 10, "duplicate_image_hash_group_count": 0},
            "validation": {"image_count": 2, "box_count": 4, "duplicate_image_hash_group_count": 0},
            "test": {"image_count": 2, "box_count": 5, "duplicate_image_hash_group_count": 0},
        },
    }


def _write_artifacts(root: Path, *, visual_status: str = "PASS") -> None:
    artifact_dir = root / "reports" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "day9_object_detection_dataset_analysis.json").write_text(
        json.dumps(_analysis_payload()), encoding="utf-8"
    )
    (artifact_dir / "day9_object_detection_dataset_split.json").write_text(
        json.dumps(_split_payload()), encoding="utf-8"
    )
    (artifact_dir / "day9_detection_visual_validation.json").write_text(
        json.dumps(
            {
                "status": visual_status,
                "all_manual_checks_passed": visual_status == "PASS",
            }
        ),
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Project\n", encoding="utf-8")


def test_replace_readme_section_adds_and_replaces_marker() -> None:
    section1 = f"{README_START}\nfirst\n{README_END}"
    section2 = f"{README_START}\nsecond\n{README_END}"
    first = replace_readme_section("# Project\n", section1)
    second = replace_readme_section(first, section2)
    assert first.count(README_START) == 1
    assert second.count(README_START) == 1
    assert "second" in second
    assert "first" not in second


def test_create_day9_docs_requires_visual_pass(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, visual_status="FAIL")
    with pytest.raises(ValueError, match="육안 검증"):
        create_day9_docs(
            project_root=tmp_path,
            regression_test_count=1368,
            warning_count=1,
            runtime_seconds=100.0,
        )


def test_create_day9_docs_generates_report_and_readme(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    report, readme = create_day9_docs(
        project_root=tmp_path,
        regression_test_count=1368,
        warning_count=1,
        runtime_seconds=170.59,
    )
    report_text = report.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    assert "Day 9 — Object Detection Dataset Analysis" in report_text
    assert "1,800" in report_text
    assert "4,189" in report_text
    assert "1,368 passed" in report_text
    assert README_START in readme_text
    assert README_END in readme_text


def test_create_day9_docs_is_idempotent(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    for _ in range(2):
        create_day9_docs(
            project_root=tmp_path,
            regression_test_count=1368,
            warning_count=1,
            runtime_seconds=None,
        )
    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme_text.count(README_START) == 1
    assert readme_text.count(README_END) == 1


def test_create_day9_docs_rejects_non_positive_test_count(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="1 이상"):
        create_day9_docs(
            project_root=tmp_path,
            regression_test_count=0,
            warning_count=1,
            runtime_seconds=None,
        )
