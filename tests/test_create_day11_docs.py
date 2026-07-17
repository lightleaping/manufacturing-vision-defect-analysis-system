"""Day 11 Report와 README Marker 갱신을 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_day11_docs import (
    README_END,
    README_START,
    create_day11_docs,
)


def dataset_payload() -> dict:
    return {
        "validation_passed": True,
        "duplicate_box_policy": "preserve",
        "num_classes_including_background": 7,
        "class_mapping": {
            "background": 0,
            "crazing": 1,
            "inclusion": 2,
            "patches": 3,
            "pitted_surface": 4,
            "rolled_in_scale": 5,
            "scratches": 6,
        },
        "totals": {
            "sample_count": 1800,
            "box_count": 4189,
            "raw_exact_duplicate_count": 3,
        },
        "splits": {
            "train": {"sample_count": 1440, "dataset_box_count": 3335},
            "validation": {"sample_count": 178, "dataset_box_count": 425},
            "test": {"sample_count": 182, "dataset_box_count": 429},
        },
    }


def model_payload() -> dict:
    return {
        "validation_passed": True,
        "execution_policy": {"smoke_input_resize": [64, 64]},
        "source_sample": {"record_id": "train/crazing_1"},
        "model": {
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "device": "cpu",
            "predictor_output_classes": 7,
            "pretrained_detection_weights": None,
            "pretrained_backbone_weights": None,
            "network_download_requested": False,
        },
        "smoke_test": {
            "training_forward": {
                "losses": {
                    "loss_classifier": 1.0,
                    "loss_box_reg": 0.5,
                    "loss_objectness": 0.25,
                    "loss_rpn_box_reg": 0.125,
                },
                "total_loss": 1.875,
                "elapsed_seconds": 0.273,
            },
            "evaluation_forward": {
                "elapsed_seconds": 0.052,
                "predictions": [
                    {
                        "box_count": 10,
                        "boxes_shape": [10, 4],
                        "labels_shape": [10],
                        "scores_shape": [10],
                    }
                ],
            },
        },
    }


def prepare_project(tmp_path: Path) -> None:
    artifacts = tmp_path / "reports" / "artifacts"
    figures = tmp_path / "reports" / "figures"
    artifacts.mkdir(parents=True)
    figures.mkdir(parents=True)
    (artifacts / "day11_detection_dataset_validation.json").write_text(
        json.dumps(dataset_payload()), encoding="utf-8"
    )
    (artifacts / "day11_detection_model_smoke_test.json").write_text(
        json.dumps(model_payload()), encoding="utf-8"
    )
    for filename in (
        "day11_detection_dataset_batch.png",
        "day11_detection_target_overlay.png",
        "day11_detection_model_predictions_smoke_test.png",
    ):
        (figures / filename).write_bytes(b"png")
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")


def test_create_docs_writes_report_and_single_readme_marker(tmp_path: Path) -> None:
    prepare_project(tmp_path)

    report_path, readme_path = create_day11_docs(
        project_root=tmp_path,
        targeted_test_count=49,
        regression_test_count=1500,
        warning_count=1,
    )

    report = report_path.read_text(encoding="utf-8")
    readme = readme_path.read_text(encoding="utf-8")
    assert "1,800" in report
    assert "4,189" in report
    assert "Full Detection Training    : Not executed" in report
    assert "Day 11 targeted tests : 49 passed" in report
    assert readme.count(README_START) == 1
    assert readme.count(README_END) == 1
    assert "Random Initialization" in readme


def test_create_docs_is_idempotent(tmp_path: Path) -> None:
    prepare_project(tmp_path)
    for regression_count in (1500, 1501):
        create_day11_docs(
            project_root=tmp_path,
            targeted_test_count=49,
            regression_test_count=regression_count,
            warning_count=1,
        )

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme.count(README_START) == 1
    assert readme.count(README_END) == 1
    assert "Full regression 1501 passed" in readme
    assert "Full regression 1500 passed" not in readme


def test_create_docs_rejects_failed_artifact(tmp_path: Path) -> None:
    prepare_project(tmp_path)
    payload = dataset_payload()
    payload["validation_passed"] = False
    path = tmp_path / "reports" / "artifacts" / "day11_detection_dataset_validation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="has not passed"):
        create_day11_docs(
            project_root=tmp_path,
            targeted_test_count=49,
            regression_test_count=1500,
            warning_count=1,
        )


def test_create_docs_requires_prediction_figure(tmp_path: Path) -> None:
    prepare_project(tmp_path)
    (
        tmp_path
        / "reports"
        / "figures"
        / "day11_detection_model_predictions_smoke_test.png"
    ).unlink()

    with pytest.raises(FileNotFoundError, match="Figure"):
        create_day11_docs(
            project_root=tmp_path,
            targeted_test_count=49,
            regression_test_count=1500,
            warning_count=1,
        )
