from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.create_day12_docs import (
    FAILURE_ARTIFACT,
    README_END,
    README_START,
    REPORT_PATH,
    REQUIRED_FIGURES,
    TRAINING_CONFIG_ARTIFACT,
    TRAINING_HISTORY_ARTIFACT,
    EVALUATION_ARTIFACT,
    BEST_CHECKPOINT,
    create_day12_docs,
    render_day12_report,
    update_marker_block,
)


CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
)


def _overall(*, precision: float, recall: float, f1: float, map_50: float):
    return {
        "image_count": 2,
        "tp": 2,
        "fp": 1,
        "fn": 3,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": 0.75,
        "map_50": map_50,
    }


def _class_metrics():
    result = {}
    for index, name in enumerate(CLASS_NAMES, start=1):
        result[name] = {
            "tp": index,
            "fp": 1,
            "fn": 7 - index,
            "ground_truth_count": 7,
            "precision": index / (index + 1),
            "recall": index / 7,
            "f1": index / 8,
            "mean_matched_iou": 0.6 + index / 100,
            "ap_50": 0.4 + index / 20,
        }
    return result


def _evaluation():
    class_metrics = _class_metrics()
    return {
        "evaluation_policy": {
            "best_checkpoint_selected_on_validation": True,
            "best_checkpoint_frozen_before_test": True,
            "test_split_used": True,
            "test_result_used_for_model_selection": False,
            "score_threshold": 0.5,
            "iou_threshold": 0.5,
            "duplicate_box_policy": "preserve",
        },
        "checkpoint": {
            "epoch_index": 2,
            "completed_epoch_number": 3,
            "best_validation_metric": 0.677418,
        },
        "model": {
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "device": "cpu",
            "min_size": 320,
            "max_size": 320,
            "pretrained_detection_weights": "COCO_V1",
            "num_classes_with_background": 7,
        },
        "validation": {
            "metrics": {
                "overall": _overall(
                    precision=0.833333,
                    recall=0.505882,
                    f1=0.629575,
                    map_50=0.677418,
                )
            }
        },
        "test": {
            "metrics": {
                "overall": _overall(
                    precision=0.81295,
                    recall=0.526807,
                    f1=0.639321,
                    map_50=0.707726,
                ),
                "class_metrics": class_metrics,
            }
        },
        "test_iou_sweep": {
            "summary": {"map_50_95": 0.310533},
            "class_map_50_95": {
                name: 0.1 + index / 20
                for index, name in enumerate(CLASS_NAMES, start=1)
            },
        },
    }


def _failure():
    return {
        "split": "test",
        "checkpoint": {"epoch_index": 2},
        "analysis": {
            "summary": {
                "image_count": 182,
                "images_with_failures": 129,
                "event_count": 229,
                "counts": {
                    "low_confidence_correct_detection": 140,
                    "false_negative": 37,
                    "low_iou_localization": 25,
                    "false_positive": 23,
                    "duplicate_prediction": 3,
                    "wrong_class": 1,
                },
            }
        },
    }


def _checkpoint():
    history = []
    values = (
        (0, False, 0.722109, 707.4, 0.590604, 0.207059, 0.306620, 0.402362),
        (1, True, 0.954561, 1336.8, 0.847619, 0.418824, 0.560630, 0.637687),
        (2, True, 1.016442, 1254.0, 0.833333, 0.505882, 0.629575, 0.677418),
    )
    for epoch, trainable, loss, elapsed, precision, recall, f1, map_50 in values:
        history.append(
            {
                "epoch": epoch,
                "stage": (
                    "frozen_backbone_one_epoch_pilot"
                    if epoch == 0
                    else "unfrozen_backbone_fine_tuning"
                ),
                "backbone_trainable": trainable,
                "learning_rates": [0.005 if epoch == 0 else 0.001],
                "train": {
                    "average_losses": {"total_loss": loss},
                    "elapsed_seconds": elapsed,
                },
                "validation": {
                    "metrics": {
                        "overall": {
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "map_50": map_50,
                        }
                    }
                },
            }
        )
    return {
        "epoch": 2,
        "history": history,
        "best_metric": 0.677418,
        "training_config": {"batch_size": 1, "learning_rate": 0.001},
        "class_mapping": {"BACKGROUND": 0, **{name: i for i, name in enumerate(CLASS_NAMES, 1)}},
    }


def _write_fixture(root: Path) -> None:
    paths = (
        TRAINING_CONFIG_ARTIFACT,
        TRAINING_HISTORY_ARTIFACT,
        EVALUATION_ARTIFACT,
        FAILURE_ARTIFACT,
    )
    for relative in paths:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / TRAINING_CONFIG_ARTIFACT).write_text(
        json.dumps({"batch_size": 1}), encoding="utf-8"
    )
    (root / TRAINING_HISTORY_ARTIFACT).write_text(
        json.dumps({"validation_passed": True}), encoding="utf-8"
    )
    (root / EVALUATION_ARTIFACT).write_text(
        json.dumps(_evaluation()), encoding="utf-8"
    )
    (root / FAILURE_ARTIFACT).write_text(
        json.dumps(_failure()), encoding="utf-8"
    )
    checkpoint_path = root / BEST_CHECKPOINT
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint(), checkpoint_path)
    for relative in REQUIRED_FIGURES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    (root / "README.md").write_text("# Project\n", encoding="utf-8")


def test_marker_block_appends_when_missing() -> None:
    updated = update_marker_block("# README\n", f"{README_START}\nA\n{README_END}")
    assert updated.count(README_START) == 1
    assert updated.count(README_END) == 1
    assert updated.startswith("# README")


def test_marker_block_replaces_without_duplicate() -> None:
    original = f"before\n{README_START}\nold\n{README_END}\nafter"
    updated = update_marker_block(
        original,
        f"{README_START}\nnew\n{README_END}",
    )
    assert "old" not in updated
    assert updated.count(README_START) == 1
    assert updated.count(README_END) == 1


def test_marker_block_rejects_unbalanced_markers() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        update_marker_block(f"{README_START}\nmissing end", "replacement")


def test_report_contains_real_metrics_and_limitations() -> None:
    report = render_day12_report(
        training_config={"batch_size": 1},
        training_history={"validation_passed": True},
        evaluation=_evaluation(),
        failure=_failure(),
        checkpoint=_checkpoint(),
        targeted_test_count=84,
        regression_test_count=1576,
        warning_count=1,
    )
    assert "0.707726" in report
    assert "0.310533" in report
    assert "`scratches`" in report
    assert "Low-confidence Correct Detection" in report
    assert "공식 `pycocotools.COCOeval`" in report
    assert "Day 13 범위" in report


def test_create_docs_writes_report_readme_and_backup(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report_path, readme_path = create_day12_docs(
        project_root=tmp_path,
        targeted_test_count=84,
        regression_test_count=1576,
        warning_count=1,
    )
    assert report_path == tmp_path / REPORT_PATH
    assert report_path.is_file()
    assert "Test mAP@0.50" in report_path.read_text(encoding="utf-8")
    readme = readme_path.read_text(encoding="utf-8")
    assert readme.count(README_START) == 1
    assert readme.count(README_END) == 1
    assert "0.707726" in readme
    assert (
        tmp_path
        / "reports/backups/day12_docs/README.md.before_day12_docs"
    ).is_file()


def test_create_docs_is_idempotent(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    for _ in range(2):
        create_day12_docs(
            project_root=tmp_path,
            targeted_test_count=84,
            regression_test_count=1576,
            warning_count=1,
        )
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme.count(README_START) == 1
    assert readme.count(README_END) == 1


def test_create_docs_rejects_test_leakage_policy(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    evaluation = _evaluation()
    evaluation["evaluation_policy"]["test_result_used_for_model_selection"] = True
    (tmp_path / EVALUATION_ARTIFACT).write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not be used"):
        create_day12_docs(
            project_root=tmp_path,
            targeted_test_count=84,
            regression_test_count=1576,
            warning_count=1,
        )


def test_create_docs_rejects_missing_figure_and_invalid_count(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / REQUIRED_FIGURES[0]).unlink()
    with pytest.raises(FileNotFoundError, match="Figure"):
        create_day12_docs(
            project_root=tmp_path,
            targeted_test_count=84,
            regression_test_count=1576,
            warning_count=1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        create_day12_docs(
            project_root=tmp_path,
            targeted_test_count=-1,
            regression_test_count=1576,
            warning_count=1,
        )
