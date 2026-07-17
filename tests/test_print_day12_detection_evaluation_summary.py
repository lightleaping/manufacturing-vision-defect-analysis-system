from __future__ import annotations

import json
from pathlib import Path

from scripts.print_day12_detection_evaluation_summary import (
    print_day12_detection_evaluation_summary,
)


def test_summary_prints_full_f1_and_class_names(
    tmp_path: Path,
    capsys,
) -> None:
    artifact_dir = tmp_path / "reports" / "artifacts"
    artifact_dir.mkdir(parents=True)
    overall = {
        "tp": 2,
        "fp": 1,
        "fn": 3,
        "precision": 0.666666,
        "recall": 0.4,
        "f1": 0.5,
        "mean_matched_iou": 0.75,
        "map_50": 0.6,
    }
    class_metric = {
        "tp": 2,
        "fp": 1,
        "fn": 3,
        "ground_truth_count": 5,
        "precision": 0.666666,
        "recall": 0.4,
        "f1": 0.5,
        "mean_matched_iou": 0.75,
        "ap_50": 0.6,
    }
    evaluation = {
        "validation": {"metrics": {"overall": overall}},
        "test": {
            "metrics": {
                "overall": overall,
                "class_metrics": {"rolled_in_scale": class_metric},
            }
        },
        "test_iou_sweep": {"class_map_50_95": {"rolled_in_scale": 0.3}},
    }
    failure = {
        "analysis": {
            "summary": {
                "image_count": 2,
                "images_with_failures": 1,
                "event_count": 3,
                "counts": {"false_negative": 3},
            }
        }
    }
    (artifact_dir / "day12_detection_evaluation.json").write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )
    (artifact_dir / "day12_detection_failure_analysis.json").write_text(
        json.dumps(failure),
        encoding="utf-8",
    )

    print_day12_detection_evaluation_summary(project_root=tmp_path)

    output = capsys.readouterr().out
    assert "rolled_in_scale" in output
    assert "0.500000" in output
    assert "mAP@.50:.95" in output
