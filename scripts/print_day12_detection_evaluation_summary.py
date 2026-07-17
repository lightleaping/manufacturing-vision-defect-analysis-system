"""Day 12 Detection 평가 결과를 터미널 폭에 의존하지 않고 출력한다.

PowerShell Format-Table은 창 폭이 좁으면 F1·Mean IoU·AP 열을 잘라 보일 수 있다.
이 Script는 각 수치를 고정 폭 문자열로 출력해 잘림과 열 혼동을 방지한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_EVALUATION_ARTIFACT = Path(
    "reports/artifacts/day12_detection_evaluation.json"
)
DEFAULT_FAILURE_ARTIFACT = Path(
    "reports/artifacts/day12_detection_failure_analysis.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact root must be an object: {path}.")
    return payload


def _metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def print_day12_detection_evaluation_summary(*, project_root: Path) -> None:
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    root = project_root.resolve()
    evaluation = _read_json(root / DEFAULT_EVALUATION_ARTIFACT)
    failure = _read_json(root / DEFAULT_FAILURE_ARTIFACT)

    print("=" * 126)
    print("DAY 12 DETECTION FINAL EVALUATION SUMMARY")
    print("=" * 126)
    header = (
        f"{'Split':<12}{'TP':>6}{'FP':>6}{'FN':>6}"
        f"{'Precision':>13}{'Recall':>13}{'F1':>13}"
        f"{'Mean IoU':>13}{'mAP@0.50':>13}"
    )
    print(header)
    print("-" * len(header))
    for split in ("validation", "test"):
        overall = evaluation[split]["metrics"]["overall"]
        print(
            f"{split:<12}"
            f"{int(overall['tp']):>6}"
            f"{int(overall['fp']):>6}"
            f"{int(overall['fn']):>6}"
            f"{_metric(overall['precision']):>13}"
            f"{_metric(overall['recall']):>13}"
            f"{_metric(overall['f1']):>13}"
            f"{_metric(overall['mean_matched_iou']):>13}"
            f"{_metric(overall['map_50']):>13}"
        )

    print("\n[TEST CLASS METRICS]")
    class_header = (
        f"{'Class':<20}{'TP':>5}{'FP':>5}{'FN':>5}{'GT':>6}"
        f"{'Precision':>12}{'Recall':>12}{'F1':>12}"
        f"{'Mean IoU':>12}{'AP@0.50':>12}{'mAP@.50:.95':>15}"
    )
    print(class_header)
    print("-" * len(class_header))
    sweep = evaluation["test_iou_sweep"]["class_map_50_95"]
    for class_name, values in evaluation["test"]["metrics"]["class_metrics"].items():
        print(
            f"{class_name:<20}"
            f"{int(values['tp']):>5}"
            f"{int(values['fp']):>5}"
            f"{int(values['fn']):>5}"
            f"{int(values['ground_truth_count']):>6}"
            f"{_metric(values['precision']):>12}"
            f"{_metric(values['recall']):>12}"
            f"{_metric(values['f1']):>12}"
            f"{_metric(values['mean_matched_iou']):>12}"
            f"{_metric(values['ap_50']):>12}"
            f"{_metric(sweep.get(class_name)):>15}"
        )

    summary = failure["analysis"]["summary"]
    print("\n[FAILURE SUMMARY]")
    print(f"Test images          : {summary['image_count']}")
    print(f"Images with failures : {summary['images_with_failures']}")
    print(f"Failure events       : {summary['event_count']}")
    for name, count in sorted(
        summary["counts"].items(),
        key=lambda item: int(item[1]),
        reverse=True,
    ):
        print(f"{name:<36}: {int(count):>4}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print Day 12 Detection metrics without PowerShell truncation."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    print_day12_detection_evaluation_summary(project_root=arguments.project_root)


if __name__ == "__main__":
    main()
