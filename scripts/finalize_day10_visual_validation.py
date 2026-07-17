"""Day 10 Figure 육안 검증 결과를 Artifact로 확정한다.

이 Script는 사용자가 세 Figure를 직접 확인한 뒤 실행한다. Figure가 존재하고
Pillow로 열리는지 다시 검사하며, 각 육안 검증 상태를 JSON으로 기록한다.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from PIL import Image, UnidentifiedImageError


DEFAULT_ANALYSIS_PATH = (
    "reports/artifacts/day10_opencv_image_analysis.json"
)
DEFAULT_PIPELINE_FIGURE_PATH = (
    "reports/figures/day10_opencv_pipeline_overview.png"
)
DEFAULT_HISTOGRAM_FIGURE_PATH = (
    "reports/figures/day10_opencv_histogram_and_metrics.png"
)
DEFAULT_CONTOUR_FIGURE_PATH = (
    "reports/figures/day10_opencv_contour_analysis.png"
)
DEFAULT_OUTPUT_PATH = (
    "reports/artifacts/day10_opencv_visual_validation.json"
)
ALLOWED_STATUSES = {"pass", "fail"}


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _inspect_png(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else 0,
        "pillow_decode": False,
        "format": None,
        "width": None,
        "height": None,
    }
    if not path.is_file():
        return record

    try:
        with Image.open(path) as image:
            image.load()
            record.update(
                {
                    "pillow_decode": True,
                    "format": str(image.format or ""),
                    "width": int(image.width),
                    "height": int(image.height),
                }
            )
    except (OSError, UnidentifiedImageError) as error:
        record["decode_error"] = repr(error)
    return record


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
            temp_path = Path(file.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]

    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--analysis-path", default=DEFAULT_ANALYSIS_PATH)
    parser.add_argument(
        "--pipeline-figure-path",
        default=DEFAULT_PIPELINE_FIGURE_PATH,
    )
    parser.add_argument(
        "--histogram-figure-path",
        default=DEFAULT_HISTOGRAM_FIGURE_PATH,
    )
    parser.add_argument(
        "--contour-figure-path",
        default=DEFAULT_CONTOUR_FIGURE_PATH,
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--pipeline-overview-status",
        required=True,
        choices=sorted(ALLOWED_STATUSES),
    )
    parser.add_argument(
        "--histogram-metrics-status",
        required=True,
        choices=sorted(ALLOWED_STATUSES),
    )
    parser.add_argument(
        "--contour-analysis-status",
        required=True,
        choices=sorted(ALLOWED_STATUSES),
    )
    parser.add_argument(
        "--contour-disclaimer-status",
        required=True,
        choices=sorted(ALLOWED_STATUSES),
        help=(
            "Pass only when the figure clearly presents contours as "
            "threshold/morphology candidates rather than ground truth."
        ),
    )
    parser.add_argument("--notes", default="")
    return parser.parse_args(arguments)


def finalize(arguments: argparse.Namespace) -> dict[str, Any]:
    project_root = arguments.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root not found: {project_root}")

    analysis_path = _resolve(project_root, arguments.analysis_path)
    if not analysis_path.is_file():
        raise FileNotFoundError(f"analysis artifact not found: {analysis_path}")

    try:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("analysis artifact is not valid JSON") from error

    if analysis.get("day") != 10 or analysis.get("sample_count") != 3:
        raise ValueError("analysis artifact must be a Day 10 result with 3 samples")

    figure_paths = {
        "pipeline_overview": _resolve(
            project_root,
            arguments.pipeline_figure_path,
        ),
        "histogram_and_metrics": _resolve(
            project_root,
            arguments.histogram_figure_path,
        ),
        "contour_analysis": _resolve(
            project_root,
            arguments.contour_figure_path,
        ),
    }
    figure_checks = {
        name: _inspect_png(path)
        for name, path in figure_paths.items()
    }

    for record in figure_checks.values():
        path = Path(record["path"])
        record["path"] = _relative_or_absolute(path, project_root)

    automated_checks_passed = all(
        record["exists"]
        and record["size_bytes"] > 0
        and record["pillow_decode"]
        and record["format"] == "PNG"
        and isinstance(record["width"], int)
        and record["width"] > 0
        and isinstance(record["height"], int)
        and record["height"] > 0
        for record in figure_checks.values()
    )

    manual_checks = {
        "pipeline_overview": arguments.pipeline_overview_status,
        "histogram_and_metrics": arguments.histogram_metrics_status,
        "contour_analysis": arguments.contour_analysis_status,
        "contour_candidate_disclaimer": arguments.contour_disclaimer_status,
    }
    manual_checks_passed = all(
        status == "pass" for status in manual_checks.values()
    )
    validation_passed = automated_checks_passed and manual_checks_passed

    output_path = _resolve(project_root, arguments.output_path)
    payload = {
        "schema_version": "1.0",
        "project_name": "Manufacturing Vision Defect Analysis System",
        "project_name_ko": "제조 비전 결함 분석 시스템",
        "day": 10,
        "title": "OpenCV Visual Validation",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_artifact": _relative_or_absolute(
            analysis_path,
            project_root,
        ),
        "automated_figure_checks": figure_checks,
        "automated_checks_passed": automated_checks_passed,
        "manual_checks": manual_checks,
        "manual_checks_passed": manual_checks_passed,
        "validation_passed": validation_passed,
        "notes": arguments.notes,
        "interpretation": (
            "Contour overlays were reviewed as threshold/morphology-based "
            "candidates, not defect ground truth or detection predictions."
        ),
    }
    _atomic_write(output_path, payload)
    return payload


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    payload = finalize(parsed)

    print("=" * 100)
    print("DAY 10 - OPENCV VISUAL VALIDATION")
    print("=" * 100)
    print(f"Automated checks passed : {payload['automated_checks_passed']}")
    print(f"Manual checks passed    : {payload['manual_checks_passed']}")
    print(f"Validation passed       : {payload['validation_passed']}")
    print()
    for name, record in payload["automated_figure_checks"].items():
        print(
            f"{name:<24}: {record['path']} | "
            f"{record['width']}x{record['height']} | "
            f"{record['size_bytes']} bytes"
        )
    print()
    if payload["validation_passed"]:
        print("[PASS] Day 10 visual validation finalized")
        return 0

    print("[FAIL] One or more Day 10 visual checks failed")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
