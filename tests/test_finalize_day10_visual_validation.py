from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.finalize_day10_visual_validation import main


def _write_analysis(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "day": 10,
                "sample_count": 3,
                "title": "OpenCV Image Analysis Pipeline",
            }
        ),
        encoding="utf-8",
    )


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), (120, 120, 120)).save(path)


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--project-root",
        str(tmp_path),
        "--pipeline-overview-status",
        "pass",
        "--histogram-metrics-status",
        "pass",
        "--contour-analysis-status",
        "pass",
        "--contour-disclaimer-status",
        "pass",
        "--notes",
        "Panels and labels inspected.",
    ]


def test_finalize_day10_visual_validation_passes(
    tmp_path: Path,
) -> None:
    _write_analysis(
        tmp_path / "reports/artifacts/day10_opencv_image_analysis.json"
    )
    _write_png(
        tmp_path / "reports/figures/day10_opencv_pipeline_overview.png"
    )
    _write_png(
        tmp_path
        / "reports/figures/day10_opencv_histogram_and_metrics.png"
    )
    _write_png(
        tmp_path / "reports/figures/day10_opencv_contour_analysis.png"
    )

    exit_code = main(_base_args(tmp_path))

    assert exit_code == 0
    output = (
        tmp_path / "reports/artifacts/day10_opencv_visual_validation.json"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["automated_checks_passed"] is True
    assert payload["manual_checks_passed"] is True
    assert payload["validation_passed"] is True
    assert payload["notes"] == "Panels and labels inspected."


def test_finalize_day10_visual_validation_records_manual_failure(
    tmp_path: Path,
) -> None:
    _write_analysis(
        tmp_path / "reports/artifacts/day10_opencv_image_analysis.json"
    )
    _write_png(
        tmp_path / "reports/figures/day10_opencv_pipeline_overview.png"
    )
    _write_png(
        tmp_path
        / "reports/figures/day10_opencv_histogram_and_metrics.png"
    )
    _write_png(
        tmp_path / "reports/figures/day10_opencv_contour_analysis.png"
    )

    args = _base_args(tmp_path)
    index = args.index("--contour-analysis-status")
    args[index + 1] = "fail"

    exit_code = main(args)

    assert exit_code == 2
    output = (
        tmp_path / "reports/artifacts/day10_opencv_visual_validation.json"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["automated_checks_passed"] is True
    assert payload["manual_checks_passed"] is False
    assert payload["validation_passed"] is False
