"""Tests for scripts.create_day10_docs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_day10_docs import (
    README_END_MARKER,
    README_START_MARKER,
    build_day10_report,
    build_readme_section,
    create_day10_documentation,
    load_documentation_inputs,
    main,
    update_readme_content,
)


def _sample(
    sample_id: str,
    semantic_role: str,
    filename: str,
    mean: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "dataset_name": "Dataset",
        "semantic_role": semantic_role,
        "class_name": "class",
        "image_path": f"data/{filename}",
        "filename": filename,
        "file_size_bytes": 100,
        "sha256": "a" * 64,
        "source_image": {
            "format": "JPEG",
            "mode": "RGB",
            "width": 200,
            "height": 200,
        },
        "metrics": {
            "width": 200,
            "height": 200,
            "channels": 3,
            "grayscale_min": 0,
            "grayscale_max": 255,
            "mean_brightness": mean,
            "brightness_standard_deviation": 20.0,
            "histogram_peak": 100,
            "histogram_peak_count": 500,
            "otsu_threshold": 120.0,
            "edge_pixel_ratio": 0.10,
            "threshold_foreground_ratio": 0.20,
            "contour_count": 3,
            "largest_contour_area_ratio": 0.05,
            "average_contour_area_ratio": 0.02,
        },
        "interpretation": "candidate only",
    }


def _write_fixture_project(
    root: Path,
    *,
    validation_passed: bool = True,
) -> None:
    (root / "reports" / "artifacts").mkdir(parents=True)
    (root / "reports" / "figures").mkdir(parents=True)

    figure_paths = {
        "pipeline_overview_figure": (
            "reports/figures/day10_opencv_pipeline_overview.png"
        ),
        "histogram_and_metrics_figure": (
            "reports/figures/day10_opencv_histogram_and_metrics.png"
        ),
        "contour_analysis_figure": (
            "reports/figures/day10_opencv_contour_analysis.png"
        ),
    }
    for value in figure_paths.values():
        path = root / value
        path.write_bytes(b"fake-png-bytes")

    analysis = {
        "schema_version": "1.0",
        "project_name": "Manufacturing Vision Defect Analysis System",
        "project_name_ko": "제조 비전 결함 분석 시스템",
        "day": 10,
        "title": "OpenCV Image Analysis Pipeline",
        "generated_at_utc": "2026-07-17T00:00:00+00:00",
        "dependency_versions": {
            "opencv_python": "4.13.0.92",
            "cv2": "4.13.0",
            "numpy": "2.4.6",
            "pillow": "12.3.0",
            "matplotlib": "3.11.0",
        },
        "config": {
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": [8, 8],
            "gaussian_kernel_size": [5, 5],
            "canny_low_threshold": 50,
            "canny_high_threshold": 150,
            "adaptive_threshold_block_size": 11,
            "morphology_kernel_size": [3, 3],
            "min_contour_area_ratio": 0.0005,
        },
        "sample_count": 3,
        "samples": [
            _sample(
                "casting_normal",
                "Casting NORMAL",
                "normal.jpeg",
                100.0,
            ),
            _sample(
                "casting_defect",
                "Casting DEFECT",
                "defect.jpeg",
                110.0,
            ),
            _sample(
                "neu_det_crazing",
                "NEU-DET Defect Image",
                "crazing_1.jpg",
                120.0,
            ),
        ],
        "artifacts": {
            "analysis_json": (
                "reports/artifacts/day10_opencv_image_analysis.json"
            ),
            **figure_paths,
        },
    }
    validation = {
        "schema_version": "1.0",
        "project_name": "Manufacturing Vision Defect Analysis System",
        "project_name_ko": "제조 비전 결함 분석 시스템",
        "day": 10,
        "title": "OpenCV Visual Validation",
        "automated_checks_passed": validation_passed,
        "manual_checks": {
            "pipeline_overview": (
                "pass" if validation_passed else "fail"
            ),
            "histogram_and_metrics": (
                "pass" if validation_passed else "fail"
            ),
            "contour_analysis": (
                "pass" if validation_passed else "fail"
            ),
            "contour_candidate_disclaimer": (
                "pass" if validation_passed else "fail"
            ),
        },
        "manual_checks_passed": validation_passed,
        "validation_passed": validation_passed,
        "analysis_artifact": (
            "reports/artifacts/day10_opencv_image_analysis.json"
        ),
    }

    (root / "reports" / "artifacts" /
     "day10_opencv_image_analysis.json").write_text(
        json.dumps(analysis),
        encoding="utf-8",
    )
    (root / "reports" / "artifacts" /
     "day10_opencv_visual_validation.json").write_text(
        json.dumps(validation),
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Project\n", encoding="utf-8")


def _load_inputs(root: Path):
    return load_documentation_inputs(
        project_root=root,
        regression_test_count=1430,
        warning_count=1,
        regression_runtime_seconds=90.32,
        day10_test_count=62,
    )


def test_load_inputs_rejects_missing_analysis_artifact(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="analysis artifact"):
        _load_inputs(tmp_path)


def test_load_inputs_rejects_failed_visual_validation(
    tmp_path: Path,
) -> None:
    _write_fixture_project(tmp_path, validation_passed=False)

    with pytest.raises(ValueError, match="requires"):
        _load_inputs(tmp_path)


def test_create_documentation_creates_report_and_readme_section(
    tmp_path: Path,
) -> None:
    _write_fixture_project(tmp_path)
    inputs = _load_inputs(tmp_path)

    report_path, readme_path = create_day10_documentation(inputs)

    assert report_path.is_file()
    assert readme_path.is_file()
    readme = readme_path.read_text(encoding="utf-8")
    assert README_START_MARKER in readme
    assert README_END_MARKER in readme


def test_create_documentation_is_idempotent_for_readme_markers(
    tmp_path: Path,
) -> None:
    _write_fixture_project(tmp_path)
    inputs = _load_inputs(tmp_path)

    create_day10_documentation(inputs)
    first = (tmp_path / "README.md").read_text(encoding="utf-8")
    create_day10_documentation(inputs)
    second = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert first == second
    assert second.count(README_START_MARKER) == 1
    assert second.count(README_END_MARKER) == 1


def test_update_readme_rejects_incomplete_marker_pair() -> None:
    readme = f"# Project\n\n{README_START_MARKER}\n"
    section = (
        f"{README_START_MARKER}\nDay 10\n{README_END_MARKER}"
    )

    with pytest.raises(ValueError, match="markers"):
        update_readme_content(readme, section)


def test_report_contains_samples_metrics_and_interpretation_policy(
    tmp_path: Path,
) -> None:
    _write_fixture_project(tmp_path)
    report = build_day10_report(_load_inputs(tmp_path))

    assert "Casting NORMAL" in report
    assert "Casting DEFECT" in report
    assert "NEU-DET Defect Image" in report
    assert "Edge Pixel Ratio" in report
    assert "Ground Truth" in report
    assert "Bounding Box" in report


def test_report_contains_actual_regression_result(
    tmp_path: Path,
) -> None:
    _write_fixture_project(tmp_path)
    report = build_day10_report(_load_inputs(tmp_path))

    assert "1430 passed" in report
    assert "Warnings             : 1" in report
    assert "90.32 seconds" in report
    assert "62 passed" in report


def test_readme_section_contains_scope_report_and_disclaimer(
    tmp_path: Path,
) -> None:
    _write_fixture_project(tmp_path)
    section = build_readme_section(_load_inputs(tmp_path))

    assert "OpenCV 기반" in section
    assert "reports/day10_opencv_image_analysis_pipeline_summary.md" in section
    assert "1430 passed, 1 warning" in section
    assert "실제 결함 Ground Truth" in section


def test_main_creates_outputs_and_prints_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_fixture_project(tmp_path)

    result = main(
        [
            "--project-root",
            str(tmp_path),
            "--regression-test-count",
            "1430",
            "--warning-count",
            "1",
            "--regression-runtime-seconds",
            "90.32",
            "--day10-test-count",
            "62",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "[PASS] Day 10 report created" in output
    assert (tmp_path / "reports" /
            "day10_opencv_image_analysis_pipeline_summary.md").is_file()


def test_load_inputs_rejects_invalid_test_counts(
    tmp_path: Path,
) -> None:
    _write_fixture_project(tmp_path)

    with pytest.raises(ValueError, match="regression_test_count"):
        load_documentation_inputs(
            project_root=tmp_path,
            regression_test_count=0,
            warning_count=1,
            regression_runtime_seconds=90.32,
        )

    with pytest.raises(ValueError, match="warning_count"):
        load_documentation_inputs(
            project_root=tmp_path,
            regression_test_count=1430,
            warning_count=-1,
            regression_runtime_seconds=90.32,
        )
