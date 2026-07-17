from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_day10_opencv_analysis import main


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((36, 48, 3), value, dtype=np.uint8)
    array[8:28, 14:34] = (230, 180, 50)
    Image.fromarray(array, mode="RGB").save(path)


def test_run_day10_opencv_analysis_creates_artifact_and_figures(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "fixtures" / "normal.png", 50)
    _write_image(tmp_path / "fixtures" / "defect.jpg", 100)
    _write_image(tmp_path / "fixtures" / "neu.jpeg", 150)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--casting-normal-path",
            "fixtures/normal.png",
            "--casting-defect-path",
            "fixtures/defect.jpg",
            "--neu-det-path",
            "fixtures/neu.jpeg",
            "--figure-dpi",
            "50",
        ]
    )

    assert exit_code == 0

    artifact_path = (
        tmp_path / "reports/artifacts/day10_opencv_image_analysis.json"
    )
    pipeline_path = (
        tmp_path / "reports/figures/day10_opencv_pipeline_overview.png"
    )
    histogram_path = (
        tmp_path
        / "reports/figures/day10_opencv_histogram_and_metrics.png"
    )
    contour_path = (
        tmp_path / "reports/figures/day10_opencv_contour_analysis.png"
    )

    assert artifact_path.is_file()
    for path in (pipeline_path, histogram_path, contour_path):
        assert path.is_file()
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            image.load()
            assert image.format == "PNG"

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["day"] == 10
    assert payload["sample_count"] == 3
    assert [sample["sample_id"] for sample in payload["samples"]] == [
        "casting_normal",
        "casting_defect",
        "neu_det_crazing",
    ]
    assert payload["config"]["canny_low_threshold"] == 50
    assert payload["dependency_versions"]["cv2"]
    assert "not defect ground truth" in (
        payload["interpretation_policy"]["contour_warning"]
    )


def test_run_day10_opencv_analysis_supports_custom_outputs(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "fixtures" / "normal.png", 50)
    _write_image(tmp_path / "fixtures" / "defect.png", 100)
    _write_image(tmp_path / "fixtures" / "neu.png", 150)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--casting-normal-path",
            "fixtures/normal.png",
            "--casting-defect-path",
            "fixtures/defect.png",
            "--neu-det-path",
            "fixtures/neu.png",
            "--artifact-path",
            "custom/result.json",
            "--pipeline-figure-path",
            "custom/pipeline.png",
            "--histogram-figure-path",
            "custom/histogram.png",
            "--contour-figure-path",
            "custom/contour.png",
            "--figure-dpi",
            "50",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "custom/result.json").is_file()
    assert (tmp_path / "custom/pipeline.png").is_file()
    assert (tmp_path / "custom/histogram.png").is_file()
    assert (tmp_path / "custom/contour.png").is_file()
