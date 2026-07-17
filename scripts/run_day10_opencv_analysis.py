"""Day 10 실제 이미지 3장을 OpenCV 파이프라인으로 분석한다.

기본 대상:
1. 기존 Day 7·8 검증에 사용한 Casting NORMAL Test 이미지
2. 기존 Day 7·8 검증에 사용한 Casting DEFECT Test 이미지
3. NEU-DET training split의 crazing 샘플

전체 Dataset을 순회하거나 성능 평가하지 않는다.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import cv2
import matplotlib
import numpy as np
from PIL import __version__ as pillow_version

from src.opencv_analysis.comparison_visualization import (
    save_day10_contour_analysis,
    save_day10_histogram_and_metrics,
    save_day10_pipeline_overview,
)
from src.opencv_analysis.config import OpenCVAnalysisConfig
from src.opencv_analysis.sample_analysis import (
    AnalyzedImageSample,
    ImageSampleSpec,
    analyze_image_samples,
)


DEFAULT_CASTING_NORMAL_PATH = (
    "data/raw/casting_product_images/casting_data/casting_data/"
    "test/ok_front/cast_ok_0_7631.jpeg"
)
DEFAULT_CASTING_DEFECT_PATH = (
    "data/raw/casting_product_images/casting_data/casting_data/"
    "test/def_front/cast_def_0_1414.jpeg"
)
DEFAULT_NEU_DET_PATH = (
    "data/raw/neu_det/NEU-DET/train/images/crazing/crazing_1.jpg"
)

DEFAULT_ARTIFACT_PATH = (
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


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _resolve_output_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
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

    return path


def _default_sample_specs(arguments: argparse.Namespace) -> tuple[ImageSampleSpec, ...]:
    return (
        ImageSampleSpec(
            sample_id="casting_normal",
            dataset_name="Casting Product Image Data for Quality Inspection",
            semantic_role="Casting NORMAL",
            class_name="NORMAL",
            relative_path=arguments.casting_normal_path,
        ),
        ImageSampleSpec(
            sample_id="casting_defect",
            dataset_name="Casting Product Image Data for Quality Inspection",
            semantic_role="Casting DEFECT",
            class_name="DEFECT",
            relative_path=arguments.casting_defect_path,
        ),
        ImageSampleSpec(
            sample_id="neu_det_crazing",
            dataset_name="NEU Surface Defect Database (NEU-DET)",
            semantic_role="NEU-DET Defect Image",
            class_name="crazing",
            relative_path=arguments.neu_det_path,
        ),
    )


def _build_artifact(
    *,
    project_root: Path,
    config: OpenCVAnalysisConfig,
    samples: Sequence[AnalyzedImageSample],
    artifact_path: Path,
    pipeline_figure_path: Path,
    histogram_figure_path: Path,
    contour_figure_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "project_name": "Manufacturing Vision Defect Analysis System",
        "project_name_ko": "제조 비전 결함 분석 시스템",
        "day": 10,
        "title": "OpenCV Image Analysis Pipeline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": (
            "OpenCV-based brightness, edge and morphology-assisted image "
            "analysis. This is not classification or object detection."
        ),
        "interpretation_policy": {
            "classification": (
                "Determines whether the whole image is NORMAL or DEFECT."
            ),
            "opencv_analysis": (
                "Provides deterministic brightness, edge, threshold and "
                "morphology calculations for human inspection."
            ),
            "object_detection": (
                "A trained model predicts defect class, location and confidence."
            ),
            "contour_warning": (
                "Contours are threshold/morphology candidates, not defect "
                "ground truth or object-detection bounding boxes."
            ),
            "dataset_comparison_warning": (
                "Casting and NEU-DET have different meanings; their metric "
                "values are not model-performance comparisons."
            ),
        },
        "dependency_versions": {
            "opencv_python": _distribution_version("opencv-python"),
            "cv2": cv2.__version__,
            "numpy": np.__version__,
            "pillow": pillow_version,
            "matplotlib": matplotlib.__version__,
        },
        "config": config.to_dict(),
        "sample_count": len(samples),
        "samples": [sample.to_artifact_record() for sample in samples],
        "artifacts": {
            "analysis_json": _project_relative(artifact_path, project_root),
            "pipeline_overview_figure": _project_relative(
                pipeline_figure_path,
                project_root,
            ),
            "histogram_and_metrics_figure": _project_relative(
                histogram_figure_path,
                project_root,
            ),
            "contour_analysis_figure": _project_relative(
                contour_figure_path,
                project_root,
            ),
        },
    }


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="Project root path.",
    )
    parser.add_argument(
        "--casting-normal-path",
        default=DEFAULT_CASTING_NORMAL_PATH,
    )
    parser.add_argument(
        "--casting-defect-path",
        default=DEFAULT_CASTING_DEFECT_PATH,
    )
    parser.add_argument(
        "--neu-det-path",
        default=DEFAULT_NEU_DET_PATH,
    )
    parser.add_argument(
        "--artifact-path",
        default=DEFAULT_ARTIFACT_PATH,
    )
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
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=140,
    )
    return parser.parse_args(arguments)


def run_analysis(arguments: argparse.Namespace) -> dict[str, Any]:
    project_root = arguments.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root not found: {project_root}")
    if arguments.figure_dpi <= 0:
        raise ValueError("--figure-dpi must be greater than 0")

    artifact_path = _resolve_output_path(
        project_root,
        arguments.artifact_path,
    )
    pipeline_figure_path = _resolve_output_path(
        project_root,
        arguments.pipeline_figure_path,
    )
    histogram_figure_path = _resolve_output_path(
        project_root,
        arguments.histogram_figure_path,
    )
    contour_figure_path = _resolve_output_path(
        project_root,
        arguments.contour_figure_path,
    )

    config = OpenCVAnalysisConfig()
    samples = analyze_image_samples(
        project_root,
        _default_sample_specs(arguments),
        config,
    )

    save_day10_pipeline_overview(
        samples,
        pipeline_figure_path,
        dpi=arguments.figure_dpi,
    )
    save_day10_histogram_and_metrics(
        samples,
        histogram_figure_path,
        dpi=arguments.figure_dpi,
    )
    save_day10_contour_analysis(
        samples,
        contour_figure_path,
        dpi=arguments.figure_dpi,
    )

    artifact = _build_artifact(
        project_root=project_root,
        config=config,
        samples=samples,
        artifact_path=artifact_path,
        pipeline_figure_path=pipeline_figure_path,
        histogram_figure_path=histogram_figure_path,
        contour_figure_path=contour_figure_path,
    )
    _atomic_write_json(artifact_path, artifact)
    return artifact


def _print_result(artifact: dict[str, Any], project_root: Path) -> None:
    print("=" * 100)
    print("DAY 10 - OPENCV REAL IMAGE ANALYSIS")
    print("=" * 100)
    print(f"Project root : {project_root}")
    print(f"Sample count : {artifact['sample_count']}")
    print()

    for sample in artifact["samples"]:
        metrics = sample["metrics"]
        print(f"[{sample['semantic_role']}]")
        print(f"File                         : {sample['image_path']}")
        print(
            "Size                         : "
            f"{metrics['width']} x {metrics['height']} x "
            f"{metrics['channels']}"
        )
        print(
            "Mean brightness              : "
            f"{metrics['mean_brightness']:.6f}"
        )
        print(
            "Brightness standard deviation: "
            f"{metrics['brightness_standard_deviation']:.6f}"
        )
        print(
            "Edge pixel ratio             : "
            f"{metrics['edge_pixel_ratio']:.6f}"
        )
        print(
            "Threshold foreground ratio   : "
            f"{metrics['threshold_foreground_ratio']:.6f}"
        )
        print(
            "Contour candidate count      : "
            f"{metrics['contour_count']}"
        )
        print(
            "Largest contour area ratio   : "
            f"{metrics['largest_contour_area_ratio']:.6f}"
        )
        print()

    print("[ARTIFACTS]")
    for name, path in artifact["artifacts"].items():
        print(f"{name:<30}: {path}")
    print()
    print(
        "[NOTICE] Contours are threshold/morphology candidates, "
        "not defect ground truth or object-detection boxes."
    )
    print("[PASS] Day 10 real-image OpenCV analysis artifacts created")


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    artifact = run_analysis(parsed)
    _print_result(artifact, parsed.project_root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
