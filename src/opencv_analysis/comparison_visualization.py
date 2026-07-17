"""Casting NORMAL·DEFECT와 NEU-DET 샘플을 한 Figure에서 비교한다.

서로 다른 Dataset의 값을 성능 우열로 비교하지 않는다. 동일 OpenCV 파이프라인이
두 종류의 제조 이미지에서 어떤 명암·경계·형태 결과를 만드는지 보여주기 위한
시각화다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .image_conversion import bgr_to_rgb
from .sample_analysis import AnalyzedImageSample


DAY10_PIPELINE_COLUMN_TITLES: tuple[str, ...] = (
    "Original",
    "Grayscale",
    "CLAHE",
    "Gaussian Blur",
    "Canny Edge",
    "Adaptive Threshold",
    "Morphology",
    "Contour Candidates",
)

DAY10_CONTOUR_COLUMN_TITLES: tuple[str, ...] = (
    "Original",
    "Morphology Mask",
    "Contour Candidates",
)


def _prepare_samples(
    samples: Sequence[AnalyzedImageSample],
) -> tuple[AnalyzedImageSample, ...]:
    resolved = tuple(samples)
    if not resolved:
        raise ValueError("samples must contain at least one analyzed image")
    if not all(isinstance(sample, AnalyzedImageSample) for sample in resolved):
        raise TypeError("every item in samples must be an AnalyzedImageSample")

    sample_ids = [sample.spec.sample_id for sample in resolved]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample ids must be unique")
    return resolved


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must use the .png extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(
    figure: plt.Figure,
    path: Path,
    *,
    dpi: int,
) -> Path:
    if isinstance(dpi, bool) or not isinstance(dpi, int):
        plt.close(figure)
        raise TypeError("dpi must be an integer")
    if dpi <= 0:
        plt.close(figure)
        raise ValueError("dpi must be greater than 0")

    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _row_label(sample: AnalyzedImageSample) -> str:
    return (
        f"{sample.spec.semantic_role}\n"
        f"{sample.spec.class_name}\n"
        f"{sample.absolute_path.name}"
    )


def save_day10_pipeline_overview(
    samples: Sequence[AnalyzedImageSample],
    output_path: str | Path,
    *,
    dpi: int = 140,
) -> Path:
    """샘플별 8단계 결과를 행 단위로 배치한다."""
    resolved = _prepare_samples(samples)
    path = _prepare_output_path(output_path)

    figure, axes = plt.subplots(
        len(resolved),
        len(DAY10_PIPELINE_COLUMN_TITLES),
        figsize=(24, 3.8 * len(resolved)),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        "Day 10 — OpenCV Image Analysis Pipeline\n"
        "Brightness, edge and morphology-assisted analysis",
        fontsize=17,
    )

    for row_index, sample in enumerate(resolved):
        result = sample.pipeline_result
        panels = (
            (bgr_to_rgb(result.original_bgr), None),
            (result.grayscale, "gray"),
            (result.clahe, "gray"),
            (result.blurred, "gray"),
            (result.edges, "gray"),
            (result.adaptive_threshold, "gray"),
            (result.morphology, "gray"),
            (bgr_to_rgb(result.contour_overlay_bgr), None),
        )

        for column_index, (image, cmap) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap=cmap, vmin=0, vmax=255)
            if row_index == 0:
                axis.set_title(DAY10_PIPELINE_COLUMN_TITLES[column_index])
            axis.axis("off")

        axes[row_index, 0].set_ylabel(
            _row_label(sample),
            rotation=0,
            ha="right",
            va="center",
            labelpad=70,
            fontsize=9,
        )

    return _save_and_close(figure, path, dpi=dpi)


def save_day10_histogram_and_metrics(
    samples: Sequence[AnalyzedImageSample],
    output_path: str | Path,
    *,
    dpi: int = 140,
) -> Path:
    """샘플별 Grayscale Histogram과 핵심 지표를 나란히 표시한다."""
    resolved = _prepare_samples(samples)
    path = _prepare_output_path(output_path)

    figure, axes = plt.subplots(
        len(resolved),
        2,
        figsize=(15, 5.0 * len(resolved)),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        "Day 10 — Grayscale Histogram and OpenCV Metrics\n"
        "Dataset meanings differ; values are not model-performance comparisons.",
        fontsize=16,
    )

    for row_index, sample in enumerate(resolved):
        result = sample.pipeline_result
        metrics = sample.metrics

        histogram_axis = axes[row_index, 0]
        histogram_axis.plot(range(256), result.histogram)
        histogram_axis.set_xlim(0, 255)
        histogram_axis.set_xlabel("Grayscale intensity")
        histogram_axis.set_ylabel("Pixel count")
        histogram_axis.set_title(
            f"{sample.spec.semantic_role} — Grayscale Histogram"
        )
        histogram_axis.grid(alpha=0.25)

        metric_lines = (
            f"Dataset: {sample.spec.dataset_name}",
            f"Role / class: {sample.spec.semantic_role} / {sample.spec.class_name}",
            f"File: {sample.absolute_path.name}",
            f"Size: {metrics.width} × {metrics.height} × {metrics.channels}",
            f"Gray min / max: {metrics.grayscale_min} / {metrics.grayscale_max}",
            f"Mean brightness: {metrics.mean_brightness:.4f}",
            f"Brightness standard deviation: "
            f"{metrics.brightness_standard_deviation:.4f}",
            f"Histogram peak: {metrics.histogram_peak} "
            f"({metrics.histogram_peak_count} pixels)",
            f"Otsu threshold: {metrics.otsu_threshold:.4f}",
            f"Edge pixel ratio: {metrics.edge_pixel_ratio:.6f}",
            f"Threshold foreground ratio: "
            f"{metrics.threshold_foreground_ratio:.6f}",
            f"Contour candidate count: {metrics.contour_count}",
            f"Largest contour area ratio: "
            f"{metrics.largest_contour_area_ratio:.6f}",
            f"Average contour area ratio: "
            f"{metrics.average_contour_area_ratio:.6f}",
        )
        metrics_axis = axes[row_index, 1]
        metrics_axis.axis("off")
        metrics_axis.set_title(
            f"{sample.spec.semantic_role} — Metrics"
        )
        metrics_axis.text(
            0.02,
            0.98,
            "\n".join(metric_lines),
            transform=metrics_axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            linespacing=1.35,
        )

    return _save_and_close(figure, path, dpi=dpi)


def save_day10_contour_analysis(
    samples: Sequence[AnalyzedImageSample],
    output_path: str | Path,
    *,
    dpi: int = 140,
) -> Path:
    """Contour 후보의 원본·생성 Mask·Overlay를 함께 표시한다."""
    resolved = _prepare_samples(samples)
    path = _prepare_output_path(output_path)

    figure, axes = plt.subplots(
        len(resolved),
        len(DAY10_CONTOUR_COLUMN_TITLES),
        figsize=(15, 4.7 * len(resolved)),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        "Day 10 — Threshold-based Contour Candidate Analysis\n"
        "Contours are not defect ground truth or object-detection boxes.",
        fontsize=16,
    )

    for row_index, sample in enumerate(resolved):
        result = sample.pipeline_result
        metrics = sample.metrics
        panels = (
            (bgr_to_rgb(result.original_bgr), None),
            (result.morphology, "gray"),
            (bgr_to_rgb(result.contour_overlay_bgr), None),
        )

        for column_index, (image, cmap) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap=cmap, vmin=0, vmax=255)
            if row_index == 0:
                axis.set_title(DAY10_CONTOUR_COLUMN_TITLES[column_index])
            axis.axis("off")

        axes[row_index, 0].set_ylabel(
            _row_label(sample),
            rotation=0,
            ha="right",
            va="center",
            labelpad=70,
            fontsize=9,
        )
        axes[row_index, 2].text(
            0.02,
            0.02,
            (
                f"count={metrics.contour_count}\n"
                f"largest ratio={metrics.largest_contour_area_ratio:.6f}"
            ),
            transform=axes[row_index, 2].transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    return _save_and_close(figure, path, dpi=dpi)
