"""OpenCV 각 처리 단계와 정량 지표를 PNG Figure로 저장한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .image_conversion import bgr_to_rgb
from .metrics import OpenCVAnalysisMetrics
from .pipeline import OpenCVPipelineResult


PIPELINE_PANEL_TITLES: tuple[str, ...] = (
    "Original",
    "Grayscale",
    "CLAHE",
    "Gaussian Blur",
    "Canny Edge",
    "Adaptive Threshold",
    "Morphology",
    "Contour Candidates",
)

CONTOUR_PANEL_TITLES: tuple[str, ...] = (
    "Original",
    "Morphology Mask",
    "Contour Candidates",
)


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must use the .png extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig: plt.Figure, output_path: Path, *, dpi: int) -> Path:
    if dpi <= 0:
        plt.close(fig)
        raise ValueError("dpi must be greater than 0")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_pipeline_overview_figure(
    result: OpenCVPipelineResult,
    output_path: str | Path,
    *,
    title: str = "OpenCV Image Analysis Pipeline",
    dpi: int = 150,
) -> Path:
    """8개 핵심 처리 단계를 2×4 패널로 저장한다."""
    path = _prepare_output_path(output_path)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

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

    for axis, panel_title, (image, cmap) in zip(
        axes.flat,
        PIPELINE_PANEL_TITLES,
        panels,
        strict=True,
    ):
        axis.imshow(image, cmap=cmap, vmin=0, vmax=255)
        axis.set_title(panel_title)
        axis.axis("off")

    return _save_and_close(fig, path, dpi=dpi)


def save_histogram_and_metrics_figure(
    result: OpenCVPipelineResult,
    metrics: OpenCVAnalysisMetrics,
    output_path: str | Path,
    *,
    title: str = "Histogram and OpenCV Metrics",
    dpi: int = 150,
) -> Path:
    """Grayscale histogram과 핵심 보조 통계를 한 Figure에 저장한다."""
    path = _prepare_output_path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    axes[0].plot(range(256), result.histogram)
    axes[0].set_title("Grayscale Histogram")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Pixel Count")
    axes[0].set_xlim(0, 255)
    axes[0].grid(alpha=0.25)

    metric_lines = (
        f"Size: {metrics.width} × {metrics.height} × {metrics.channels}",
        f"Gray min/max: {metrics.grayscale_min} / {metrics.grayscale_max}",
        f"Mean brightness: {metrics.mean_brightness:.4f}",
        f"Brightness std: {metrics.brightness_standard_deviation:.4f}",
        f"Histogram peak: {metrics.histogram_peak}",
        f"Otsu threshold: {metrics.otsu_threshold:.4f}",
        f"Edge pixel ratio: {metrics.edge_pixel_ratio:.6f}",
        f"Threshold foreground ratio: {metrics.threshold_foreground_ratio:.6f}",
        f"Contour count: {metrics.contour_count}",
        f"Largest contour area ratio: {metrics.largest_contour_area_ratio:.6f}",
        f"Average contour area ratio: {metrics.average_contour_area_ratio:.6f}",
    )
    axes[1].axis("off")
    axes[1].set_title("Metrics")
    axes[1].text(
        0.02,
        0.98,
        "\n".join(metric_lines),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.5,
    )

    return _save_and_close(fig, path, dpi=dpi)


def save_contour_analysis_figure(
    result: OpenCVPipelineResult,
    metrics: OpenCVAnalysisMetrics,
    output_path: str | Path,
    *,
    title: str = "Threshold-based Contour Candidate Analysis",
    dpi: int = 150,
) -> Path:
    """Contour 후보가 어떤 이진 Mask에서 만들어졌는지 함께 보여준다."""
    path = _prepare_output_path(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(
        f"{title} | count={metrics.contour_count}, "
        f"largest_ratio={metrics.largest_contour_area_ratio:.6f}",
        fontsize=15,
    )

    panels = (
        (bgr_to_rgb(result.original_bgr), None),
        (result.morphology, "gray"),
        (bgr_to_rgb(result.contour_overlay_bgr), None),
    )

    for axis, panel_title, (image, cmap) in zip(
        axes,
        CONTOUR_PANEL_TITLES,
        panels,
        strict=True,
    ):
        axis.imshow(image, cmap=cmap, vmin=0, vmax=255)
        axis.set_title(panel_title)
        axis.axis("off")

    return _save_and_close(fig, path, dpi=dpi)
