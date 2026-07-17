"""OpenCV 파이프라인 결과에서 JSON 직렬화 가능한 정량 지표 계산."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .pipeline import OpenCVPipelineResult


@dataclass(frozen=True, slots=True)
class OpenCVAnalysisMetrics:
    """명암·경계·Threshold·Contour 후보에 대한 보조 통계."""

    width: int
    height: int
    channels: int
    grayscale_min: int
    grayscale_max: int
    mean_brightness: float
    brightness_standard_deviation: float
    histogram_peak: int
    histogram_peak_count: int
    otsu_threshold: float
    edge_pixel_ratio: float
    threshold_foreground_ratio: float
    contour_count: int
    largest_contour_area_ratio: float
    average_contour_area_ratio: float

    def to_dict(self, *, round_digits: int | None = 6) -> dict[str, Any]:
        """Artifact와 API에서 사용할 수 있는 기본 Python 타입 dict를 반환한다."""
        result = asdict(self)
        if round_digits is not None:
            for key, value in result.items():
                if isinstance(value, float):
                    result[key] = round(value, round_digits)
        return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else 0.0


def calculate_metrics(result: OpenCVPipelineResult) -> OpenCVAnalysisMetrics:
    """파이프라인 결과를 사람이 비교 가능한 보조 지표로 요약한다."""
    if not isinstance(result, OpenCVPipelineResult):
        raise TypeError("result must be an OpenCVPipelineResult")

    height, width = result.grayscale.shape
    image_area = float(width * height)

    histogram_peak = int(np.argmax(result.histogram))
    histogram_peak_count = int(result.histogram[histogram_peak])

    edge_pixels = int(np.count_nonzero(result.edges))
    threshold_pixels = int(np.count_nonzero(result.morphology))

    contour_count = len(result.contour_areas)
    largest_contour_area = max(result.contour_areas, default=0.0)
    average_contour_area = (
        float(np.mean(result.contour_areas)) if result.contour_areas else 0.0
    )

    return OpenCVAnalysisMetrics(
        width=int(width),
        height=int(height),
        channels=int(result.original_bgr.shape[2]),
        grayscale_min=int(result.grayscale.min()),
        grayscale_max=int(result.grayscale.max()),
        mean_brightness=float(np.mean(result.grayscale, dtype=np.float64)),
        brightness_standard_deviation=float(
            np.std(result.grayscale, dtype=np.float64)
        ),
        histogram_peak=histogram_peak,
        histogram_peak_count=histogram_peak_count,
        otsu_threshold=float(result.otsu_threshold),
        edge_pixel_ratio=_safe_ratio(edge_pixels, image_area),
        threshold_foreground_ratio=_safe_ratio(threshold_pixels, image_area),
        contour_count=contour_count,
        largest_contour_area_ratio=_safe_ratio(largest_contour_area, image_area),
        average_contour_area_ratio=_safe_ratio(average_contour_area, image_area),
    )
