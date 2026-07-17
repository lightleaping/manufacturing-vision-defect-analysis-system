"""OpenCV 기반 명암·경계·형태 특성 보조 분석 파이프라인."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import OpenCVAnalysisConfig
from .image_conversion import to_grayscale


@dataclass(frozen=True, slots=True)
class OpenCVPipelineResult:
    """각 OpenCV 처리 단계의 결과.

    배열은 Figure 생성에 사용하고, 통계는 metrics 모듈에서 별도로 계산한다.
    contours는 threshold·morphology 기반 후보 형태이며 실제 결함 정답이 아니다.
    """

    original_bgr: np.ndarray
    grayscale: np.ndarray
    histogram: np.ndarray
    clahe: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    adaptive_threshold: np.ndarray
    morphology: np.ndarray
    contours: tuple[np.ndarray, ...]
    contour_areas: tuple[float, ...]
    contour_overlay_bgr: np.ndarray
    otsu_threshold: float


def _copy_readonly(array: np.ndarray) -> np.ndarray:
    """외부 입력과 메모리를 공유하지 않는 읽기 전용 결과를 만든다."""
    copied = np.ascontiguousarray(array.copy())
    copied.setflags(write=False)
    return copied


def _find_filtered_contours(
    binary: np.ndarray,
    *,
    min_area: float,
    max_contours: int,
) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
    """작은 Noise contour를 면적으로 제거하고 큰 순서로 정렬한다."""
    found, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filtered: list[tuple[np.ndarray, float]] = []
    for contour in found:
        area = float(cv2.contourArea(contour))
        if area >= min_area:
            filtered.append((contour.copy(), area))

    filtered.sort(key=lambda item: item[1], reverse=True)
    filtered = filtered[:max_contours]

    contours = tuple(item[0] for item in filtered)
    areas = tuple(item[1] for item in filtered)
    return contours, areas


def run_opencv_pipeline(
    image_bgr: np.ndarray,
    config: OpenCVAnalysisConfig | None = None,
) -> OpenCVPipelineResult:
    """BGR 또는 Grayscale 이미지에 결정론적 OpenCV 파이프라인을 적용한다.

    입력 배열을 수정하지 않으며 동일 입력과 동일 Config에는 동일 결과를 반환한다.
    """
    resolved_config = config or OpenCVAnalysisConfig()

    # 입력 검증과 Grayscale 변환은 공통 변환 함수에서 수행한다.
    grayscale = to_grayscale(image_bgr)

    # 원본 표시와 contour overlay를 위해 항상 3채널 BGR을 만든다.
    if image_bgr.ndim == 2:
        original_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = image_bgr.copy()

    histogram = cv2.calcHist([grayscale], [0], None, [256], [0, 256]).reshape(-1)
    histogram = histogram.astype(np.float64, copy=False)

    clahe_operator = cv2.createCLAHE(
        clipLimit=float(resolved_config.clahe_clip_limit),
        tileGridSize=resolved_config.clahe_tile_grid_size,
    )
    clahe = clahe_operator.apply(grayscale)

    blurred = cv2.GaussianBlur(
        clahe,
        resolved_config.gaussian_kernel_size,
        sigmaX=float(resolved_config.gaussian_sigma_x),
    )

    edges = cv2.Canny(
        blurred,
        threshold1=resolved_config.canny_low_threshold,
        threshold2=resolved_config.canny_high_threshold,
    )

    threshold_type = (
        cv2.THRESH_BINARY_INV
        if resolved_config.adaptive_threshold_invert
        else cv2.THRESH_BINARY
    )
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        resolved_config.adaptive_threshold_block_size,
        float(resolved_config.adaptive_threshold_c),
    )

    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        resolved_config.morphology_kernel_size,
    )
    morphology = adaptive_threshold.copy()
    if resolved_config.morphology_open_iterations > 0:
        morphology = cv2.morphologyEx(
            morphology,
            cv2.MORPH_OPEN,
            morphology_kernel,
            iterations=resolved_config.morphology_open_iterations,
        )
    if resolved_config.morphology_close_iterations > 0:
        morphology = cv2.morphologyEx(
            morphology,
            cv2.MORPH_CLOSE,
            morphology_kernel,
            iterations=resolved_config.morphology_close_iterations,
        )

    image_area = float(grayscale.shape[0] * grayscale.shape[1])
    min_contour_area = image_area * resolved_config.min_contour_area_ratio
    contours, contour_areas = _find_filtered_contours(
        morphology,
        min_area=min_contour_area,
        max_contours=resolved_config.max_contours,
    )

    contour_overlay = original_bgr.copy()
    if contours:
        # 초록색 선은 후보 contour를 시각적으로 구분하기 위한 표시일 뿐이다.
        cv2.drawContours(
            contour_overlay,
            list(contours),
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=resolved_config.contour_line_thickness,
        )

    otsu_threshold, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return OpenCVPipelineResult(
        original_bgr=_copy_readonly(original_bgr),
        grayscale=_copy_readonly(grayscale),
        histogram=_copy_readonly(histogram),
        clahe=_copy_readonly(clahe),
        blurred=_copy_readonly(blurred),
        edges=_copy_readonly(edges),
        adaptive_threshold=_copy_readonly(adaptive_threshold),
        morphology=_copy_readonly(morphology),
        contours=tuple(_copy_readonly(contour) for contour in contours),
        contour_areas=contour_areas,
        contour_overlay_bgr=_copy_readonly(contour_overlay),
        otsu_threshold=float(otsu_threshold),
    )
