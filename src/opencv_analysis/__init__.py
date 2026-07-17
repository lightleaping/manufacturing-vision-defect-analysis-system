"""OpenCV 기반 이미지 명암·경계·형태 특성 보조 분석 패키지.

이 패키지는 Classification 또는 Object Detection 모델을 대체하지 않는다.
Contour는 전처리 결과에서 계산한 후보 형태일 뿐 실제 결함 Ground Truth가 아니다.
"""

from .config import OpenCVAnalysisConfig
from .image_conversion import (
    bgr_to_rgb,
    pillow_to_bgr,
    pillow_to_grayscale,
    rgb_to_bgr,
    to_grayscale,
)
from .metrics import OpenCVAnalysisMetrics, calculate_metrics
from .pipeline import OpenCVPipelineResult, run_opencv_pipeline
from .visualization import (
    CONTOUR_PANEL_TITLES,
    PIPELINE_PANEL_TITLES,
    save_contour_analysis_figure,
    save_histogram_and_metrics_figure,
    save_pipeline_overview_figure,
)

__all__ = [
    "OpenCVAnalysisConfig",
    "OpenCVAnalysisMetrics",
    "OpenCVPipelineResult",
    "calculate_metrics",
    "run_opencv_pipeline",
    "pillow_to_bgr",
    "pillow_to_grayscale",
    "bgr_to_rgb",
    "rgb_to_bgr",
    "to_grayscale",
    "PIPELINE_PANEL_TITLES",
    "CONTOUR_PANEL_TITLES",
    "save_pipeline_overview_figure",
    "save_histogram_and_metrics_figure",
    "save_contour_analysis_figure",
]
