"""Day 13 Detection Streamlit 페이지 전용 Session State."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from src.dashboard.detection_api_client import (
    DashboardDetectionPrediction,
    DetectionDashboardApiError,
)


DETECTION_ANALYSIS_EXECUTED_KEY = "detection_analysis_executed"
DETECTION_RESULT_KEY = "detection_result"
DETECTION_ERROR_KEY = "detection_error"
DETECTION_LAST_UPLOAD_FILENAME_KEY = "detection_last_upload_filename"
DETECTION_LAST_SCORE_THRESHOLD_KEY = "detection_last_score_threshold"


def initialize_detection_state(
    state: MutableMapping[str, Any],
) -> None:
    """Detection 페이지가 사용하는 Key를 최초 한 번만 생성한다."""

    defaults: dict[str, object] = {
        DETECTION_ANALYSIS_EXECUTED_KEY: False,
        DETECTION_RESULT_KEY: None,
        DETECTION_ERROR_KEY: None,
        DETECTION_LAST_UPLOAD_FILENAME_KEY: "",
        DETECTION_LAST_SCORE_THRESHOLD_KEY: 0.5,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def begin_detection_analysis(
    state: MutableMapping[str, Any],
    *,
    filename: str,
    score_threshold: float,
) -> None:
    """새 요청 직전에 이전 결과와 오류를 제거한다."""

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must not be empty")

    state[DETECTION_ANALYSIS_EXECUTED_KEY] = True
    state[DETECTION_RESULT_KEY] = None
    state[DETECTION_ERROR_KEY] = None
    state[DETECTION_LAST_UPLOAD_FILENAME_KEY] = filename.strip()
    state[DETECTION_LAST_SCORE_THRESHOLD_KEY] = float(score_threshold)


def save_detection_result(
    state: MutableMapping[str, Any],
    *,
    result: DashboardDetectionPrediction,
) -> None:
    if not isinstance(result, DashboardDetectionPrediction):
        raise TypeError(
            "result must be DashboardDetectionPrediction"
        )
    state[DETECTION_RESULT_KEY] = result
    state[DETECTION_ERROR_KEY] = None


def save_detection_error(
    state: MutableMapping[str, Any],
    *,
    error: DetectionDashboardApiError,
) -> None:
    if not isinstance(error, DetectionDashboardApiError):
        raise TypeError(
            "error must be DetectionDashboardApiError"
        )
    state[DETECTION_RESULT_KEY] = None
    state[DETECTION_ERROR_KEY] = error
