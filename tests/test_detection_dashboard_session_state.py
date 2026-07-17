from __future__ import annotations

from src.dashboard.detection_api_client import (
    DashboardDetectionPrediction,
    DetectionDashboardApiError,
)
from src.dashboard.detection_session_state import (
    DETECTION_ANALYSIS_EXECUTED_KEY,
    DETECTION_ERROR_KEY,
    DETECTION_LAST_SCORE_THRESHOLD_KEY,
    DETECTION_LAST_UPLOAD_FILENAME_KEY,
    DETECTION_RESULT_KEY,
    begin_detection_analysis,
    initialize_detection_state,
    save_detection_error,
    save_detection_result,
)


def result() -> DashboardDetectionPrediction:
    return DashboardDetectionPrediction(
        detections=(),
        detection_count=0,
        score_threshold=0.5,
        iou_threshold=0.5,
        model_name="model",
        model_version="version",
        architecture="architecture",
        device="cpu",
        checkpoint_epoch=3,
        checkpoint_metric_name="map_50",
        checkpoint_metric_value=0.677418,
        original_filename="sample.png",
        content_type="image/png",
        image_width=12,
        image_height=10,
        image_mode="RGB",
        model_input_mode="RGB",
        inference_time_ms=1.0,
    )


def test_initialize_detection_state_is_idempotent() -> None:
    state = {}
    initialize_detection_state(state)
    state[DETECTION_RESULT_KEY] = "keep"

    initialize_detection_state(state)

    assert state[DETECTION_RESULT_KEY] == "keep"
    assert state[DETECTION_ANALYSIS_EXECUTED_KEY] is False


def test_begin_detection_analysis_clears_previous_state() -> None:
    state = {}
    initialize_detection_state(state)
    state[DETECTION_RESULT_KEY] = result()
    state[DETECTION_ERROR_KEY] = DetectionDashboardApiError(
        code="API_TIMEOUT"
    )

    begin_detection_analysis(
        state,
        filename="sample.png",
        score_threshold=0.65,
    )

    assert state[DETECTION_ANALYSIS_EXECUTED_KEY] is True
    assert state[DETECTION_RESULT_KEY] is None
    assert state[DETECTION_ERROR_KEY] is None
    assert state[DETECTION_LAST_UPLOAD_FILENAME_KEY] == "sample.png"
    assert state[DETECTION_LAST_SCORE_THRESHOLD_KEY] == 0.65


def test_save_detection_result_clears_error() -> None:
    state = {}
    initialize_detection_state(state)
    value = result()

    save_detection_result(state, result=value)

    assert state[DETECTION_RESULT_KEY] is value
    assert state[DETECTION_ERROR_KEY] is None


def test_save_detection_error_clears_result() -> None:
    state = {}
    initialize_detection_state(state)
    state[DETECTION_RESULT_KEY] = result()
    error = DetectionDashboardApiError(
        code="API_TIMEOUT"
    )

    save_detection_error(state, error=error)

    assert state[DETECTION_RESULT_KEY] is None
    assert state[DETECTION_ERROR_KEY] is error
