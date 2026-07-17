from __future__ import annotations

from src.dashboard.api_client import (
    DashboardApiError,
    DashboardHealth,
    DashboardPrediction,
)
from src.dashboard.session_state import (
    ANALYSIS_EXECUTED_KEY,
    HEALTH_ERROR_KEY,
    HEALTH_KEY,
    LAST_UPLOAD_FILENAME_KEY,
    PREDICTION_ERROR_KEY,
    PREDICTION_KEY,
    begin_analysis,
    initialize_dashboard_state,
    reset_prediction_state,
    save_health,
    save_prediction,
    save_prediction_error,
)


def _prediction() -> DashboardPrediction:
    return DashboardPrediction(
        prediction=1,
        prediction_class_name="DEFECT",
        defect_probability=0.9,
        normal_probability=0.1,
        raw_logit=2.1972246,
        classification_threshold=0.5,
        model_name="ResNet18Transfer",
        model_version="resnet18_transfer_best",
        positive_class="DEFECT",
        original_filename="sample.png",
        content_type="image/png",
        image_width=300,
        image_height=300,
        image_mode="RGB",
        inference_time_ms=10.0,
    )


def test_initialization_creates_defaults_without_overwriting_existing_values() -> None:
    state: dict[str, object] = {LAST_UPLOAD_FILENAME_KEY: "keep.png"}

    initialize_dashboard_state(state)
    initialize_dashboard_state(state)

    assert state[LAST_UPLOAD_FILENAME_KEY] == "keep.png"
    assert state[PREDICTION_KEY] is None
    assert state[ANALYSIS_EXECUTED_KEY] is False


def test_begin_analysis_clears_previous_result_and_error() -> None:
    state: dict[str, object] = {
        PREDICTION_KEY: _prediction(),
        PREDICTION_ERROR_KEY: DashboardApiError(code="INVALID_IMAGE"),
    }

    begin_analysis(state, filename="new.png")

    assert state[PREDICTION_KEY] is None
    assert state[PREDICTION_ERROR_KEY] is None
    assert state[LAST_UPLOAD_FILENAME_KEY] == "new.png"
    assert state[ANALYSIS_EXECUTED_KEY] is True


def test_prediction_and_error_are_mutually_exclusive() -> None:
    state: dict[str, object] = {}
    prediction = _prediction()

    save_prediction(state, prediction=prediction)
    assert state[PREDICTION_KEY] == prediction
    assert state[PREDICTION_ERROR_KEY] is None

    error = DashboardApiError(code="MODEL_NOT_READY")
    save_prediction_error(state, error=error)
    assert state[PREDICTION_KEY] is None
    assert state[PREDICTION_ERROR_KEY] == error


def test_health_state_is_saved_separately() -> None:
    state: dict[str, object] = {}
    health = DashboardHealth(
        status="ok",
        service="Manufacturing Vision Defect Analysis System",
        model_loaded=True,
        model_name="ResNet18Transfer",
        device="cpu",
    )

    save_health(state, health=health, error=None)

    assert state[HEALTH_KEY] == health
    assert state[HEALTH_ERROR_KEY] is None


def test_reset_clears_only_prediction_related_state() -> None:
    state: dict[str, object] = {HEALTH_KEY: "keep"}
    begin_analysis(state, filename="sample.png")
    save_prediction(state, prediction=_prediction())

    reset_prediction_state(state)

    assert state[HEALTH_KEY] == "keep"
    assert state[PREDICTION_KEY] is None
    assert state[PREDICTION_ERROR_KEY] is None
    assert state[LAST_UPLOAD_FILENAME_KEY] is None
    assert state[ANALYSIS_EXECUTED_KEY] is False
