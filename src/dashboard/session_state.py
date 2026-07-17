"""Streamlit 재실행에서도 마지막 분석 결과를 유지하는 최소 Session State."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from src.dashboard.api_client import (
    DashboardApiError,
    DashboardHealth,
    DashboardPrediction,
)

HEALTH_KEY = "dashboard_health"
HEALTH_ERROR_KEY = "dashboard_health_error"
PREDICTION_KEY = "dashboard_prediction"
PREDICTION_ERROR_KEY = "dashboard_prediction_error"
LAST_UPLOAD_FILENAME_KEY = "dashboard_last_upload_filename"
ANALYSIS_EXECUTED_KEY = "dashboard_analysis_executed"


DEFAULT_STATE: dict[str, object] = {
    HEALTH_KEY: None,
    HEALTH_ERROR_KEY: None,
    PREDICTION_KEY: None,
    PREDICTION_ERROR_KEY: None,
    LAST_UPLOAD_FILENAME_KEY: None,
    ANALYSIS_EXECUTED_KEY: False,
}


def initialize_dashboard_state(state: MutableMapping[str, Any]) -> None:
    """기존 값은 유지하고 누락된 초기 Key만 생성한다."""

    for key, value in DEFAULT_STATE.items():
        state.setdefault(key, value)


def begin_analysis(
    state: MutableMapping[str, Any],
    *,
    filename: str,
) -> None:
    """새 분석 시작 시 이전 Prediction과 오류를 정리한다."""

    initialize_dashboard_state(state)
    state[PREDICTION_KEY] = None
    state[PREDICTION_ERROR_KEY] = None
    state[LAST_UPLOAD_FILENAME_KEY] = filename
    state[ANALYSIS_EXECUTED_KEY] = True


def save_health(
    state: MutableMapping[str, Any],
    *,
    health: DashboardHealth | None,
    error: DashboardApiError | None,
) -> None:
    initialize_dashboard_state(state)
    state[HEALTH_KEY] = health
    state[HEALTH_ERROR_KEY] = error


def save_prediction(
    state: MutableMapping[str, Any],
    *,
    prediction: DashboardPrediction,
) -> None:
    initialize_dashboard_state(state)
    state[PREDICTION_KEY] = prediction
    state[PREDICTION_ERROR_KEY] = None


def save_prediction_error(
    state: MutableMapping[str, Any],
    *,
    error: DashboardApiError,
) -> None:
    initialize_dashboard_state(state)
    state[PREDICTION_KEY] = None
    state[PREDICTION_ERROR_KEY] = error


def reset_prediction_state(state: MutableMapping[str, Any]) -> None:
    initialize_dashboard_state(state)
    state[PREDICTION_KEY] = None
    state[PREDICTION_ERROR_KEY] = None
    state[LAST_UPLOAD_FILENAME_KEY] = None
    state[ANALYSIS_EXECUTED_KEY] = False
